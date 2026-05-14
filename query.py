"""
File: query.py
Role: Inference & AB Test
Description:
    Method A (baseline) : ChromaDB retrieval → cross-encoder rerank
    Method B (proposed) : HyDE + hybrid retrieval → XGBoost rerank (cosine + BM25 + metadata)

    Method B enhancements over A:
      - HyDE (Hypothetical Document Embeddings): confidence-gated; local Ollama
        qwen2.5:3b generates a hypothetical prompt only when retrieval confidence
        is low (avg top-3 cosine < 0.68). Its embedding is averaged with the
        original query embedding to bridge the query–document vocabulary gap.
      - Hybrid retrieval: top-N vector results ∪ top-N/2 BM25 results,
        deduplicated, giving XGBoost a richer candidate pool.

    Imports from processing.py: build_bm25_index, build_category_index
    Imports from XGboost.py  : enrich_candidates, build_features
    Imports from rerank.py   : load_reranker, cross_encode
"""

import time
import numpy as np
import chromadb
import xgboost as xgb
import ollama
from sentence_transformers import SentenceTransformer

from rerank import load_reranker, cross_encode
from processing import build_bm25_index, build_category_index, _tokenize_with_bigrams
from XGboost import (
    enrich_candidates,
    build_features,
    EMBED_MODEL_NAME,
    DATASET_PATH,
    DB_PATH,
    COLLECTION_NAME,
    RERANK_MODEL_NAME,
    MODEL_SAVE_PATH,
    TITLE_VECS_PATH,
)
from dataload import load_data

TOP_K_RETRIEVE = 50
TOP_K_RETURN   = 10
MIN_COSINE     = 0.35   # strong semantic gate for vector candidates
BM25_MIN_COSINE = 0.20  # weak semantic floor for lexical-only candidates
MIN_BM25_REL_RANK = 0.70  # keep only high-ranking BM25 candidates

HYDE_MODEL       = "qwen2.5:3b"
HYDE_THRESHOLD   = 0.68   # avg top-3 cosine ≥ this → skip HyDE
_HYDE_PROMPT = (
    'A user is searching a prompt library with this query: "{query}"\n'
    "Write one example prompt that would perfectly satisfy this search. "
    "Output the prompt text only, no explanations.\n"
    "Prompt:"
)


def _generate_hyde_doc(query: str) -> str:
    """Generate a hypothetical matching prompt via local Ollama (qwen2.5:3b).
    Falls back to the original query if Ollama is unavailable."""
    try:
        resp = ollama.generate(
            model=HYDE_MODEL,
            prompt=_HYDE_PROMPT.format(query=query),
            options={"num_predict": 150, "temperature": 0.2},
        )
        text = resp.get("response", "").strip()
        return text if text else query
    except Exception:
        return query


# ---------------------------------------------------------------------------
# Search Engine
# ---------------------------------------------------------------------------
class SearchEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_collection(name=COLLECTION_NAME)

        print(f"Loading reranker for Method A: {RERANK_MODEL_NAME}...")
        self.reranker = load_reranker(RERANK_MODEL_NAME)

        print(f"Loading XGBoost model for Method B: {MODEL_SAVE_PATH}...")
        self.xgb_model = xgb.XGBRanker()
        self.xgb_model.load_model(MODEL_SAVE_PATH)

        print("Building BM25 index for Method B...")
        self.df = load_data(DATASET_PATH)
        self.bm25, self.id_to_idx = build_bm25_index(self.df)

        print("Building category index for category_match feature...")
        self.cat_embeddings = build_category_index(self.df, self.embedder)

        print("Loading precomputed title embeddings...")
        title_data = np.load(TITLE_VECS_PATH)
        self.title_vecs_dict = {
            str(id_): vec for id_, vec in zip(title_data["ids"], title_data["vecs"])
        }

        print("Search engine ready.\n")

    # -----------------------------------------------------------------------
    # Method A retrieval — pure vector search (clean baseline)
    # Returns (candidates, q_vec_np)
    # -----------------------------------------------------------------------
    def _retrieve(self, query: str, n: int = TOP_K_RETRIEVE):
        q_vec_np = self.embedder.encode([query], normalize_embeddings=True)[0]

        results = self.collection.query(
            query_embeddings=[q_vec_np.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        docs, metas, distances = (results["documents"][0],
                                  results["metadatas"][0],
                                  results["distances"][0])
        candidates = [
            {
                "content":      docs[i],
                "metadata":     metas[i],
                "cosine_score": 1.0 - distances[i],
            }
            for i in range(len(docs))
        ]
        return candidates, q_vec_np

    # -----------------------------------------------------------------------
    # Method B retrieval — HyDE + hybrid (vector ∪ BM25)
    # HyDE bridges the vocabulary gap for abstract queries ("sci-fi story",
    # "horror prompt") by generating a hypothetical document and averaging
    # its embedding with the original query vector.
    # BM25 adds candidates the vector search missed via exact/stem match.
    # Returns (candidates, q_vec_np) — q_vec_np is the ORIGINAL query vector,
    # kept unmodified so enrich_candidates computes cosine_title correctly.
    # -----------------------------------------------------------------------
    def _retrieve_enhanced(self, query: str, n: int = TOP_K_RETRIEVE):
        # 1. Original query embedding (used for cosine_title in enrich_candidates)
        q_vec_np = self.embedder.encode([query], normalize_embeddings=True)[0]

        # 2. Confidence gate: probe top-3 to decide whether HyDE is needed
        probe = self.collection.query(
            query_embeddings=[q_vec_np.tolist()],
            n_results=3,
            include=["distances"]
        )
        avg_cosine = float(np.mean([1.0 - d for d in probe["distances"][0]]))

        if avg_cosine >= HYDE_THRESHOLD:
            search_vec = q_vec_np  # precise query — skip HyDE
        else:
            # 3. HyDE: generate hypothetical doc and average embeddings
            hyde_doc = _generate_hyde_doc(query)
            if hyde_doc != query:
                hyde_vec = self.embedder.encode([hyde_doc], normalize_embeddings=True)[0]
                combined = q_vec_np + hyde_vec
                norm     = np.linalg.norm(combined)
                search_vec = (combined / norm) if norm > 0 else q_vec_np
            else:
                search_vec = q_vec_np

        # 4. Vector search with (optionally HyDE-enriched) query vector
        results = self.collection.query(
            query_embeddings=[search_vec.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        seen_ids   = set()
        candidates = []
        for i in range(len(results["documents"][0])):
            meta   = results["metadatas"][0][i]
            doc_id = str(meta.get("id", ""))
            seen_ids.add(doc_id)
            candidates.append({
                "content":      results["documents"][0][i],
                "metadata":     meta,
                "cosine_score": 1.0 - results["distances"][0][i],
                "retrieval_source": "vector",
            })

        # 5. BM25 hybrid: add top-N/2 BM25 hits not already retrieved
        q_tokens   = _tokenize_with_bigrams(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_bm25   = sorted(range(len(bm25_scores)),
                            key=lambda i: bm25_scores[i], reverse=True)[:n // 2]

        bm25_rows = []
        bm25_ids  = []
        for idx in top_bm25:
            row    = self.df.iloc[idx]
            doc_id = str(row["id"])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            bm25_rows.append(row)
            bm25_ids.append(doc_id)

        bm25_vecs_by_id = {}
        if bm25_ids:
            fetched = self.collection.get(ids=bm25_ids, include=["embeddings"])
            fetched_ids = fetched.get("ids", [])
            fetched_embeddings = fetched.get("embeddings")
            if fetched_embeddings is not None:
                for doc_id, vec in zip(fetched_ids, fetched_embeddings):
                    bm25_vecs_by_id[str(doc_id)] = np.asarray(vec, dtype=np.float32)

        for row, doc_id in zip(bm25_rows, bm25_ids):
            doc_vec = bm25_vecs_by_id.get(doc_id)
            cosine_score = 0.0
            if doc_vec is not None and doc_vec.shape == q_vec_np.shape:
                cosine_score = float(doc_vec @ q_vec_np)

            candidates.append({
                "content":  str(row.get("title", "")) + ". " + str(row.get("content", "")),
                "metadata": {
                    "id":               doc_id,
                    "title":            str(row.get("title", "")),
                    "category":         str(row.get("category", "")),
                    "likes":            int(row.get("likes", 0)),
                    "upvotes":          int(row.get("upvotes", 0)),
                    "downvotes":        int(row.get("downvotes", 0)),
                    "views":            int(row.get("views", 0)),
                    "uses":             int(row.get("uses", 0)),
                    "fork_count":       int(row.get("fork_count", 0)),
                    "author_reputation":int(row.get("author_reputation", 0)),
                },
                "cosine_score": cosine_score,
                "retrieval_source": "bm25",
            })

        return candidates, q_vec_np

    # -----------------------------------------------------------------------
    # Baseline: pure vector search — no reranker, no metadata weighting
    # -----------------------------------------------------------------------
    def search_baseline(self, query: str, top_k: int = TOP_K_RETURN) -> list:
        """ChromaDB top-k by cosine similarity only — no reranker."""
        candidates, _ = self._retrieve(query, n=top_k)
        return candidates

    # -----------------------------------------------------------------------
    # Method A: cross-encoder rerank  (baseline)
    # -----------------------------------------------------------------------
    def search_A(self, query: str, top_k: int = TOP_K_RETURN) -> list:
        """ChromaDB → cross-encoder rerank → top-k"""
        candidates, _ = self._retrieve(query)
        candidates = cross_encode(query, candidates, self.reranker)
        return candidates[:top_k]

    # -----------------------------------------------------------------------
    # Method B: HyDE + hybrid retrieval → XGBoost rerank
    # -----------------------------------------------------------------------
    def search_B(self, query: str, top_k: int = TOP_K_RETURN) -> list:
        """HyDE + hybrid retrieval → enrich → XGBoost rerank → top-k"""
        candidates, q_vec_np = self._retrieve_enhanced(query)

        # Enrich with BM25, retrieval_rank, cosine_title (same as training)
        enrich_candidates(candidates, query, q_vec_np,
                          self.bm25, self.id_to_idx, self.embedder,
                          self.cat_embeddings, title_vecs_dict=self.title_vecs_dict)

        candidates = [
            c for c in candidates
            if (
                c["cosine_score"] >= MIN_COSINE
                or (
                    c.get("bm25_score", 0.0) > 0
                    and c.get("bm25_rel_rank", 0.0) >= MIN_BM25_REL_RANK
                    and c["cosine_score"] >= BM25_MIN_COSINE
                )
            )
        ]
        if not candidates:
            return []

        features = build_features(candidates)
        scores   = self.xgb_model.predict(features)
        ranked   = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        results = []
        for c, score in ranked[:top_k]:
            c["xgb_score"] = float(score)
            results.append(c)
        return results

    # -----------------------------------------------------------------------
    # A/B/Baseline Test: three-way comparison with per-method latency
    # -----------------------------------------------------------------------
    def ab_test(self, query: str, top_k: int = 5):
        t0 = time.perf_counter(); results_base = self.search_baseline(query, top_k); t_base = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); results_a    = self.search_A(query, top_k);        t_a    = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); results_b    = self.search_B(query, top_k);        t_b    = (time.perf_counter() - t0) * 1000

        W = 114
        print(f"\n{'='*W}")
        print(f"Query: {query}")
        print(f"{'='*W}")
        print(f"{'Rank':<4}  {'Baseline — vector only':<28}  {'Method A — cross-encoder':<28}  {'Method B — XGBoost+Meta':<28}")
        print(f"{'-'*W}")

        for i in range(top_k):
            base = results_base[i] if i < len(results_base) else {}
            a    = results_a[i]    if i < len(results_a)    else {}
            b    = results_b[i]    if i < len(results_b)    else {}

            base_title = base.get("metadata", {}).get("title", "")[:24]
            a_title    = a.get("metadata", {}).get("title", "")[:24]
            b_title    = b.get("metadata", {}).get("title", "")[:24]
            base_score = base.get("cosine_score", 0.0)
            a_score    = a.get("reranker_score", 0.0)
            b_score    = b.get("xgb_score", 0.0)

            print(f"{i+1:<4}  {base_title:<24} ({base_score:.3f})  "
                  f"{a_title:<24} ({a_score:+.3f})  "
                  f"{b_title:<24} ({b_score:.4f})")

        print(f"{'-'*W}")
        print(f"Latency — Baseline: {t_base:5.0f} ms  |  Method A: {t_a:5.0f} ms  |  Method B: {t_b:5.0f} ms")
        print()
        return results_base, results_a, results_b

    # -----------------------------------------------------------------------
    # Latency benchmark: median and P95 across N queries
    # -----------------------------------------------------------------------
    def latency_benchmark(self, queries: list, top_k: int = 5, warmup: int = 1):
        """
        Run all queries through Baseline / Method A / Method B.
        Reports median and P95 latency (ms) for each method.
        """
        print(f"\nLatency benchmark — {len(queries)} queries (warmup={warmup})\n")

        # Warmup pass to avoid cold-start skew
        for q in queries[:warmup]:
            self.search_baseline(q, top_k)
            self.search_A(q, top_k)
            self.search_B(q, top_k)

        times = {"Baseline": [], "Method A": [], "Method B": []}
        for q in queries:
            t0 = time.perf_counter(); self.search_baseline(q, top_k); times["Baseline"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); self.search_A(q, top_k);        times["Method A"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); self.search_B(q, top_k);        times["Method B"].append((time.perf_counter() - t0) * 1000)

        print(f"{'Method':<12}  {'Median (ms)':>12}  {'P95 (ms)':>10}")
        print(f"{'-'*38}")
        for method, ms in times.items():
            arr = np.array(ms)
            print(f"{method:<12}  {np.median(arr):>12.0f}  {np.percentile(arr, 95):>10.0f}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = SearchEngine()

    test_queries = [
        # Creative writing
        "develop a serialized fantasy story arc with character motivations and episode outline",

        # Marketing / SEO / e-commerce
        "create an SEO audit checklist for an e-commerce product page",
        "write persuasive ad copy for a new skincare product launch",

        # Sales
        "write a cold call script for qualifying B2B sales leads",
        "create follow-up email templates after a trade show conversation",

        # Legal / compliance
        "review a vendor contract for legal risks and liability clauses",
        "draft a GDPR consent form for collecting customer data",

        # Customer support
        "write an empathetic customer support reply about a duplicate billing charge",
        "create an FAQ article explaining a return policy clearly",

        # Design / UX
        "generate an accessible color palette and typography system for a website",
        "explain form design best practices for better user experience",

        # Data analysis / SQL
        "explain SQL joins with simple examples for a beginner analyst",
        "clean messy survey data and summarize the main findings",

        # Translation / localization
        "translate a formal business email from English to Spanish with polite tone",
        "prepare source content for app internationalization and localization",

        # Finance / accounting
        "build a due diligence checklist for acquiring a small manufacturing company",
        "forecast monthly revenue for a subscription business model",

        # Coding / DevOps / security
        "refactor a React component and write Jest unit tests",
        "debug docker compose networking between backend and database services",
        "create a vulnerability assessment checklist for a small company network",
    ]

    for q in test_queries:
        engine.ab_test(q, top_k=5)

    engine.latency_benchmark(test_queries, top_k=5)
