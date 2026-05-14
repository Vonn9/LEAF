"""
File: XGboost.py
Role: XGBoost Learning-to-Rank — Offline Training
Description:
    1. Stratified-sample N titles as pseudo-queries (fixed by RANDOM_SEED)
       - Each category contributes at least MIN_PER_CATEGORY queries
       - Remaining budget filled proportionally from larger categories
       - Ensures all 420+ categories get training coverage
    2. For each query: ChromaDB top-K ∪ BM25 top-K/2 + enrich candidates
       (cosine_score, bm25_score, retrieval_rank, cosine_title,
        log_content_length, title_token_overlap)
    3. Cross-encoder → per-query min-max normalized labels (y)
    4. Train XGBRanker (rank:pairwise)
    5. Save model to xgb_model.json

Cache:
    After the expensive cross-encoder loop, X/y/qids are saved to
    CACHE_PATH. Re-running main() loads from cache and skips straight
    to XGBoost training (seconds instead of hours).
    Bump CACHE_VERSION when you change features or N_QUERIES to
    invalidate the old cache.

Split (70 / 15 / 15 by unique query ID):
    three_way_split() is shared with optimization.py.
    Train+val are used here for the final production model.
    Test is held out — only evaluated once in optimization.py.

Best params:
    If optimization.py has been run, best_params.json is loaded automatically.
    Otherwise falls back to DEFAULT_PARAMS.
"""

import os
import json
import random
import numpy as np
import xgboost as xgb
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dataload import load_data
from rerank import load_reranker, cross_encode
from processing import (
    _tokenize,
    _tokenize_with_bigrams,
    build_bm25_index,
    build_category_index,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Shipped artifacts (xgb_model.json / training_cache / best_params.json) live
# inside the src/ folder so they travel with the codebase. Inputs and
# regenerated artifacts (dataset, ChromaDB, title vectors) live one level up,
# at the project root, so they are not bundled into src.zip on submission.
_HERE = os.path.dirname(os.path.abspath(__file__))           # src/
_ROOT = os.path.dirname(_HERE)                                # parent of src/

DATASET_PATH      = os.path.join(_ROOT, "dataset.json")
DB_PATH           = os.path.join(_ROOT, "chroma_db")
TITLE_VECS_PATH   = os.path.join(_ROOT, "title_vecs.npz")

COLLECTION_NAME   = "prompt_collection"
EMBED_MODEL_NAME  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

MODEL_SAVE_PATH   = os.path.join(_HERE, "xgb_model.json")
BEST_PARAMS_PATH  = os.path.join(_HERE, "best_params.json")

N_QUERIES        = 1200
MIN_PER_CATEGORY = 3
TOP_K            = 50
RANDOM_SEED      = 42
QUERY_BATCH_SIZE = 32
CACHE_VERSION    = "v10"
CACHE_PATH       = os.path.join(_HERE, f"training_cache_{CACHE_VERSION}.npz")

VAL_RATIO  = 0.15
TEST_RATIO = 0.15

DEFAULT_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.0,
    "reg_lambda":       1.0,
}

# ---------------------------------------------------------------------------
# Features  (13 features)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "cosine_score",
    "bm25_score",
    "retrieval_rank",
    "bm25_rel_rank",
    "cosine_title",
    "log_content_length",
    "title_token_overlap",
    "log_upvotes",
    "vote_rate",
    "log_author_rep",
    "log_uses×cosine",
    "engagement_rate×cosine",
    "category_match",
]


# ---------------------------------------------------------------------------
# Train / Val / Test split  (shared with optimization.py)
# ---------------------------------------------------------------------------
def three_way_split(all_qids_arr):
    """
    Split unique query IDs into train / val / test masks (70 / 15 / 15).
    Uses RANDOM_SEED + 2 (distinct from other seeded operations).
    Returns (train_mask, val_mask, test_mask) as boolean arrays over all_qids_arr.
    """
    unique_q = np.unique(all_qids_arr)
    rng      = np.random.RandomState(RANDOM_SEED + 2)
    shuffled = unique_q[rng.permutation(len(unique_q))]

    n      = len(shuffled)
    n_test = max(1, int(round(n * TEST_RATIO)))
    n_val  = max(1, int(round(n * VAL_RATIO)))

    test_q = set(shuffled[:n_test].tolist())
    val_q  = set(shuffled[n_test : n_test + n_val].tolist())

    test_mask  = np.array([q in test_q for q in all_qids_arr])
    val_mask   = np.array([q in val_q  for q in all_qids_arr])
    train_mask = ~test_mask & ~val_mask

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Stratified query sampler
# ---------------------------------------------------------------------------
def stratified_sample_queries(df, n_total: int, min_per_cat: int, seed: int) -> list:
    """
    Sample pseudo-queries so every category gets at least min_per_cat titles,
    with remaining budget distributed proportionally.
    Returns a list of (title, doc_id) tuples (length ≤ n_total).
    """
    rng = random.Random(seed)

    cat_col = "category" if "category" in df.columns else None
    if cat_col is None:
        titles = df["title"].dropna().tolist()
        return rng.sample(titles, min(n_total, len(titles)))

    groups = {}
    for _, row in df.iterrows():
        cat    = str(row.get("category", "unknown") or "unknown")
        title  = row["title"]
        doc_id = str(row["id"])
        if isinstance(title, str) and title.strip():
            groups.setdefault(cat, []).append((title, doc_id))

    cats = list(groups.keys())

    # Phase 1: guarantee min_per_cat per category
    sampled = {}
    for cat, titles in groups.items():
        k = min(min_per_cat, len(titles))
        sampled[cat] = rng.sample(titles, k)

    n_floor = sum(len(v) for v in sampled.values())

    # Phase 2: fill remaining budget proportionally (larger categories first)
    remaining = n_total - n_floor
    if remaining > 0:
        sizes   = {cat: len(groups[cat]) - len(sampled[cat]) for cat in cats}
        total_r = sum(sizes.values())
        if total_r > 0:
            for cat in sorted(cats, key=lambda c: sizes[c], reverse=True):
                if remaining <= 0:
                    break
                pool    = [t for t in groups[cat] if t not in sampled[cat]]
                extra_n = max(0, round(remaining * sizes[cat] / total_r))
                extra_n = min(extra_n, len(pool), remaining)
                if extra_n > 0:
                    sampled[cat].extend(rng.sample(pool, extra_n))
                    remaining -= extra_n

    result = [pair for pairs in sampled.values() for pair in pairs]
    remaining = n_total - len(result)
    if remaining > 0:
        selected = set(result)
        leftovers = [
            pair
            for cat in cats
            for pair in groups[cat]
            if pair not in selected
        ]
        if leftovers:
            result.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    rng.shuffle(result)
    return result[:n_total]


# ---------------------------------------------------------------------------
# Candidate enrichment
# ---------------------------------------------------------------------------
def enrich_candidates(candidates, query_str, q_vec_np, bm25, id_to_idx, embedder,
                      cat_embeddings=None, title_vecs_dict=None):
    """
    Add bm25_score, retrieval_rank, cosine_title, log_content_length,
    title_token_overlap, category_match to each candidate dict in-place.

    MUST be called BEFORE cross_encode so that retrieval_rank reflects the
    original ChromaDB order (sorted by cosine similarity).
    """
    n = len(candidates)

    # 1. retrieval_rank
    for i, c in enumerate(candidates):
        c["retrieval_rank"] = 1.0 - i / n

    # 2. bm25_score + bm25_rel_rank
    q_tokens_bm25 = _tokenize_with_bigrams(query_str)
    q_token_set   = set(_tokenize(query_str))
    all_scores    = bm25.get_scores(q_tokens_bm25)
    for c in candidates:
        idx             = id_to_idx.get(str(c["metadata"].get("id", "")), 0)
        c["bm25_score"] = float(all_scores[idx])

    sorted_by_bm25 = sorted(range(n), key=lambda i: candidates[i]["bm25_score"], reverse=True)
    for rank, idx in enumerate(sorted_by_bm25):
        candidates[idx]["bm25_rel_rank"] = 1.0 - rank / max(n - 1, 1)

    # 3. cosine_title
    titles = [c["metadata"].get("title", "") or "" for c in candidates]
    if title_vecs_dict is not None:
        cosine_titles = np.array([
            title_vecs_dict.get(str(c["metadata"].get("id", "")), np.zeros_like(q_vec_np)) @ q_vec_np
            for c in candidates
        ], dtype=np.float32)
    else:
        cosine_titles = embedder.encode(titles, normalize_embeddings=True, show_progress_bar=False) @ q_vec_np

    # 4. remaining features (single pass)
    for i, c in enumerate(candidates):
        c["cosine_title"] = float(cosine_titles[i])

        content = c.get("content", "") or ""
        c["log_content_length"] = float(np.log1p(len(content)))

        title_tokens = set(_tokenize(titles[i]))
        union = q_token_set | title_tokens
        c["title_token_overlap"] = float(len(q_token_set & title_tokens) / len(union)) if union else 0.0

        if cat_embeddings is not None:
            cat_vec = cat_embeddings.get(c["metadata"].get("category", ""))
            c["category_match"] = float(q_vec_np @ cat_vec) if cat_vec is not None else 0.0
        else:
            c["category_match"] = 0.0


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------
def build_features(candidates: list) -> np.ndarray:
    """
    Build feature matrix from enriched candidate list.
    Returns: np.ndarray of shape (n_candidates, 13)
    """
    rows = []
    for c in candidates:
        meta      = c["metadata"]
        upvotes   = float(meta.get("upvotes",  0))
        downvotes = float(meta.get("downvotes", 0))
        uses      = float(meta.get("uses",  0))
        views     = float(meta.get("views", 0))

        rows.append([
            float(c["cosine_score"]),
            float(c["bm25_score"]),
            float(c["retrieval_rank"]),
            float(c.get("bm25_rel_rank",       0.0)),
            float(c["cosine_title"]),
            float(c.get("log_content_length",  0.0)),
            float(c.get("title_token_overlap", 0.0)),
            np.log1p(upvotes),
            upvotes / (upvotes + downvotes + 1),
            np.log1p(float(meta.get("author_reputation", 0))),
            np.log1p(uses) * float(c["cosine_score"]),
            np.log1p(uses) / (np.log1p(views) + 1) * float(c["cosine_score"]),
            float(c.get("category_match", 0.0)),
        ])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Fast path: load cache ────────────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"Cache found ({CACHE_PATH}), loading training data...")
        cache    = np.load(CACHE_PATH)
        X        = cache["X"]
        y        = cache["y"]
        all_qids = cache["qids"].tolist()
        print(f"       Loaded {X.shape[0]} samples, {X.shape[1]} features.")

    # ── Slow path: build training data from scratch ──────────────────────────
    else:
        print("No cache found. Building training data from scratch...")

        print(f"[1/5] Loading dataset from {DATASET_PATH}...")
        df = load_data(DATASET_PATH)
        sampled_pairs = stratified_sample_queries(
            df, n_total=N_QUERIES, min_per_cat=MIN_PER_CATEGORY, seed=RANDOM_SEED
        )
        sampled_queries    = [title  for title, _      in sampled_pairs]
        sampled_source_ids = [doc_id for _,     doc_id in sampled_pairs]
        print(f"       Sampled {len(sampled_queries)} pseudo-queries "
              f"(stratified, min_per_cat={MIN_PER_CATEGORY}, seed={RANDOM_SEED}).")

        print(f"[2/5] Loading models and indexes...")
        embedder   = SentenceTransformer(EMBED_MODEL_NAME)
        client     = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        bm25, id_to_idx = build_bm25_index(df)
        print(f"       BM25 index built ({len(df)} docs).")
        cat_embeddings = build_category_index(df, embedder)
        print(f"       Category index built ({len(cat_embeddings)} categories).")

        print(f"[3/5] Loading reranker as teacher: {RERANK_MODEL_NAME}...")
        reranker = load_reranker(RERANK_MODEL_NAME)

        print(f"[4/5] Generating training data ({len(sampled_queries)} queries × {TOP_K} candidates)...")
        all_features, all_labels, all_qids = [], [], []

        print("       Pre-encoding all queries...")
        all_q_vecs = embedder.encode(
            sampled_queries, normalize_embeddings=True,
            batch_size=64, show_progress_bar=True
        )

        n_batches = (len(sampled_queries) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
        for batch_start in tqdm(range(0, len(sampled_queries), QUERY_BATCH_SIZE),
                                 total=n_batches, desc="Building training data"):
            batch_end        = min(batch_start + QUERY_BATCH_SIZE, len(sampled_queries))
            batch_queries    = sampled_queries[batch_start:batch_end]
            batch_q_vecs     = all_q_vecs[batch_start:batch_end]
            batch_source_ids = sampled_source_ids[batch_start:batch_end]

            batch_results = collection.query(
                query_embeddings=batch_q_vecs.tolist(),
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )

            batch_candidates = []
            batch_q_indices  = []

            for i, (query, q_vec_np, source_id) in enumerate(
                    zip(batch_queries, batch_q_vecs, batch_source_ids)):
                docs      = batch_results["documents"][i]
                metas     = batch_results["metadatas"][i]
                distances = batch_results["distances"][i]

                if not docs:
                    batch_candidates.append(None)
                    batch_q_indices.append(None)
                    continue

                candidates = [
                    {
                        "content":      docs[j],
                        "metadata":     metas[j],
                        "cosine_score": 1.0 - distances[j],
                        "retrieval_source": "vector",
                    }
                    for j in range(len(docs))
                ]
                # Remove the source document to avoid trivial self-match label=1.0
                candidates = [
                    c for c in candidates
                    if str(c["metadata"].get("id", "")) != source_id
                ]

                seen_ids = {str(c["metadata"].get("id", "")) for c in candidates}
                q_tokens_bm25 = _tokenize_with_bigrams(query)
                bm25_scores = bm25.get_scores(q_tokens_bm25)
                top_bm25 = sorted(
                    range(len(bm25_scores)),
                    key=lambda idx: bm25_scores[idx],
                    reverse=True,
                )[:TOP_K // 2]

                bm25_rows = []
                bm25_ids = []
                for bm25_idx in top_bm25:
                    row = df.iloc[bm25_idx]
                    doc_id = str(row["id"])
                    if doc_id == source_id or doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    bm25_rows.append(row)
                    bm25_ids.append(doc_id)

                bm25_vecs_by_id = {}
                if bm25_ids:
                    fetched = collection.get(ids=bm25_ids, include=["embeddings"])
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
                        "content": str(row.get("title", "")) + ". " + str(row.get("content", "")),
                        "metadata": {
                            "id": str(row.get("id", "")),
                            "title": str(row.get("title", "")),
                            "category": str(row.get("category", "")),
                            "likes": int(row.get("likes", 0)),
                            "upvotes": int(row.get("upvotes", 0)),
                            "downvotes": int(row.get("downvotes", 0)),
                            "views": int(row.get("views", 0)),
                            "uses": int(row.get("uses", 0)),
                            "fork_count": int(row.get("fork_count", 0)),
                            "author_reputation": int(row.get("author_reputation", 0)),
                        },
                        "cosine_score": cosine_score,
                        "retrieval_source": "bm25",
                    })
                if not candidates:
                    batch_candidates.append(None)
                    batch_q_indices.append(None)
                    continue

                enrich_candidates(candidates, query, q_vec_np, bm25, id_to_idx, embedder,
                                  cat_embeddings)
                batch_candidates.append(candidates)
                batch_q_indices.append(batch_start + i)

            all_pairs    = []
            pair_offsets = []

            for i, (query, candidates) in enumerate(zip(batch_queries, batch_candidates)):
                if candidates is None:
                    pair_offsets.append((len(all_pairs), 0))
                    continue
                start = len(all_pairs)
                for c in candidates:
                    title    = c["metadata"].get("title", "")
                    doc_text = f"{title}. {c['content']}" if title else c["content"]
                    all_pairs.append((query, doc_text))
                pair_offsets.append((start, len(candidates)))

            if not all_pairs:
                continue

            batch_scores = reranker.predict(all_pairs, batch_size=8, show_progress_bar=False)

            for i, (candidates, q_idx) in enumerate(zip(batch_candidates, batch_q_indices)):
                if candidates is None:
                    continue
                p_start, p_len = pair_offsets[i]
                scores_i = batch_scores[p_start : p_start + p_len]

                for j, c in enumerate(candidates):
                    c["reranker_score"] = float(scores_i[j])
                candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

                labels = np.array([c["reranker_score"] for c in candidates], dtype=np.float32)
                span   = labels.max() - labels.min()
                if span > 1e-6:
                    labels = (labels - labels.min()) / span

                all_features.append(build_features(candidates))
                all_labels.extend(labels.tolist())
                all_qids.extend([q_idx] * len(candidates))

        X = np.vstack(all_features)
        y = np.array(all_labels, dtype=np.float32)
        print(f"       Training matrix: {X.shape[0]} samples, {X.shape[1]} features.")

        np.savez(CACHE_PATH, X=X, y=y, qids=np.array(all_qids))
        print(f"       Training data cached to {CACHE_PATH}")

    # ── 70/15/15 train/val/test split by query ───────────────────────────────
    all_qids_arr = np.array(all_qids)
    train_mask, val_mask, test_mask = three_way_split(all_qids_arr)

    n_train_q = len(np.unique(all_qids_arr[train_mask]))
    n_val_q   = len(np.unique(all_qids_arr[val_mask]))
    n_test_q  = len(np.unique(all_qids_arr[test_mask]))

    print(f"[5/5] Training XGBoost ranker (rank:pairwise)...")
    print(f"       Split → train: {n_train_q}q  val: {n_val_q}q  test: {n_test_q}q (held out)")

    # ── Load best params from Optuna, or fall back to defaults ──────────────
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            params = json.load(f)
        print(f"       Params: loaded from {BEST_PARAMS_PATH}")
    else:
        params = DEFAULT_PARAMS.copy()
        print(f"       Params: defaults (run optimization.py to tune)")

    # ── Train on train + val (test is permanently held out) ──────────────────
    tv_mask  = train_mask | val_mask
    X_tv     = X[tv_mask]
    y_tv     = y[tv_mask]
    qids_tv  = all_qids_arr[tv_mask]

    # XGBRanker requires all rows of the same query to be contiguous
    sort_idx = np.argsort(qids_tv, kind="stable")
    X_tv, y_tv, qids_tv = X_tv[sort_idx], y_tv[sort_idx], qids_tv[sort_idx]

    print(f"       Training on {X_tv.shape[0]} samples ({len(np.unique(qids_tv))} queries)")

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        random_state=RANDOM_SEED,
        verbosity=1,
        **params
    )
    model.fit(X_tv, y_tv, qid=qids_tv, verbose=True)
    model.save_model(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    importance = model.get_booster().get_score(importance_type="gain")
    print("\nFeature importances (gain):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:20s}: {importance.get(f'f{i}', 0.0):.4f}")


if __name__ == "__main__":
    main()
