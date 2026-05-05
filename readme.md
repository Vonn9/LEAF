# Semantic Prompt Search Engine

A semantic search system for prompt discovery that retrieves and ranks the most relevant prompts for arbitrary natural-language queries by combining dense retrieval, lexical search, and learning-to-rank. Combines bi-encoder vector search, BM25 lexical retrieval, HyDE query expansion, and XGBoost reranking to surface the most relevant prompts for any natural-language query.

---

## Architecture Overview

Two retrieval methods are implemented and compared side-by-side:

### Method A — Baseline
```
Query → bi-encoder embedding → ChromaDB vector search (top-50) → cross-encoder rerank → Top-K
```

### Method B — Proposed
```
Query ──► confidence probe (top-3 cosine avg)
               │
               ├─ avg ≥ 0.68 ──────────────────────────────┐
               │                                            │
               └─ avg < 0.68 ──► HyDE (qwen2.5:3b)         │
                                  averaged query embedding   │
                                            │               │
                                            ▼               ▼
                               ChromaDB vector search (top-50)
                               BM25 lexical search    (top-25)  ◄── union, deduplicated
                                            │
                                            ▼
                               Enrich candidates with 13 features
                               (cosine · BM25 · retrieval rank · engagement · category match …)
                                            │
                                            ▼
                               XGBoost ranker → Top-K
```

### Design Decisions

| Choice | Rationale |
|--------|-----------|
| **BGE-base-en-v1.5** bi-encoder | Strong quality/speed balance at 768 dimensions; normalized vectors enable cosine similarity directly in ChromaDB |
| **ChromaDB + HNSW** | Persistent local vector store with approximate nearest-neighbor search — no external service required |
| **BM25 with stemmed bigrams** | Catches exact/stem-match candidates that vector search misses when query and document vocabularies diverge (e.g. "binary search" → bigram `binari_search` prevents false matches on "search" alone) |
| **HyDE** (Hypothetical Document Embeddings) | Local LLM generates a sample prompt that would satisfy the query; averaging its embedding with the query vector bridges the semantic gap for abstract or domain-specific queries |
| **Hybrid retrieval** | Union of vector top-50 and BM25 top-25 gives XGBoost a richer, more diverse candidate pool than either method alone |
| **XGBoost Learning-to-Rank** | Combines 13 heterogeneous signals (semantic similarity, lexical match, community engagement, domain alignment) that no single model captures; `rank:pairwise` objective is robust with continuous pseudo-labels |
| **BGE-reranker-v2-m3 as teacher** | Cross-encoder generates offline pseudo-labels per query; paired with the same BGE embedding family for vocabulary consistency |

---

## Dataset

**20,000 prompts**, each with the following fields:

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `id` | `string` | Unique prompt identifier. Zero-padded to 5 digits. |
| 2 | `author_reputation` | `integer` | Author's reputation score on the platform. |
| 3 | `version` | `integer` | How many times the prompt has been revised. |
| 4 | `fork_count` | `integer` | Number of times other users forked (copied & modified) this prompt. |
| 5 | `likes` | `integer` | Total likes received. |
| 6 | `upvotes` | `integer` | Community upvotes. |
| 7 | `downvotes` | `integer` | Community downvotes. |
| 8 | `views` | `integer` | Total view count. |
| 9 | `uses` | `integer` | Number of times the prompt was actually sent to an LLM through the platform. |
| 10 | `created_at` | `string` | ISO 8601 UTC timestamp of when the prompt was first published. Spans roughly the last 2 years. |
| 11 | `title` | `string` | Short title for the prompt. |
| 12 | `content` | `string` | The actual prompt text a user would send to an LLM. This is the **primary semantic field**. |
| 13 | `category` | `string` | Lowercase, hyphenated topic category (e.g. `"coding"`, `"creative-writing"`, `"data-analysis"`, `"marketing"`). Assigned per prompt. |
| 14 | `subcategory` | `string` | More specific label within the category (e.g. `"performance-analysis"`, `"email-marketing"`). Lowercase, hyphenated. |
| 15 | `tags` | `array[string]` | 2–8 lowercase descriptive tags (e.g. `["nodejs", "performance", "debugging"]`). |
| 16 | `has_placeholders` | `boolean` | `true` if the prompt contains `{{variable}}` template placeholders, `false` otherwise. |
| 17 | `placeholders` | `array[string]` | List of placeholder variable names found in `content` (without braces). Empty array `[]` if none. Examples: `["language", "code_snippet"]`. |
| 18 | `difficulty` | `string` | One of: **`beginner`**, **`intermediate`**, **`advanced`**, **`expert`**. |
| 19 | `language` | `string` | ISO 639-1 language code of the prompt content. Predominantly `"en"`, with occasional `"it"`, `"es"`, `"fr"`, `"de"`, `"pt"`, `"zh"`, `"ja"`. |
| 20 | `target_model` | `string` | The LLM the author intended the prompt for. |

---

## Models

| Model | Role |
|-------|------|
| `BAAI/bge-base-en-v1.5` | Bi-encoder for document and query embeddings (768-dim, normalized) |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder — Method A reranker + offline teacher for XGBoost labels |
| `qwen2.5:3b` via Ollama | Local LLM for HyDE hypothetical document generation (confidence-gated) |
| `XGBRanker` (rank:pairwise) | Learned ranker combining 13 features for Method B |

---

## Project Structure

```
LEAF-promptkaban-dataset/
│
├── dataset.json                  # Raw prompt dataset (20K prompts)
├── requirements.txt              # Python dependencies
├── chroma_db/                    # ChromaDB vector index (generated)
├── title_vecs.npz                # Precomputed title embeddings (generated)
├── xgb_model.json                # Trained XGBoost ranker (generated)
├── best_params.json              # Optuna best hyperparameters (generated)
├── training_cache_v9.npz         # Cached training matrix — X, y, query IDs
├── FIELDS.md                     # Dataset field reference
├── a.ipynb                       # Quick ChromaDB sanity check
│
└── src/
    ├── dataload.py               # Load dataset.json → pandas DataFrame
    ├── processing.py             # Preprocessing · BM25 index · category embeddings
    ├── embedding.py              # Batch bi-encoder embedding generation
    ├── Build Index.py            # [Step 1] Build ChromaDB vector index + title embeddings
    ├── rerank.py                 # Cross-encoder load + rerank helper
    ├── XGboost.py                # [Step 2] Train XGBoost ranker offline
    ├── optimization.py           # [Optional] Bayesian hyperparameter tuning via Optuna
    └── query.py                  # [Step 3] Run search — Method A / B / A/B test
```

---

## Requirements

Python 3.12

```bash
pip install -r requirements.txt
```

For HyDE support, Ollama must be running locally with qwen2.5:3b(However, you are freely to choose any local language model that you want):

```bash
ollama pull qwen2.5:3b
ollama serve
```

HyDE is confidence-gated — it only triggers when the query's average top-3 retrieval cosine score falls below 0.68. For precise queries this path is skipped entirely. If Ollama is unavailable, Method B silently falls back to the original query embedding.

---

## How to Run

All commands are run from the **project root**.

### Step 1 — Build the vector index

Loads `dataset.json`, preprocesses text (title + content + category + subcategory + tags), generates embeddings, and inserts them into ChromaDB. Run once, or whenever the dataset changes.

```bash
python src/"Build Index.py"
```

Runtime: ~3 minutes for 20K documents. Output: `chroma_db/`, `title_vecs.npz`

---

### Step 2 — Train the XGBoost ranker

Stratified-samples 500 document titles as pseudo-queries (at least 3 per category), retrieves candidates for each via ChromaDB, scores them with the cross-encoder as a teacher model, and trains an `XGBRanker` on 13 engineered features.

Training data is cached after the first run — subsequent runs load from `training_cache_v9.npz` and complete in seconds.

```bash
python src/XGboost.py
```

Runtime: ~10-20 mins on first run (cross-encoder scoring); seconds on cache hit. Output: `xgb_model.json`

> Bump `CACHE_VERSION` in `XGboost.py` to invalidate the cache when features or queries change.

---

### Step 2.5 — Tune hyperparameters (optional)

Runs 50 Optuna trials on the validation split to find the best XGBoost hyperparameters, then evaluates once on the held-out test set. Saves results to `best_params.json`, which `XGboost.py` loads automatically on the next training run.

```bash
python src/optimization.py
```

Runtime: ~1 min (50 trials × cross-encoder scoring on val set). Skip if the default parameters are sufficient.

---

### Step 3 — Run search

**A/B comparison** over the built-in test query set:

```bash
python src/query.py
```

**Programmatic usage:**

```python
import sys
sys.path.insert(0, "src")
from query import SearchEngine

engine = SearchEngine()

# Method A: vector search + cross-encoder rerank
results_a = engine.search_A("write a REST API with FastAPI", top_k=5)

# Method B: HyDE + hybrid retrieval + XGBoost rerank
results_b = engine.search_B("write a REST API with FastAPI", top_k=5)

# Side-by-side comparison
engine.ab_test("write a REST API with FastAPI", top_k=5)

# Each result dict contains:
# result["metadata"]["title"]     → prompt title
# result["metadata"]["category"]  → category
# result["cosine_score"]          → bi-encoder similarity
# result["xgb_score"]             → XGBoost rank score (Method B only)
# result["reranker_score"]        → cross-encoder score (Method A only)
```

---

## XGBoost Features (13 total)

| Feature | Type | Description |
|---------|------|-------------|
| `cosine_score` | query-dependent | Bi-encoder cosine similarity (query ↔ document) |
| `bm25_score` | query-dependent | Absolute bigram BM25 score |
| `retrieval_rank` | query-dependent | ChromaDB rank normalised to [0,1] (1.0 = rank 1) |
| `bm25_rel_rank` | query-dependent | BM25 rank within this query's candidate pool (normalised) |
| `cosine_title` | query-dependent | Cosine similarity between query and title embedding only |
| `log_content_length` | static | log(len(content) + 1) — distinguishes detailed vs generic prompts |
| `title_token_overlap` | query-dependent | Jaccard similarity between stemmed query and title tokens |
| `log_upvotes` | static | Community approval signal |
| `vote_rate` | static | upvotes / (upvotes + downvotes + 1) |
| `log_author_rep` | static | Author reputation score (log-scaled) |
| `log_uses × cosine` | query-dependent | Usage count gated by semantic relevance — social signal only when contextually relevant |
| `engagement_rate × cosine` | query-dependent | log(uses) / (log(views) + 1) × cosine — penalises high-view, low-use prompts |
| `category_match` | query-dependent | Cosine similarity between query vector and category name embedding |

---

## Evaluation

Two complementary evaluations are reported. They use different oracles and query sources, so their absolute numbers are not directly comparable — each captures a different dimension of system quality.

### Part 1 — Held-out test set (Method B, quantitative)

75 queries held out from training (pseudo-queries = document titles, never seen during Optuna tuning).  
Relevance oracle: normalized cross-encoder label > 0.5 (derived from the same teacher model used to generate training labels).

| Split | P@5 | MRR |
|-------|:---:|:---:|
| Val (75 queries) | 0.725 | 0.952 |
| **Test (75 queries)** | **0.731** | **0.922** |

Val ≈ Test confirms that Bayesian tuning did not overfit to the validation set.

### Part 2 — Real user queries, A/B comparison (qualitative)

19 natural-language queries across 9 categories, manually inspected.  
Relevance oracle: cross-encoder score > 0.5 (absolute logit, applied to Method A's reranker output and estimated for Method B via document overlap).

| Method | Mean P@5 | Mean MRR | Notes |
|--------|:--------:|:--------:|-------|
| Method A — baseline | 0.442 | 0.684 | Cross-encoder rerank |
| Method B — proposed | 0.463 | 0.649 | XGBoost + HyDE + hybrid |

Method B surfaces more relevant documents across the top-5 (higher P@5). Method A is more consistent at placing the single best result at rank 1 (higher MRR). Both methods fail on the same 6 query types where the dataset has no relevant coverage — a data gap, not a model failure.

> **Oracle note:** Part 1 uses a normalized per-query label (relative relevance within the candidate pool). Part 2 uses absolute cross-encoder logits. Part 1 metrics are higher partly because pseudo-queries (document titles) match the dataset vocabulary more directly than real user queries.

### Part 3 — Latency benchmark (15 queries, top-5, 1 warmup)

| Method | Median (ms) | P95 (ms) |
|--------|:-----------:|:--------:|
| Baseline — vector only | 14 | 21 |
| Method A — cross-encoder | 2,227 | 3,046 |
| **Method B — XGBoost+Meta** | **63** | **74** |

Method B is **35× faster** than Method A. Two optimizations account for the difference:

1. **Confidence-gated HyDE** — a lightweight probe retrieves top-3 results; if their average cosine similarity ≥ 0.68, the query is deemed precise and the Ollama generation is skipped entirely. All 15 benchmark queries cleared this threshold, reducing HyDE cost to zero for this test set.
2. **Precomputed title embeddings** — `cosine_title` is computed via dictionary lookup + dot product instead of a per-query batch encode of up to 75 candidate titles (~300–500 ms saved).

XGBoost inference remains negligible (<1 ms). Method B's end-to-end path for precise queries is: probe (~10 ms) → vector search (~20 ms) → BM25 (~20 ms) → title lookup + XGBoost (~5 ms).

---

## Known Limitations

- **Dataset coverage**: Queries outside the dataset's domain (specific programming algorithms, historical topics, niche wellness) return low-quality results regardless of ranking method.
- **Greeting-titled prompts**: Some dataset prompts have informal titles ("Hello dear…") despite relevant content; these may appear in results with misleading titles.
- **Offline training**: The XGBoost model is trained on pseudo-labels derived from document titles. Queries that differ significantly in style from document titles may be ranked suboptimally.
- **HyDE triggering**: Confidence-gated HyDE skips Ollama generation when retrieval confidence is high (avg top-3 cosine ≥ 0.68). For highly abstract queries below this threshold, Method B will still incur a ~1–2 s Ollama round-trip.

---

## Author
