"""
File: evaluate.py
Role: Reproduce the headline numbers reported in readme.md

Usage:
    python src/evaluate.py                  (Part 1 + Part 2: full evaluation)
    python src/evaluate.py --no-latency     (Part 1 only — fast, ~10 s)
    python src/evaluate.py --latency-only   (Part 2 only — needs full SearchEngine)

Description:
    Single source of truth for the metrics quoted in readme.md.

    Part 1 — Held-out test set quality (~10 s)
        Loads the cached training matrix and the trained XGBoost model,
        applies the deterministic 70/15/15 query-ID split (the same one
        used by XGboost.py / optimization.py), evaluates on the held-out
        test queries.
        Reports: P@5 and MRR with relevance threshold (label > 0.5).
        Reproduces readme Part 1: P@5 = 0.731, MRR = 0.922.

    Part 2 — End-to-end latency (~5 min — loads models)
        Instantiates the live SearchEngine, runs Baseline / Method A /
        Method B over a 20-query benchmark set with 1 warmup query,
        reports median and P95 latency.
        Reproduces readme Part 3: Method B ≈ 63 ms (35× faster than A).

    No Optuna runs, no XGBoost re-training, no cross-encoder re-scoring.
    The script is intentionally fast and uses pre-built artifacts:
        - training_cache_v9.npz   (built by XGboost.py)
        - xgb_model.json          (built by XGboost.py)
        - chroma_db/              (built by build_index.py)

Run from the project root.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python src/evaluate.py` from the project root.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================================
# Part 1 — Test-set P@5 and MRR
# ============================================================================
def evaluate_test_set() -> dict:
    """
    Load training cache + trained model, evaluate on the held-out test split.

    Returns a dict with the val and test metrics so the caller can print them.
    Re-uses the same compute_metrics implementation as optimization.py to
    guarantee the numbers match.
    """
    import xgboost as xgb
    from XGboost import CACHE_PATH, MODEL_SAVE_PATH, three_way_split
    from optimization import compute_metrics, sort_by_qid

    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: cache not found at {CACHE_PATH}", file=sys.stderr)
        print("       Run `python src/XGboost.py` first.", file=sys.stderr)
        return {}

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: model not found at {MODEL_SAVE_PATH}", file=sys.stderr)
        print("       Run `python src/XGboost.py` first.", file=sys.stderr)
        return {}

    print(f"Loading training cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    X     = cache["X"]
    y     = cache["y"]
    qids  = cache["qids"]
    print(f"    {X.shape[0]} samples · {X.shape[1]} features · "
          f"{len(np.unique(qids))} queries")

    print(f"Loading trained model: {MODEL_SAVE_PATH}")
    model = xgb.XGBRanker()
    model.load_model(MODEL_SAVE_PATH)

    # Identical split mechanism to XGboost.py / optimization.py — RANDOM_SEED+2.
    train_mask, val_mask, test_mask = three_way_split(qids)
    X_val,  y_val,  q_val  = sort_by_qid(X[val_mask],  y[val_mask],  qids[val_mask])
    X_test, y_test, q_test = sort_by_qid(X[test_mask], y[test_mask], qids[test_mask])

    print(f"    Val  : {len(np.unique(q_val))} queries  · {X_val.shape[0]} samples")
    print(f"    Test : {len(np.unique(q_test))} queries · {X_test.shape[0]} samples")

    p5_val,  mrr_val  = compute_metrics(model, X_val,  y_val,  q_val)
    p5_test, mrr_test = compute_metrics(model, X_test, y_test, q_test)

    return {
        "val":  {"p5": p5_val,  "mrr": mrr_val,  "n": len(np.unique(q_val))},
        "test": {"p5": p5_test, "mrr": mrr_test, "n": len(np.unique(q_test))},
    }


def print_quality_table(metrics: dict) -> None:
    """Pretty-print the val / test results in the same shape as readme.md."""
    if not metrics:
        return
    print()
    print("Part 1 — Held-out test set quality")
    print("-" * 50)
    print(f"  {'Split':<8}  {'#Queries':>9}  {'P@5':>8}  {'MRR':>8}")
    for split in ("val", "test"):
        m = metrics[split]
        print(f"  {split:<8}  {m['n']:>9}  {m['p5']:>8.3f}  {m['mrr']:>8.3f}")
    print()
    print("  Reference (readme.md): test P@5 = 0.731 · MRR = 0.922")


# ============================================================================
# Part 2 — End-to-end latency benchmark
# ============================================================================
# Same 20-query benchmark set referenced in readme.md so the numbers are
# directly comparable. Spans high-coverage dataset categories.
BENCHMARK_QUERIES = [
    "develop a serialized fantasy story arc with character motivations and episode outline",
    "create an SEO audit checklist for an e-commerce product page",
    "write persuasive ad copy for a new skincare product launch",
    "write a cold call script for qualifying B2B sales leads",
    "create follow-up email templates after a trade show conversation",
    "review a vendor contract for legal risks and liability clauses",
    "draft a GDPR consent form for collecting customer data",
    "write an empathetic customer support reply about a duplicate billing charge",
    "create an FAQ article explaining a return policy clearly",
    "generate an accessible color palette and typography system for a website",
    "explain form design best practices for better user experience",
    "explain SQL joins with simple examples for a beginner analyst",
    "clean messy survey data and summarize the main findings",
    "translate a formal business email from English to Spanish with polite tone",
    "prepare source content for app internationalization and localization",
    "build a due diligence checklist for acquiring a small manufacturing company",
    "forecast monthly revenue for a subscription business model",
    "refactor a React component and write Jest unit tests",
    "debug docker compose networking between backend and database services",
    "create a vulnerability assessment checklist for a small company network",
]


def evaluate_latency() -> None:
    """
    Instantiate the live SearchEngine and run the latency benchmark.

    This step is slow on first run (downloads model weights) so it is gated
    behind a CLI flag. After warmup the benchmark itself takes ~1 minute.
    """
    print("Loading SearchEngine (this can take 30 s – 30 min on first run)...")
    from query import SearchEngine

    t0 = time.perf_counter()
    engine = SearchEngine()
    print(f"    SearchEngine ready in {time.perf_counter() - t0:.1f}s")

    print()
    print("Part 2 — End-to-end latency benchmark")
    print("-" * 50)
    print(f"  {len(BENCHMARK_QUERIES)} queries · top_k = 5 · 1 warmup")
    engine.latency_benchmark(BENCHMARK_QUERIES, top_k=5, warmup=1)
    print("  Reference (readme.md): Method B ≈ 63 ms median · 35× faster than A")


# ============================================================================
# Entry point
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--no-latency",   action="store_true",
                       help="Skip Part 2 (latency); run only Part 1.")
    group.add_argument("--latency-only", action="store_true",
                       help="Skip Part 1; run only Part 2.")
    args = parser.parse_args()

    print("=" * 50)
    print("LEAF Prompt Search — Reproducibility Evaluation")
    print("=" * 50)

    if not args.latency_only:
        metrics = evaluate_test_set()
        print_quality_table(metrics)

    if not args.no_latency:
        evaluate_latency()

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
