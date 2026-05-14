"""
File: warmup.py
Role: One-shot model pre-download

Usage:
    python warmup.py

Description:
    The Streamlit demo (streamlit_app.py) and the offline training scripts
    both depend on two HuggingFace models:
        - BAAI/bge-base-en-v1.5         (~440 MB)  — bi-encoder embeddings
        - BAAI/bge-reranker-v2-m3       (~2.3 GB)  — cross-encoder reranker

    On a fresh machine the first call to SearchEngine() in the demo would
    silently spend 5-30 minutes downloading these weights with only a
    generic Streamlit spinner — easy to mistake for a frozen UI and quit.

    This script does the download up-front so reviewers can:
        1. Run `python warmup.py`        (visible HuggingFace progress bars)
        2. Run `streamlit run streamlit_app.py`  (loads in ~30 s instead)

    Models are stored in the user's standard HuggingFace cache
    (~/.cache/huggingface by default), so any later script in the project
    re-uses them without re-downloading.

    Safe to re-run: HuggingFace skips already-downloaded files.
"""

import sys
import time

EMBED_MODEL_NAME  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _section(title: str) -> None:
    """Print a visible separator so progress bars are easy to find in the log."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    overall_start = time.perf_counter()

    # ── 1. Bi-encoder ────────────────────────────────────────────────────────
    _section(f"[1/2] Downloading bi-encoder: {EMBED_MODEL_NAME}  (~440 MB)")
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.perf_counter()
        SentenceTransformer(EMBED_MODEL_NAME)
        print(f"    ✓ ready in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"    ✗ failed: {e}")
        return 1

    # ── 2. Cross-encoder reranker ────────────────────────────────────────────
    _section(f"[2/2] Downloading cross-encoder: {RERANK_MODEL_NAME}  (~2.3 GB)")
    print("    Note: this is the largest model and takes the longest on first run.")
    try:
        from sentence_transformers import CrossEncoder
        t0 = time.perf_counter()
        CrossEncoder(RERANK_MODEL_NAME)
        print(f"    ✓ ready in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"    ✗ failed: {e}")
        return 1

    # ── Done ─────────────────────────────────────────────────────────────────
    total = time.perf_counter() - overall_start
    _section("Warm-up complete")
    print(f"Total elapsed: {total:.1f}s")
    print()
    print("Next step:")
    print("    streamlit run streamlit_app.py")
    print()
    print("Or, to rebuild the index / re-train the ranker first:")
    print("    python src/build_index.py")
    print("    python src/XGboost.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
