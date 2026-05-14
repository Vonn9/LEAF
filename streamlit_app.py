import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


# PROJECT_ROOT is two levels up from this file (src/streamlit_app.py → project root).
# SRC_DIR is the directory this file lives in — added to sys.path so that sibling
# modules (query.py, XGboost.py, …) are importable without a package prefix.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "dataset.json"
DB_PATH = PROJECT_ROOT / "chroma_db"
XGB_MODEL_PATH = SRC_DIR / "xgb_model.json"   # shipped artifact — lives next to the code
CACHE_ROOT = PROJECT_ROOT / ".local_cache"

# Headline metrics — pulled from readme.md held-out test set results.
# Shown at the top of the search tab as the "selling point" of the pipeline,
# so reviewers see the quantitative claims before typing any query.
HEADLINE_METRICS = {
    "P@5 (test)": "0.682",
    "MRR (test)": "0.900",
    "Method B p50 latency": "104 ms",
    "Method B vs A speedup": "58×",
}


def configure_local_runtime() -> None:
    """
    Force model/download caches to stay inside the project directory.

    This keeps the app self-contained:
    - model files downloaded by Hugging Face stay under `.local_cache`
    - torch cache stays local as well
    - local imports like `query.py` remain resolvable from Streamlit
    - `os.chdir(PROJECT_ROOT)` makes relative paths in query.py / XGboost.py
      (e.g. `./dataset.json`, `./chroma_db`) resolve regardless of where
      `streamlit run` was invoked from.
    """
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    cache_paths = {
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
        "SENTENCE_TRANSFORMERS_HOME": CACHE_ROOT / "sentence_transformers",
        "TORCH_HOME": CACHE_ROOT / "torch",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in cache_paths.items():
        os.environ[key] = str(path)

    # Anchor the working directory so query.py's relative paths resolve no
    # matter where `streamlit run` was executed from.
    os.chdir(PROJECT_ROOT)


# Execute once at import time so every later model/script call inherits the
# same local cache layout.
configure_local_runtime()


def artifact_status() -> dict:
    """Return the existence of the key generated artifacts used by the UI."""
    return {
        "dataset": DATASET_PATH.exists(),
        "chroma_db": DB_PATH.exists(),
        "xgb_model": XGB_MODEL_PATH.exists(),
    }


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """
    Load the raw dataset for dashboard/statistics usage.

    `cache_data` is appropriate here because the return value is plain data,
    not a heavyweight model object. Streamlit can safely memoize it between
    reruns and refresh it only when the file content changes.
    """
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def load_readme() -> str:
    """Load README content for direct display in the third tab."""
    readme_path = PROJECT_ROOT / "src/readme.md"
    return readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""


@st.cache_resource(show_spinner=False)
def _build_engine():
    """Build SearchEngine once; cached across all reruns."""
    from query import SearchEngine
    return SearchEngine()


def load_engine():
    """
    Return the cached SearchEngine, showing a status panel only on first load.

    `st.status` lives outside `@st.cache_resource` so that `status.update()`
    is always called on every Streamlit rerun — otherwise the spinner stays
    stuck in "loading" state after the engine is already cached.
    """
    already_built = "engine_ready" in st.session_state
    with st.status(
        "Search engine ready" if already_built else "Loading search engine…",
        expanded=not already_built,
    ) as status:
        if not already_built:
            st.write("➤ Importing query module")
            st.write("➤ Loading bi-encoder (BAAI/bge-base-en-v1.5)")
            st.write("➤ Connecting to ChromaDB")
            st.write("➤ Loading cross-encoder (BAAI/bge-reranker-v2-m3)")
            st.write("➤ Loading XGBoost ranker + BM25 index + title vectors")

        engine = _build_engine()
        st.session_state["engine_ready"] = True
        status.update(label="Search engine ready", state="complete", expanded=False)
    return engine


def run_project_script(script_name: str):
    """
    Run an existing project script and stream its stdout into the page.

    This is used for the two long-running setup actions exposed in the sidebar:
    - `src/build_index.py`
    - `src/XGboost.py`

    The implementation intentionally reuses the current Python interpreter so
    the Streamlit app and child scripts run inside the same environment.
    """
    env = os.environ.copy()
    command = [sys.executable, script_name]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Keep the latest chunk of logs visible while the subprocess is running.
    log_box = st.empty()
    collected = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            collected.append(line.rstrip())
            log_box.code("\n".join(collected[-120:]), language="text")

    return_code = process.wait()
    return return_code, "\n".join(collected)


def time_search(fn, *args, **kwargs):
    """
    Run a search function and return (results, elapsed_ms).

    Wrapping the call here keeps each search-mode branch in `main()` short
    and ensures every mode reports latency in the same way.
    """
    t0 = time.perf_counter()
    results = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


def score_table_three_way(
    results_baseline: list, results_a: list, results_b: list
) -> pd.DataFrame:
    """
    Convert Baseline / Method A / Method B retrieval outputs into a
    side-by-side comparison table.

    Each row represents one rank position. Three method columns are kept
    separate so differences in title and ranking signals are immediately
    visible.
    """
    rows = []
    max_len = max(len(results_baseline), len(results_a), len(results_b))
    for idx in range(max_len):
        row = {"rank": idx + 1}
        if idx < len(results_baseline):
            item = results_baseline[idx]
            row.update(
                {
                    "Baseline_title": item["metadata"].get("title", ""),
                    "Baseline_cosine": round(float(item.get("cosine_score", 0.0)), 4),
                }
            )
        if idx < len(results_a):
            item = results_a[idx]
            row.update(
                {
                    "A_title": item["metadata"].get("title", ""),
                    "A_reranker": round(float(item.get("reranker_score", 0.0)), 4),
                }
            )
        if idx < len(results_b):
            item = results_b[idx]
            row.update(
                {
                    "B_title": item["metadata"].get("title", ""),
                    "B_xgb": round(float(item.get("xgb_score", 0.0)), 4),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_result_card(result: dict, rank: int, method: str) -> None:
    """
    Render one retrieval result as a readable card.

    The card exposes:
    - semantic / rerank scores
    - engagement metadata
    - original prompt text
    """
    meta = result.get("metadata", {})
    title = meta.get("title", "Untitled")
    category = meta.get("category", "-")
    st.markdown(f"### {rank}. {title}")
    st.caption(f"{method} | category={category} | id={meta.get('id', '-')}")

    # The first row focuses on ranking signals so users can inspect why a
    # result surfaced where it did.
    metric_cols = st.columns(4)
    metric_cols[0].metric("Cosine", f"{float(result.get('cosine_score', 0.0)):.4f}")
    metric_cols[1].metric("Reranker", f"{float(result.get('reranker_score', 0.0)):.4f}")
    metric_cols[2].metric("XGBoost", f"{float(result.get('xgb_score', 0.0)):.4f}")
    vote_rate = (
        float(meta.get("upvotes", 0))
        / max(float(meta.get("upvotes", 0)) + float(meta.get("downvotes", 0)) + 1.0, 1.0)
    )
    metric_cols[3].metric("Vote Rate", f"{vote_rate:.3f}")

    # The second row shows popularity / usage signals that already exist in
    # the source dataset and help interpret prompt quality.
    info_cols = st.columns(4)
    info_cols[0].metric("Likes", int(meta.get("likes", 0)))
    info_cols[1].metric("Uses", int(meta.get("uses", 0)))
    info_cols[2].metric("Views", int(meta.get("views", 0)))
    info_cols[3].metric("Author Rep", int(meta.get("author_reputation", 0)))

    # Prompt text is collapsed by default to keep result pages scannable.
    with st.expander("Prompt Content", expanded=False):
        st.write(result.get("content", ""))


def render_headline_metrics() -> None:
    """
    Show held-out test set metrics at the top of the search tab.

    These numbers come from readme.md and represent the headline "selling
    points" of the retrieval pipeline — placed up front so reviewers don't
    have to dig through the README.
    """
    cols = st.columns(len(HEADLINE_METRICS))
    for col, (label, value) in zip(cols, HEADLINE_METRICS.items()):
        col.metric(label, value)


def render_dataset_overview(df: pd.DataFrame) -> None:
    """
    Build the dataset dashboard tab.

    This tab is deliberately lightweight: it gives high-level distribution
    information and a quick sample table without touching the retrieval stack.
    """
    st.subheader("Dataset Overview")

    # Top summary metrics are intended for a first-pass health check of the
    # dataset currently backing the index.
    metric_cols = st.columns(4)
    metric_cols[0].metric("Prompts", f"{len(df):,}")
    metric_cols[1].metric("Categories", int(df["category"].nunique()))
    metric_cols[2].metric("Languages", int(df["language"].nunique()))
    metric_cols[3].metric("Avg Uses", f"{df['uses'].mean():.1f}")

    # Category and difficulty distributions help confirm that the imported
    # dataset looks structurally sane.
    chart_cols = st.columns(2)
    top_categories = df["category"].value_counts().head(15)
    chart_cols[0].caption("Top 15 Categories")
    chart_cols[0].bar_chart(top_categories)

    difficulty_counts = df["difficulty"].value_counts()
    chart_cols[1].caption("Difficulty Distribution")
    chart_cols[1].bar_chart(difficulty_counts)

    lang_cols = st.columns(2)
    language_counts = df["language"].value_counts().head(10)
    lang_cols[0].caption("Top Languages")
    lang_cols[0].bar_chart(language_counts)

    # Limit the scatter sample to keep the page responsive on a 20K-row dataset.
    df_scatter = df[["uses", "views", "upvotes"]].copy()
    df_scatter = df_scatter[(df_scatter["views"] > 0) & (df_scatter["uses"] > 0)].head(3000)
    lang_cols[1].caption("Uses vs Views")
    lang_cols[1].scatter_chart(df_scatter, x="views", y="uses")

    st.subheader("Sample Rows")
    visible_cols = [
        "id",
        "title",
        "category",
        "subcategory",
        "difficulty",
        "language",
        "likes",
        "uses",
        "views",
    ]
    st.dataframe(df[visible_cols].head(50), use_container_width=True)


def main():
    """
    Assemble the full Streamlit application.

    Page structure:
    - Sidebar: project status + one-click setup actions
    - Tab 1: retrieval demo (Baseline / Method A / Method B / three-way)
    - Tab 2: dataset dashboard
    - Tab 3: rendered README
    """
    st.set_page_config(page_title="Prompt Search Visualizer", layout="wide")
    st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.70rem !important;
}

h1 {
    font-size: 1.8rem !important;
}

h2 {
    font-size: 1.25rem !important;
}

h3 {
    font-size: 1.05rem !important;
}

[data-testid="stCaptionContainer"] {
    font-size: 0.70rem !important;
}

button[kind="secondary"] {
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)
    st.title("Prompt Search Visualizer")
    st.caption(
        "Streamlit UI for the existing ChromaDB + Cross-Encoder + XGBoost retrieval pipeline"
    )

    status = artifact_status()
    with st.sidebar:
        # The sidebar doubles as a lightweight operations panel for initializing
        # the local artifacts required by the app.
        st.header("Project Status")
        st.write(f"Dataset: {'ready' if status['dataset'] else 'missing'}")
        st.write(f"ChromaDB: {'built' if status['chroma_db'] else 'not built'}")
        st.write(f"XGBoost Model: {'trained' if status['xgb_model'] else 'not trained'}")
        st.write(f"Python: `{sys.executable}`")
        st.write(f"Cache Root: `{CACHE_ROOT}`")

        if st.button("Build ChromaDB Index", use_container_width=True):
            # Rebuild the vector store by invoking the existing script rather
            # than duplicating indexing logic inside the UI.
            with st.spinner("Running build_index.py ..."):
                code, _ = run_project_script("src/build_index.py")
            if code == 0:
                st.success("ChromaDB build completed.")
                st.cache_resource.clear()
            else:
                st.error("ChromaDB build failed. Check the log output above.")

        if st.button("Train XGBoost Ranker", use_container_width=True):
            with st.spinner("Running XGboost.py ..."):
                code, _ = run_project_script("src/XGboost.py")
            if code == 0:
                st.success("XGBoost training completed.")
                st.cache_resource.clear()
            else:
                st.error("XGBoost training failed. Check the log output above.")

        if st.button("Refresh Status", use_container_width=True):
            # Clear cached data/resources so the UI re-reads files that may
            # have been generated by the setup scripts.
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if not status["dataset"]:
        st.error("dataset.json was not found. The app cannot continue.")
        return

    df = load_dataset()
    tab_search, tab_data, tab_readme = st.tabs(
        ["Search Demo", "Dataset Dashboard", "README"]
    )

    with tab_search:
        # Headline numbers anchor the tab so reviewers see the project's
        # quantitative claims before they even type a query.
        render_headline_metrics()
        st.divider()

        st.subheader("Search Demo")
        method = st.selectbox(
            "Search mode",
            options=[
                "Baseline (vector only)",
                "Method A (Cross-Encoder)",
                "Method B (XGBoost + HyDE)",
                "Three-way Compare (Baseline / A / B)",
            ],
            index=3,
        )
        top_k = st.slider("Top K", min_value=3, max_value=10, value=5)
        default_query = "write a REST API with FastAPI"
        query_text = st.text_input("Query", value=default_query)

        if not status["chroma_db"]:
            st.warning("ChromaDB is not built yet. Use the sidebar action first.")

        # Method B and the three-way comparison both require the trained
        # XGBoost model. Baseline and Method A only need the vector index.
        needs_xgb = method in {
            "Method B (XGBoost + HyDE)",
            "Three-way Compare (Baseline / A / B)",
        }
        if needs_xgb and not status["xgb_model"]:
            st.warning(
                "This mode requires xgb_model.json. Train the ranker from the sidebar first."
            )

        can_search = status["chroma_db"] and (not needs_xgb or status["xgb_model"])

        if st.button(
            "Run Search", type="primary", use_container_width=True, disabled=not can_search
        ):
            engine = load_engine()

            if method == "Baseline (vector only)":
                results, elapsed = time_search(
                    engine.search_baseline, query_text, top_k=top_k
                )
                st.metric("Latency", f"{elapsed:.0f} ms")
                for idx, result in enumerate(results, start=1):
                    render_result_card(result, idx, "Baseline")

            elif method == "Method A (Cross-Encoder)":
                results, elapsed = time_search(
                    engine.search_A, query_text, top_k=top_k
                )
                st.metric("Latency", f"{elapsed:.0f} ms")
                for idx, result in enumerate(results, start=1):
                    render_result_card(result, idx, "Method A")

            elif method == "Method B (XGBoost + HyDE)":
                results, elapsed = time_search(
                    engine.search_B, query_text, top_k=top_k
                )
                st.metric("Latency", f"{elapsed:.0f} ms")
                for idx, result in enumerate(results, start=1):
                    render_result_card(result, idx, "Method B")

            else:
                # Three-way comparison: time each method separately so the
                # 35× Method B speedup is visible at a glance.
                results_baseline, t_baseline = time_search(
                    engine.search_baseline, query_text, top_k=top_k
                )
                results_a, t_a = time_search(engine.search_A, query_text, top_k=top_k)
                results_b, t_b = time_search(engine.search_B, query_text, top_k=top_k)

                lat_cols = st.columns(4)
                lat_cols[0].metric("Baseline", f"{t_baseline:.0f} ms")
                lat_cols[1].metric("Method A", f"{t_a:.0f} ms")
                lat_cols[2].metric("Method B", f"{t_b:.0f} ms")
                speedup = (t_a / t_b) if t_b > 0 else 0.0
                lat_cols[3].metric("B vs A speedup", f"{speedup:.1f}×")

                st.markdown("### Side-by-side ranking")
                st.dataframe(
                    score_table_three_way(results_baseline, results_a, results_b),
                    use_container_width=True,
                )

                col_baseline, col_a, col_b = st.columns(3)
                with col_baseline:
                    st.markdown("## Baseline")
                    for idx, result in enumerate(results_baseline, start=1):
                        render_result_card(result, idx, "Baseline")
                with col_a:
                    st.markdown("## Method A")
                    for idx, result in enumerate(results_a, start=1):
                        render_result_card(result, idx, "Method A")
                with col_b:
                    st.markdown("## Method B")
                    for idx, result in enumerate(results_b, start=1):
                        render_result_card(result, idx, "Method B")

    with tab_data:
        render_dataset_overview(df)

    with tab_readme:
        st.markdown(load_readme())


if __name__ == "__main__":
    main()
