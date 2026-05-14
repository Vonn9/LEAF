"""
File: eda.py
Role: Exploratory Data Analysis — output PNG figures

Usage:
    python src/eda.py                 (run from project root)
    python eda.py                     (run from src/)

Description:
    Generates the EDA figures referenced in the technical report (Section 2)
    and the presentation (Slide 2). Output goes to ./figures/ at the project
    root, one PNG per panel so each can be embedded individually.

    Figures produced (1 PNG each, 300 dpi):
        1. figures/category_distribution.png   — top-15 categories by count
        2. figures/difficulty_distribution.png — beginner/intermediate/...
        3. figures/language_distribution.png   — language code share (top 8)
        4. figures/engagement_distribution.png — likes / upvotes / uses (log)
        5. figures/content_length.png          — content character-length hist
        6. figures/uses_vs_views.png           — engagement scatter (log-log)

Output: ./figures/*.png
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running both as `python src/eda.py` and from inside src/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
sys.path.insert(0, str(SCRIPT_DIR))

from dataload import load_data  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "dataset.json"
OUT_DIR = PROJECT_ROOT / "figures"


# Matplotlib defaults — keep figures legible when embedded in slides / PDF.
plt.rcParams.update({
    "figure.dpi":         110,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":        "sans-serif",
})


def _save(fig, name: str) -> None:
    """Save a figure to figures/<name> and close it to free memory."""
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"    ✓ {out.relative_to(PROJECT_ROOT)}")


def plot_category_distribution(df: pd.DataFrame) -> None:
    """Top-15 categories by prompt count — horizontal bar chart."""
    top = df["category"].value_counts().head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#4ADE80")
    ax.set_xlabel("Number of prompts")
    ax.set_title(f"Top 15 categories  (total = {len(df):,} prompts)")
    for i, v in enumerate(top.values):
        ax.text(v + max(top.values) * 0.01, i, str(v), va="center", fontsize=9)
    _save(fig, "category_distribution.png")


def plot_difficulty_distribution(df: pd.DataFrame) -> None:
    """Difficulty levels — vertical bar chart, ordered by intended skill."""
    order = ["beginner", "intermediate", "advanced", "expert"]
    counts = df["difficulty"].value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#FB923C")
    ax.set_ylabel("Number of prompts")
    ax.set_title("Difficulty distribution")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(counts.values) * 0.01,
                str(v), ha="center", fontsize=9)
    _save(fig, "difficulty_distribution.png")


def plot_language_distribution(df: pd.DataFrame) -> None:
    """Top-8 ISO language codes — pie chart."""
    counts = df["language"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    ax.pie(counts.values, labels=counts.index, colors=colors,
           autopct="%1.1f%%", startangle=90, pctdistance=0.85)
    ax.set_title(f"Language distribution  (top {len(counts)})")
    _save(fig, "language_distribution.png")


def plot_engagement_distribution(df: pd.DataFrame) -> None:
    """Histograms of likes / upvotes / uses on log(1+x) scale."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = [("likes", "#4ADE80"), ("upvotes", "#FB923C"), ("uses", "#60A5FA")]
    for ax, (field, color) in zip(axes, fields):
        values = np.log1p(df[field].fillna(0).astype(float))
        ax.hist(values, bins=40, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(f"log(1 + {field})")
        ax.set_xlabel(f"log(1 + {field})")
        ax.set_ylabel("Number of prompts")
    fig.suptitle("Engagement signals (log scale)", fontsize=13)
    _save(fig, "engagement_distribution.png")


def plot_content_length(df: pd.DataFrame) -> None:
    """Distribution of prompt content character length."""
    lengths = df["content"].fillna("").str.len()
    p99 = lengths.quantile(0.99)  # clip the long tail for readability

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths.clip(upper=p99), bins=50, color="#A78BFA",
            edgecolor="white", linewidth=0.4)
    ax.axvline(lengths.median(), color="#4ADE80", linestyle="--",
               linewidth=1.5, label=f"median = {int(lengths.median())}")
    ax.axvline(lengths.mean(), color="#FB923C", linestyle="--",
               linewidth=1.5, label=f"mean = {int(lengths.mean())}")
    ax.set_xlabel("content length (characters, clipped at 99th percentile)")
    ax.set_ylabel("Number of prompts")
    ax.set_title("Prompt content length distribution")
    ax.legend()
    _save(fig, "content_length.png")


def plot_uses_vs_views(df: pd.DataFrame, sample_n: int = 3000) -> None:
    """log-log scatter of uses vs views — engagement quality proxy."""
    sub = df[(df["views"] > 0) & (df["uses"] > 0)].copy()
    if len(sub) > sample_n:
        sub = sub.sample(n=sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub["views"], sub["uses"], s=10, alpha=0.35, color="#4ADE80")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("views  (log)")
    ax.set_ylabel("uses  (log)")
    ax.set_title(f"Uses vs Views  (n = {len(sub):,} sampled)")
    _save(fig, "uses_vs_views.png")


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading dataset from {DATASET_PATH.name}...")
    df = load_data(str(DATASET_PATH))
    print(f"    {len(df):,} prompts loaded.\n")

    print(f"Generating figures into {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    plot_category_distribution(df)
    plot_difficulty_distribution(df)
    plot_language_distribution(df)
    plot_engagement_distribution(df)
    plot_content_length(df)
    plot_uses_vs_views(df)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} PNG files written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
