"""
File: processing.py
Role: Data Preprocessing + Text Tokenization + Index Building
Description:
    Creates text_to_embed combining Title, Content, Category, Subcategory, and Tags
    for richer semantic embeddings — captures genre/domain keywords that live only in
    tags (e.g. "sci-fi", "horror") and never appear in the content text.
    Also creates metadata dicts used as XGBoost reranking features.

    Shared helpers used by both XGboost.py (training) and query.py (inference):
      - _tokenize / _tokenize_with_bigrams : stemmed token lists for BM25 and Jaccard
      - build_bm25_index                   : BM25Okapi over title+content
      - build_category_index               : category-name embeddings for category_match feature

Input: DataFrame from dataload.py
Output: DataFrame with text_to_embed column, and list of metadata dicts
"""

import numpy as np
from rank_bm25 import BM25Okapi
from nltk.stem import SnowballStemmer

_stemmer = SnowballStemmer("english")


def _tokenize(text: str) -> list:
    """Lowercase, split, and stem — used for Jaccard overlap and base tokenization."""
    return [_stemmer.stem(t) for t in text.lower().split()]


def _tokenize_with_bigrams(text: str) -> list:
    """Unigrams + bigrams for BM25: phrase "binary search" → bigram "binari_search" avoids matching on "search" alone."""
    tokens = _tokenize(text)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def build_bm25_index(df):
    """
    Build a BM25 index over title + content using stemmed bigram tokens.
    Returns (bm25, id_to_idx) where id_to_idx maps str(id) → corpus integer index.
    """
    corpus    = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist()
    tokenized = [_tokenize_with_bigrams(doc) for doc in corpus]
    bm25      = BM25Okapi(tokenized)
    id_to_idx = {str(row["id"]): idx for idx, row in df.iterrows()}
    return bm25, id_to_idx


def build_category_index(df, embedder) -> dict:
    """
    Encode all unique category names into normalized vectors.
    Returns {category_str: np.ndarray} — used to compute category_match feature.
    """
    unique_cats = df["category"].dropna().unique().tolist()
    cat_texts   = [c.replace("-", " ") for c in unique_cats]
    cat_vecs    = embedder.encode(cat_texts, normalize_embeddings=True,
                                  batch_size=64, show_progress_bar=False)
    return {cat: vec for cat, vec in zip(unique_cats, cat_vecs)}


def preprocess_data(df):
    def build_text(row):
        tags = row.get('tags', [])
        if not isinstance(tags, list):
            tags = []
        parts = [
            str(row.get('title',       '')).strip(),
            str(row.get('content',     '')).strip(),
            str(row.get('category',    '')).replace('-', ' ').strip(),
            str(row.get('subcategory', '')).replace('-', ' ').strip(),
            ' '.join(tags).replace('-', ' ').strip(),
        ]
        return '. '.join(p for p in parts if p)

    df['text_to_embed'] = df.apply(build_text, axis=1)

    rows = df.to_dict(orient="records")
    metadatas = [
        {
            "id": str(r["id"]),
            "title": str(r["title"]),
            "category": str(r.get("category", "Unknown")),
            "likes": int(r.get("likes", 0)),
            "upvotes": int(r.get("upvotes", 0)),
            "downvotes": int(r.get("downvotes", 0)),
            "views": int(r.get("views", 0)),
            "uses": int(r.get("uses", 0)),
            "fork_count": int(r.get("fork_count", 0)),
            "author_reputation": int(r.get("author_reputation", 0))
        }
        for r in rows
    ]
    return df, metadatas
