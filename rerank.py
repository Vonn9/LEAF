"""
File: rerank.py
Role: Reranking module using cross-encoder models from sentence-transformers.
Description:
    Takes the initial retrieval results and applies a cross-encoder to re-score and re-rank them based on relevance to the query.
    Using "BAAI/bge-reranker-v2-m3" as the default model, which is with the embedding model "BAAI/bge-base-en-v1.5" to ensure compatibility and optimal performance.
Input : query (str) + list of candidate dicts from extract_results()
Output: same list, sorted by reranker score (descending)
"""
from sentence_transformers import CrossEncoder

def load_reranker(reranker_model_name = "BAAI/bge-reranker-v2-m3") -> CrossEncoder:
    """
    Load the cross-encoder model for re-ranking.

    Args:
        reranker_model_name (str): The name of the cross-encoder model to load.

    Returns:
        CrossEncoder: The loaded cross-encoder model.
    """
    print(f"Loading reranker model: {reranker_model_name}...")
    reranker_model = CrossEncoder(reranker_model_name)
    print("Reranker model loaded successfully.")
    return reranker_model

def cross_encode(query: str, candidates: list, reranker: CrossEncoder) -> list:
    if not candidates:
        return []
    
    # build (query, document) pairs for cross-encoder with same text embedded : title + content
    pairs = []
    for c in candidates:
        meta = c['metadata']
        title = meta.get('title', '')
        content = c['content']
        doc_text = f"{title}. {content}" if title else content
        pairs.append((query, doc_text))

    scores = reranker.predict(pairs, show_progress_bar=True)

    # Attach scores to candidates
    for i, c in enumerate(candidates):
        c['reranker_score'] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x['reranker_score'], reverse=True)
    return reranked

if __name__ == "__main__":
    # Minimal test: load model and rerank two fake candidates
    reranker_model_name = "BAAI/bge-reranker-v2-m3"
    reranker = load_reranker(reranker_model_name)

    fake_candidates = [
        {
            "content": "Write a Python function to reverse a string.",
            "metadata": {"title": "Reverse string", "likes": 10, "upvotes": 5,
                         "author_reputation": 20, "category": "coding", "id": "001"}
        },
        {
            "content": "Create a marketing email for a product launch.",
            "metadata": {"title": "Marketing email", "likes": 50, "upvotes": 30,
                         "author_reputation": 80, "category": "marketing", "id": "002"}
        },
    ]
 
    query = "email to marketing department about product"
    results = cross_encode(query, fake_candidates, reranker)
 
    print("\nReranked results:")
    for r in results:
        print(f"  [{r['reranker_score']:.4f}] {r['metadata']['title']}")