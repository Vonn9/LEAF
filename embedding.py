"""
File: embedding.py
Role: Embedding Generation
Description:
    Generates embeddings for text_to_embed
"""
from sentence_transformers import SentenceTransformer
def embedding(data, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        data['text_to_embed'].tolist(),
        show_progress_bar=True, 
        normalize_embeddings=True # For cosine similarity
    ).tolist()
    return embeddings