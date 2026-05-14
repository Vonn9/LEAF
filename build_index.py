"""
File: build_index.py
Role: Create a local vector database using ChromaDB and store the preprocessed data with their corresponding embeddings in the database.
Description:
    Load data by dataload.py
    Preprocess data by processing.py
    Generate embeddings by embedding.py
    clean old database if exists to ensure reproducibility
    The database is stored locally at './chroma_db' and can be queried for retrieval tasks.
"""

import os
import numpy as np
import chromadb
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dataload import load_data
from processing import preprocess_data
from embedding import embedding

# Inputs and regenerated artifacts live one level up from src/ (project root),
# so chroma_db/ and title_vecs.npz never travel inside src.zip.
_HERE = os.path.dirname(os.path.abspath(__file__))           # src/
_ROOT = os.path.dirname(_HERE)                                # parent of src/

TITLE_VECS_PATH = os.path.join(_ROOT, "title_vecs.npz")


def main():


    # 1. Configuration paths
    dataset_path = os.path.join(_ROOT, "dataset.json")
    db_path      = os.path.join(_ROOT, "chroma_db")
    collection_name = 'prompt_collection'
    model_name = 'BAAI/bge-base-en-v1.5'
    # Clean old DB (important for reproducibility)
    if os.path.exists(db_path):
        print("Removing old database...")
        shutil.rmtree(db_path)  

    # 2. Load and preprocess the dataset
    print(f"[1/4] Loading and preprocessing data from {dataset_path}...")
    df = load_data(dataset_path)
    df, metadatas = preprocess_data(df)
    print(f"Successfully loaded and preprocessed {len(df)} records.")

    # 3. Generate embeddings
    print(f"[2/4] Generating embeddings using model: {model_name}... (may take a few minutes)")
    embeddings = embedding(df, model_name)

    # 4. Initialize Vector Database (ChromaDB) and insert data
    print(f"[3/4] Initializing local ChromaDB at '{db_path}' and inserting data...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    batch_size = 5000
    total_records = len(df)
    ids = df['id'].astype(str).tolist()
    docs = df['text_to_embed'].tolist()
    for i in tqdm(range(0, total_records, batch_size), desc="Database Insertion"):
        end = min(i + batch_size, total_records)
        batch_embeddings = embeddings[i:end]
        batch_metadatas = metadatas[i:end]

        collection.add(
            ids=ids[i:end],
            documents=docs[i:end],
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
    
    print(f"Pipeline completed! Database is ready at '{db_path}'.")

    # 4. Precompute title embeddings (used by enrich_candidates for cosine_title)
    print(f"[4/4] Precomputing title embeddings → {TITLE_VECS_PATH}...")
    embedder = SentenceTransformer(model_name)
    titles   = df["title"].fillna("").tolist()
    doc_ids  = df["id"].astype(str).tolist()
    title_vecs = embedder.encode(
        titles, normalize_embeddings=True, batch_size=256, show_progress_bar=True
    )
    np.savez(TITLE_VECS_PATH, ids=np.array(doc_ids), vecs=title_vecs)
    print(f"Title embeddings saved ({len(doc_ids)} docs).")

if __name__ == "__main__":
    main()