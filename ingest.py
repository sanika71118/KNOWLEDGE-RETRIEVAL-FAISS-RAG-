# ingest.py
import os, glob, pickle
from typing import List, Dict
import numpy as np
import faiss
from tqdm import tqdm
from config import (
    USE_OPENAI, EMBED_MODEL, OPENAI_API_KEY,
    DATA_DIR, INDEX_DIR, FAISS_INDEX_PATH, DOCS_PICKLE_PATH
)

def embed_openai(texts: List[str]) -> np.ndarray:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    return vecs

def embed_local(_):
    raise NotImplementedError("Set USE_OPENAI=True for now.")

def chunk_text(text: str, chunk_size=900, overlap=150) -> List[str]:
    chunks, i = [], 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        i = j - overlap if j - overlap > i else j
    return chunks

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}")

    docs: List[Dict] = []
    corpus: List[str] = []

    print(f"Reading {len(files)} files…")
    for p in files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        for ch in chunk_text(raw):
            docs.append({"path": p, "text": ch})
            corpus.append(ch)

    print(f"Created {len(corpus)} chunks. Embedding (USE_OPENAI={USE_OPENAI})…")
    if USE_OPENAI:
        vecs = embed_openai(corpus)
    else:
        vecs = embed_local(corpus)

    # cosine similarity via normalized vectors + inner product index
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCS_PICKLE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Saved index → {FAISS_INDEX_PATH}")
    print(f"Saved docs  → {DOCS_PICKLE_PATH}")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
