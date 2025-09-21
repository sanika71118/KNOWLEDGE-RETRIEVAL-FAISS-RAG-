import os, glob, pickle
from typing import List, Dict
import numpy as np
import faiss
from tqdm import tqdm
from config import USE_OPENAI, EMBED_MODEL, OPENAI_API_KEY, DATA_DIR, INDEX_DIR, FAISS_INDEX_PATH, DOCS_PICKLE_PATH, CACHE_PATH
from openai import OpenAI

def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_openai(texts: List[str], client: OpenAI) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

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

    # Load cached embeddings if available
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    docs: List[Dict] = []
    corpus: List[str] = []
    to_embed: List[str] = []

    print(f"Reading {len(files)} files…")
    for p in files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        for ch in chunk_text(raw):
            docs.append({"path": p, "text": ch})
            corpus.append(ch)
            if ch not in cache:
                to_embed.append(ch)

    print(f"{len(corpus)} total chunks, {len(to_embed)} new chunks to embed.")

    client = get_openai_client()
    if USE_OPENAI:
        if to_embed:
            new_vecs = embed_openai(to_embed, client)
            for i, text in enumerate(to_embed):
                cache[text] = new_vecs[i]

        # Construct vectors array
        vecs = np.array([cache[ch] for ch in corpus], dtype="float32")
    else:
        raise NotImplementedError("USE_OPENAI=False is not supported.")

    # Save FAISS index
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save docs and cache
    with open(DOCS_PICKLE_PATH, "wb") as f:
        pickle.dump(docs, f)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved index → {FAISS_INDEX_PATH}")
    print(f"Saved docs  → {DOCS_PICKLE_PATH}")
    print(f"Saved cache → {CACHE_PATH}")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
