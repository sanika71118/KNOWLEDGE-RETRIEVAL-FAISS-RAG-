# query.py
import os
import pickle
import numpy as np
import faiss
from config import USE_OPENAI, OPENAI_API_KEY, EMBED_MODEL, FAISS_INDEX_PATH, DOCS_PICKLE_PATH
from openai import OpenAI

# Load OpenAI client
def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)

# Embed query using OpenAI
def embed_query(text: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    vec = vec.reshape(1, -1)          # <-- make it 2D
    faiss.normalize_L2(vec)           # now works
    return vec


# Load FAISS index and documents
def load_index_and_docs():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PICKLE_PATH):
        raise FileNotFoundError("FAISS index or docs pickle not found. Run ingest.py first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCS_PICKLE_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs

# Search FAISS for top-k similar chunks
def search_index(query_vec, index, docs, top_k=5):
    D, I = index.search(query_vec, top_k)
    results = []
    for idx in I[0]:
        results.append(docs[idx]["text"])
    return results

def main():
    index, docs = load_index_and_docs()
    print("FAISS index and documents loaded. Type your questions (type 'exit' to quit).")

    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ["exit", "quit"]:
            print("Exiting…")
            break

        if USE_OPENAI:
            q_vec = embed_query(q)
            answers = search_index(q_vec, index, docs)
            print("\nTop relevant chunks:")
            for i, a in enumerate(answers, 1):
                print(f"{i}. {a[:200]}{'...' if len(a) > 200 else ''}")
        else:
            print("USE_OPENAI=False is not supported currently.")

if __name__ == "__main__":
    main()
