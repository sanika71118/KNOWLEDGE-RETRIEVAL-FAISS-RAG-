# rag_backend.py
import os
import pickle
import numpy as np
import faiss
from config import USE_OPENAI, OPENAI_API_KEY, EMBED_MODEL, FAISS_INDEX_PATH, DOCS_PICKLE_PATH
from openai import OpenAI

def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def load_index_and_docs():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCS_PICKLE_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs

def search_index(query_vec, index, docs, top_k=5):
    D, I = index.search(query_vec, top_k)
    results = [{"text": docs[idx]["text"], "source": docs[idx]["path"]} for idx in I[0]]
    return results

def generate_answer(question: str, context: str) -> str:
    client = get_openai_client()
    prompt = f"Use the following context to answer the question clearly. Include references if possible.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def main_query(question: str, top_k=5):
    index, docs = load_index_and_docs()
    q_vec = embed_query(question)
    top_chunks = search_index(q_vec, index, docs, top_k)
    context = "\n".join([c['text'] for c in top_chunks])
    answer = generate_answer(question, context)
    return answer, top_chunks
