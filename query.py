import os
import pickle
import numpy as np
import faiss
from config import USE_OPENAI, OPENAI_API_KEY, EMBED_MODEL, FAISS_INDEX_PATH, DOCS_PICKLE_PATH
from openai import OpenAI

def get_openai_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def load_index_and_docs():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PICKLE_PATH):
        raise FileNotFoundError("FAISS index or docs pickle not found. Run ingest.py first.")
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
            top_chunks = search_index(q_vec, index, docs, top_k=5)

            # Show sources
            print("\nTop sources retrieved:")
            for i, c in enumerate(top_chunks, 1):
                print(f"{i}. {c['source']} -> {c['text'][:150]}{'...' if len(c['text'])>150 else ''}")

            # Generate final RAG answer
            context = "\n".join([c['text'] for c in top_chunks])
            answer = generate_answer(q, context)
            print("\nRAG Answer:\n")
            print(answer)
        else:
            print("USE_OPENAI=False is not supported.")

if __name__ == "__main__":
    main()
