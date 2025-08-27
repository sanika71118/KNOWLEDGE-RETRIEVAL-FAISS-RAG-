# query.py
import argparse, pickle
from typing import List, Tuple
import numpy as np
import faiss
from config import (
    USE_OPENAI, OPENAI_API_KEY, EMBED_MODEL, CHAT_MODEL,
    FAISS_INDEX_PATH, DOCS_PICKLE_PATH, TOP_K,
    MAX_DOC_CHARS_PER_DOC, MAX_TOTAL_CONTEXT_CHARS, SYSTEM_PROMPT
)

def embed_query_openai(q: str) -> np.ndarray:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    vec = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(vec)
    return vec

def search(qvec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    index = faiss.read_index(FAISS_INDEX_PATH)
    D, I = index.search(qvec, top_k)
    return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

def load_docs():
    with open(DOCS_PICKLE_PATH, "rb") as f:
        return pickle.load(f)

def build_context(docs, hits) -> str:
    # budget context across hits
    if not hits:
        return ""
    per_doc = min(MAX_DOC_CHARS_PER_DOC, MAX_TOTAL_CONTEXT_CHARS // max(1, len(hits)))
    parts = []
    total = 0
    for idx, score in hits:
        snippet = docs[idx]["text"]
        if len(snippet) > per_doc:
            snippet = snippet[:per_doc] + "…"
        part = f"[score={score:.3f} | source={docs[idx]['path']}]\n{snippet}"
        parts.append(part)
        total += len(snippet)
        if total >= MAX_TOTAL_CONTEXT_CHARS:
            break
    return "\n\n".join(parts)

def answer_openai(context: str, question: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="your question")
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()

    qvec = embed_query_openai(args.q) if USE_OPENAI else None
    hits = search(qvec, args.k)
    docs = load_docs()
    context = build_context(docs, hits)

    print("----- retrieved context -----")
    print(context or "(no context)")
    print("-----------------------------")

    if USE_OPENAI:
        print("\n=== answer ===")
        print(answer_openai(context, args.q))
    else:
        print("\n(OpenAI disabled) Implement local LLM call here.")

if __name__ == "__main__":
    main()
