# config.py
import os

# === cost/quality ===
USE_OPENAI = True
EMBED_MODEL = "text-embedding-3-small"   # cheap + good
CHAT_MODEL  = "gpt-4o-mini"              # low-cost for answers

# add your key as env var:  export/setx OPENAI_API_KEY=...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === paths ===
DATA_DIR = "./data"
INDEX_DIR = "./index_store"
FAISS_INDEX_PATH = f"{INDEX_DIR}/faiss.index"
DOCS_PICKLE_PATH = f"{INDEX_DIR}/docs.pkl"

# === retrieval ===
TOP_K = 4
MAX_DOC_CHARS_PER_DOC = 1200
MAX_TOTAL_CONTEXT_CHARS = 3500

SYSTEM_PROMPT = (
    "You are a concise assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know."
)
