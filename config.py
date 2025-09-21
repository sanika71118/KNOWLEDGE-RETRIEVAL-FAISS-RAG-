# config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration flags
USE_OPENAI = True
EMBED_MODEL = "text-embedding-3-small"  # or your preferred embedding model

# Paths
DATA_DIR = "data"
INDEX_DIR = "index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DOCS_PICKLE_PATH = os.path.join(INDEX_DIR, "docs.pkl")
