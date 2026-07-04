from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# Folder Paths
# ==========================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = BASE_DIR / "uploads"

VECTOR_DB_DIR = BASE_DIR / "vector_db"


# ==========================================
# Embedding Model
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# LLM
# ==========================================

LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ==========================================
# Chunk Settings
# ==========================================

CHUNK_SIZE = 350

CHUNK_OVERLAP = 150


# ==========================================
# Retrieval
# ==========================================

TOP_K = 5


# ==========================================
# Generation
# ==========================================

MAX_NEW_TOKENS = 512

TEMPERATURE = 0.1