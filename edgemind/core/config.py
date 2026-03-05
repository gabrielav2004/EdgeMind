# ============================================
# EdgeMind Configuration
# Edit this file to customize your setup
# ============================================

import os

from dotenv import load_dotenv

load_dotenv()

# --- MODE ---
# "local"     — runs fully offline using a local GGUF model
# "cloud"     — any openai compatible API (openai, groq, gemini, ollama)
# "anthropic" — anthropic claude models (separate SDK)
MODE = "cloud"

# --- LOCAL MODEL ---
# path to your GGUF model file
# options:
# "models/smollm-135m-instruct-v0.2-q4_k_m.gguf" — 80MB,  fastest
# "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  — 670MB, better quality
MODEL_PATH = "models/tinyllama.gguf"
N_THREADS = 4
N_CTX = 2048
N_GPU_LAYERS = 0

# --- CLOUD / ANTHROPIC ---
# for MODE = "cloud" or "anthropic"
API_KEY = "none"
MODEL_NAME = "tinyllama"

# only needed for MODE = "cloud"
# openai:  https://api.openai.com/v1
# groq:    https://api.groq.com/openai/v1
# gemini:  https://generativelanguage.googleapis.com/v1beta/openai/
# ollama:  http://localhost:11434/v1
API_BASE_URL = "http://localhost:8080/v1"

# --- GENERATION SETTINGS ---
TEMPERATURE = 0.1
MAX_TOKENS = 256

# --- EMBEDDING MODEL ---
# "all-MiniLM-L6-v2"        — 22MB, fast, good quality (default)
# "all-MiniLM-L12-v2"       — 33MB, better quality
# "paraphrase-MiniLM-L3-v2" — 17MB, fastest load
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# HUGGING FACE CONFIG (for authenticated inference...)

HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBEDDING_CACHE = "models/embeddings"

# --- RETRIEVAL SETTINGS ---
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RERANK_CANDIDATES = 20
TOP_K = 3

# --- SERVER SETTINGS ---
HOST = "0.0.0.0"
PORT = 8000

# --- PATHS ---
DB_FILE = "data/knowledge.bin"
IDX_FILE = "data/knowledge.idx"
DOCS_FOLDER = "data/docs"

# --- FORMATTER SETTINGS ---
USE_LLM_FORMATTER = False  # set False to skip LLM formatting at ingestion

# --- VALIDATION ---
def validate():
    errors = []

    if MODE not in ("local", "cloud", "anthropic"):
        errors.append("MODE must be 'local', 'cloud', or 'anthropic'")

    if MODE == "local" and not os.path.exists(MODEL_PATH):
        errors.append(f"local model not found at: {MODEL_PATH}")

    if MODE in ("cloud", "anthropic") and not API_KEY:
        errors.append("API_KEY is required for cloud and anthropic modes")

    if MODE == "cloud" and not API_BASE_URL:
        errors.append("API_BASE_URL is required for cloud mode")

    if MODE in ("cloud", "anthropic") and not MODEL_NAME:
        errors.append("MODEL_NAME is required for cloud and anthropic modes")

    if errors:
        print("\n=== EdgeMind config errors ===")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nfix these in config.py and try again")
        raise SystemExit(1)

validate()
