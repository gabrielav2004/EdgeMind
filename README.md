# EdgeMind

A lightweight semantic retrieval system for edge devices and robotics. No cloud required, no vector database, minimal runtime dependencies.

## The Problem

Semantic search typically requires vector databases, float32 embeddings, and cloud-hosted models — too heavy for edge hardware. EdgeMind replaces all of that with binary embeddings in a flat binary file, retrieved via three-stage retrieval, with responses from a fully local or cloud model of your choice.

## How It Works

**Ingestion (runs once):**
```
documents → LLM formatter → clean structured text → chunk → embed → binary quantize → flat binary file
```

**Retrieval + Response:**
```
query → BGE embed → hamming search → dot product rerank → keyword + name boost → model → answer
```

Three-stage retrieval: hamming distance finds candidates fast, dot product reranks precisely, keyword and name boost fixes exact match misses. A floor ensures boosting never hurts a good semantic score.

## Why Binary Embeddings

A 384-dimensional float32 embedding is 1536 bytes. Binary quantization converts each dimension to a single bit based on mean threshold — the same embedding becomes 48 bytes. Retrieval uses hamming distance which is a native CPU bit operation.

Results from testing:
- 32x storage reduction vs float32
- High top-1 retrieval accuracy on domain-specific corpora
- Sub-millisecond hamming search at thousands of chunks
- No vector database required

## Project Structure

```
EdgeMind/
  config.py        # all settings in one place — edit this file only
  models_cache.py  # singleton embedding model, loads once
  parse.py         # ingest + LLM formatter + chunker
  store.py         # embed, quantize, write to binary database
  search.py        # three-stage retrieval engine
  respond.py       # local and cloud response generation
  run.py           # CLI interface
  serve.py         # FastAPI service
  setup.py         # one-command setup
  data/
    knowledge.bin  # flat binary database
    knowledge.idx  # byte offset index
    docs/          # place documents here
  models/          # local GGUF model files
```

## Installation

```bash
git clone https://github.com/you/edgemind
cd edgemind
python setup.py
```

Or manually:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` — this is the only file you need to touch:

**Local mode:**
```python
MODE = "local"
MODEL_PATH = "models/tinyllama.gguf"
```

**Cloud mode (any OpenAI compatible provider):**
```python
MODE = "cloud"
API_KEY = "your-key"
API_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama3-8b-8192"
```

**Anthropic:**
```python
MODE = "anthropic"
API_KEY = "your-anthropic-key"
MODEL_NAME = "claude-haiku-4-5-20251001"
```

**LLM formatter (runs at ingestion time):**
```python
USE_LLM_FORMATTER = True   # rewrite docs into clean structured text before chunking
```

## Compatible Cloud Providers

| Provider | BASE_URL | Notes |
|----------|----------|-------|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini recommended |
| Groq | https://api.groq.com/openai/v1 | free tier, very fast |
| Gemini | https://generativelanguage.googleapis.com/v1beta/openai/ | gemini-1.5-flash |
| Ollama | http://localhost:11434/v1 | local, no API key needed |
| Anthropic | use MODE = "anthropic" | separate SDK |

## Usage

```bash
# ingest documents
python run.py ingest data/docs

# single query
python run.py query "how many registrations were made"

# interactive mode
python run.py interactive

# start API server
python serve.py
```

## API

```bash
# health check
curl http://localhost:8000/health

# query with response
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "how many registrations were made"}'

# retrieval only no LLM
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "how many registrations", "respond": false}'

# ingest documents
curl -X POST "http://localhost:8000/ingest?folder=data/docs"
```

Interactive API docs at `http://localhost:8000/docs`

## Retrieval Architecture

**Stage 1 — Binary hamming search**
Query embedded via BGE, quantized to binary using mean threshold (preserves more information than sign-based quantization). XORed against all stored vectors. Top 20 candidates by minimum hamming distance.

**Stage 2 — Dot product rerank**
Top 20 candidates rescored using dot product on normalized vectors. Mathematically equivalent to cosine similarity but faster. Fixes rank-2 semantic misses.

**Stage 3 — Keyword and name boost**
Final scores blended with exact keyword match rate and proper name match rate. Fixes cases where numbers, names, or specific facts rank below semantically similar but factually wrong chunks. A floor ensures boosting never hurts a good semantic score:
```
final_score = max(boosted_score, float_score * 0.9)
```

Dynamic weights based on query type:
- Name query: semantic 40% + keyword 30% + name 30%
- Regular query: semantic 50% + keyword 50%

## Embedding Model

EdgeMind uses `BAAI/bge-small-en-v1.5` — a retrieval-optimized embedding model that significantly outperforms general purpose models like MiniLM on domain-specific corpora.

BGE queries are automatically prefixed for optimal retrieval:
```
"Represent this sentence for searching relevant passages: <query>"
```

Configurable in config.py:
```python
# "BAAI/bge-small-en-v1.5"  — 33MB, retrieval optimized (default)
# "BAAI/bge-base-en-v1.5"   — 109MB, better quality
# "all-MiniLM-L6-v2"        — 22MB, general purpose
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

## LLM Formatter

At ingestion time EdgeMind optionally rewrites documents into clean structured paragraphs using the configured model. Each paragraph covers exactly one topic, making chunking more precise and retrieval more accurate.

The formatter uses the same provider configured in config.py — local TinyLlama, Groq, Gemini, or any other provider. Set `USE_LLM_FORMATTER = False` to skip formatting and use rule-based cleaning only.

## Database Format

`knowledge.bin` — one entry per chunk:
```
[48 bytes binary vector][2 bytes chunk length][raw utf-8 bytes]
```

`knowledge.idx` — one entry per chunk:
```
[8 bytes uint64 offset into knowledge.bin]
```

Two flat files. No third-party database.

## Chunking

Documents split at natural boundaries in priority order: paragraph → sentence → clause → word. Overlap position snapped forward to next sentence start — chunks never begin mid-sentence.

```python
CHUNK_SIZE = 400    # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
```

## Local Model Options

| Model | Size | Quality | RAM |
|-------|------|---------|-----|
| TinyLlama 1.1B Q4 | 670MB | good | 2GB+ |
| TinyLlama 1.1B Q2 | 400MB | decent | 1.5GB+ |
| Qwen2 0.5B Q4 | 350MB | decent | 1GB+ |
| SmolLM2 135M Q4 | 95MB | minimal | 512MB+ |

## Performance

| Metric | Value |
|--------|-------|
| Storage per chunk | ~50 bytes + text |
| Hamming search (1k chunks) | < 5ms |
| Dot product rerank (20 candidates) | ~20ms |
| Total retrieval latency | ~50ms |
| Embedding model size | 33MB |

## Supported Input Formats

- `.txt` — plain text
- `.pdf` — via pypdf
- `.json` — recursively flattened to text

## Design Principle

Everything heavy runs at ingestion time on a capable machine. Everything that runs on the edge device is minimal. The knowledge base is two files. Retrieval is arithmetic. The model is local or configurable.

## Roadmap

- [x] Binary embedding pipeline with mean threshold quantization
- [x] Flat binary database
- [x] Three-stage retrieval: hamming + dot product rerank + keyword/name boost
- [x] Smart chunking at sentence boundaries
- [x] LLM formatter at ingestion
- [x] BGE retrieval-optimized embedding model
- [x] Multi-provider response generation
- [x] FastAPI service
- [x] Unified config system
- [ ] llama.cpp embeddings — eliminate HuggingFace dependency
- [ ] C implementation of hamming search
- [ ] Single binary deployment
- [ ] Raspberry Pi 4 validated deployment