# EdgeMind

A lightweight semantic retrieval and response system for edge devices and robotics. No cloud, no vector database, no heavy runtime dependencies.

## The Problem

Semantic search and local inference typically require vector databases, float32 embeddings, and cloud-hosted models — too heavy for edge hardware. EdgeMind replaces all of that with binary embeddings in a flat binary file and a locally running quantized model.

## How It Works

**Ingestion (runs once):**
```
documents → chunk → embed → binary quantize → flat binary file
```

**Retrieval + Response (runs on edge):**
```
query → binary embed → hamming distance search → relevant chunks → local model → answer
```

The embedding model runs at ingestion time on any capable machine. At retrieval time, search is pure bit operations. The response model runs fully local via llama.cpp — no internet required.

## Why Binary Embeddings

A 384-dimensional float32 embedding is 1536 bytes. Binary quantization converts each dimension to a single bit based on sign — the same embedding becomes 48 bytes. Retrieval uses hamming distance instead of cosine similarity, which is a native CPU operation requiring no floating point math.

Results from testing:
- 32x storage reduction vs float32
- 100% top-1 retrieval accuracy preserved on domain-specific corpora
- Sub-millisecond search at thousands of chunks
- No vector database required

## Project Structure

```
EdgeMind/
  parse.py       # ingest txt, pdf, json into chunks
  store.py       # embed, quantize, write to binary database
  search.py      # hamming distance retrieval
  respond.py     # local response generation via llama.cpp
  run.py         # unified CLI
  quantize.py    # verify binary embedding quality
  data/
    knowledge.bin  # flat binary database
    knowledge.idx  # byte offset index
    docs/          # place documents here
  models/
    tinyllama.gguf # local quantized model
```

## Installation

```bash
pip install sentence-transformers pypdf numpy scikit-learn huggingface_hub
```

Install llama-cpp-python:
```bash
pip install llama-cpp-python
```

If that fails on Windows, use the prebuilt wheel:
```bash
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Download the local model:
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "models\tinyllama.gguf"

# or via Python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', local_dir='models')"
```

## Usage

```bash
# ingest documents
python run.py ingest data/docs

# single query
python run.py query "how do robot motors work"

# interactive mode (keeps model resident, faster responses)
python run.py interactive
```

## Database Format

`knowledge.bin` — one entry per chunk:
```
[48 bytes binary vector][2 bytes chunk length][raw utf-8 bytes]
```

`knowledge.idx` — one entry per chunk:
```
[8 bytes uint64 offset into knowledge.bin]
```

No third-party database. Two flat files. Read and written with Python's built-in `struct` module.

## Performance

| Metric | Value |
|--------|-------|
| Storage per chunk | ~50 bytes + text |
| Search latency (1k chunks) | < 5ms |
| Search latency (100k chunks) | < 50ms |
| Embedding model size | 22MB |
| Response model size | 670MB (Q4) |
| RAM at runtime | < 1.5GB |

## Supported Input Formats

- `.txt` — plain text
- `.pdf` — via pypdf
- `.json` — recursively flattened to text

## Configuration

In `parse.py`:
```python
CHUNK_SIZE = 200    # characters per chunk
CHUNK_OVERLAP = 30  # overlap between chunks
```

In `respond.py`:
```python
MODEL_PATH = "models/tinyllama.gguf"
n_threads = 4       # cpu threads
n_ctx = 2048        # context window
```

## Model Options

| Model | Size | Quality | RAM needed |
|-------|------|---------|------------|
| TinyLlama Q4_K_M | 670MB | good | 2GB+ |
| TinyLlama Q2_K | 400MB | decent | 1.5GB+ |
| Qwen2-0.5B Q4 | 350MB | lighter | 1GB+ |
| SmolLM-135M Q4 | 80MB | minimal | 512MB+ |

## Design Principle

Everything heavy runs at ingestion time on a capable machine. Everything that runs on the edge device is minimal. The knowledge base is two files. Retrieval is arithmetic. The model is local.

## Roadmap

- [x] Binary embedding pipeline
- [x] Flat binary database
- [x] Hamming distance retrieval
- [x] Local response generation via llama.cpp
- [ ] C implementation of hamming search
- [ ] Single binary deployment, no Python dependency
- [ ] Raspberry Pi 4 validated deployment