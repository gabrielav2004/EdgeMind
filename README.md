# EdgeMind

A lightweight semantic retrieval system for edge devices and robotics. No cloud required, no vector database, minimal runtime dependencies.

## The Problem

Semantic search typically requires vector databases, float32 embeddings, and cloud-hosted models — too heavy for edge hardware. EdgeMind replaces all of that with binary embeddings in a flat binary file, retrieved via three-stage retrieval, with responses from a fully local or cloud model of your choice.

## How It Works

**Ingestion (runs once, on any machine):**
```
documents → optional LLM formatter → clean structured text → chunk → embed → binary quantize → flat binary file
```

**Retrieval + Response (runs on edge):**
```
query → BGE embed → hamming search → dot product rerank → keyword + name boost → gap filter → model → answer
```

## Portable Knowledge Base

One of EdgeMind's core design principles: **ingest on powerful hardware, deploy anywhere.**

```
Powerful machine (ingestion)
  - cloud LLM formatter (GPT-4, Claude, Groq)
  - BGE large embedding model
  - fast processing of large documents
  - outputs knowledge.bin + knowledge.idx
          ↓
    copy two files to edge device
          ↓
Edge device (runtime)
  - no LLM needed for retrieval
  - just loads binary file
  - runs hamming search
  - tiny memory footprint
```

The binary files have zero dependency on the machine, OS, Python version, or any library that created them. Once written they are portable forever — copy them to a Raspberry Pi, Jetson Nano, tablet, or any device and retrieval works instantly.

**The only constraint:** the embedding model must match on both machines:
```python
# must be identical on ingestion machine and edge device
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
```

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
  parse.py         # ingest + optional LLM formatter + chunker
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

**LLM formatter — disabled by default:**
```python
USE_LLM_FORMATTER = False  # see considerations below before enabling
```

## Compatible Cloud Providers

| Provider | BASE_URL | Notes |
|----------|----------|-------|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini recommended |
| Groq | https://api.groq.com/openai/v1 | free tier, very fast |
| Gemini | https://generativelanguage.googleapis.com/v1beta/openai/ | gemini-1.5-flash |
| Ollama | http://localhost:11434/v1 | local, no API key needed |
| Anthropic | use MODE = "anthropic" | separate SDK |

## Special Case Usage of OpenAI Compatible API in Cloud Mode

Add this section to the README under the **Deployment Architecture** section:

## llama.cpp Server Mode

Instead of running inference through Python bindings, llama.cpp can run as a standalone HTTP server. This is faster, keeps the model permanently warm, and exposes an OpenAI-compatible endpoint — no code changes needed in EdgeMind.

This is useful in cases you want to offload or centralize the response or formatting llm deployments.

**Install:**
```bash
pip install llama-cpp-python[server] --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir
```

**Run:**
```bash
python -m llama_cpp.server \
  --model models/tinyllama.gguf \
  --port 8080 \
  --n_threads 4
```

For constrained devices leave one core free for the OS:
```bash
# Raspberry Pi 4 — 4 cores, use 3
python -m llama_cpp.server \
  --model models/tinyllama.gguf \
  --port 8080 \
  --n_threads 3 \
  --n_ctx 512
```

**Test it's running:**
```bash
curl http://localhost:8080/v1/models
```

**Point EdgeMind to it in config.py:**
```python
MODE = "cloud"
API_BASE_URL = "http://localhost:8080/v1"
API_KEY = "none"
MODEL_NAME = "tinyllama"
```

**Why this is better than Python bindings:**
- No Python overhead in the inference loop
- Model stays loaded permanently — no reload between queries
- Same OpenAI-compatible API as Groq, Ollama, and other providers
- Works seamlessly with existing cloud mode in EdgeMind

**Thread count guidelines for edge devices:**

| Device | Physical Cores | Recommended Threads |
|--------|---------------|---------------------|
| Raspberry Pi 4 | 4 | 3 |
| Raspberry Pi 5 | 4 | 3 |
| Jetson Nano | 4 | 3 |
| Jetson Orin | 6 | 5 |
| Generic SBC | varies | total cores - 1 |

Always leave one core free for the OS. Using all cores makes the device unresponsive during inference.

**Expected inference speed on edge hardware with TinyLlama Q4:**

| Device | Tokens/sec |
|--------|-----------|
| Raspberry Pi 4 | ~3-4 t/s |
| Raspberry Pi 5 | ~6-8 t/s |
| Jetson Orin | ~15-20 t/s |

For faster inference on the same hardware use TinyLlama Q2 — approximately 2x faster than Q4 with acceptable quality loss.

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
Final scores blended with exact keyword match rate and proper name match rate. Dynamic weights based on query type:
- Name query: semantic 40% + keyword 30% + name 30%
- Regular query: semantic 50% + keyword 50%

A floor ensures boosting never hurts a good semantic score:
```
final_score = max(boosted_score, float_score * 0.9)
```

**Gap filter**
Before passing chunks to the response model, EdgeMind checks the score gap between rank 1 and rank 2. If rank 1 scores significantly higher than rank 2, only rank 1 is passed to the model. This prevents the model from getting confused by irrelevant lower-ranked chunks and hallucinating a combined answer.

## Response Architecture

Context is passed in the system role, question in the user role. This separation prevents the model from meta-commenting on the context and produces more direct answers.

```
system = instructions + context (ground truth)
user   = question only
```

For local models, the same separation is enforced via the chat template:
```
<|system|> instructions + context </s>
<|user|> question </s>
<|assistant|>
```

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

At ingestion time EdgeMind can optionally rewrite documents into clean structured paragraphs using the configured model before chunking.

**When to use it:**
- PDFs with broken line endings, hyphenation artifacts, or page break issues
- Documents with inconsistent paragraph structure
- Raw exports from tools that produce messy text

**When NOT to use it:**
- Clean text files — formatter adds no value and risks corrupting facts
- Factual documents with names, numbers, and dates — small and medium models hallucinate when reformatting, silently changing names, inventing organizations, and fabricating details that were never in the source
- Any document where factual accuracy is critical

**Known issues with LLM formatting:**
- Models echo back prompt instructions into the output — mitigated by `clean_formatter_output()` but not always fully caught
- Models hallucinate facts during reformatting — a model rewrote "founded by Mr. Karthik and Mr. Raghul Gopal" as "founded by Mr. Karthik and Mrs. Megha" and invented "University of Hyderabad" which does not appear anywhere in the source document
- Prompt leakage into chunks — formatter rules and labels end up stored in the knowledge base, polluting retrieval
- Only large capable models (GPT-4, Claude Opus) are reliable enough for factual reformatting — smaller models introduce more problems than they solve

**Recommendation:** Leave `USE_LLM_FORMATTER = False` unless you have a specific formatting problem that rule-based cleaning cannot fix, and use a large capable cloud model if you do enable it.

## Database Format

`knowledge.bin` — one entry per chunk:
```
[48 bytes binary vector][2 bytes chunk length][raw utf-8 bytes]
```

`knowledge.idx` — one entry per chunk:
```
[8 bytes uint64 offset into knowledge.bin]
```

Two flat files. No third-party database. Fully portable across machines and platforms.

## Chunking

Documents split at natural boundaries in priority order: paragraph → sentence → clause → word. Overlap position snapped forward to next sentence start — chunks never begin mid-sentence.

```python
CHUNK_SIZE = 400    # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
```

## Local Model Considerations

Small local models have significant limitations for response generation:

- **Context bleeding** — llama.cpp reuses KV cache between calls, causing previous answers to bleed into new ones. Fixed by calling `model.reset()` before each inference.
- **Hallucination** — models below 1B parameters frequently ignore context and fabricate answers. TinyLlama 1.1B is the minimum viable size for reliable instruction following.
- **Chunk blending** — small models combine information from multiple chunks even when chunks are unrelated, producing answers that mix facts from different topics. Fixed by the gap filter which limits context to the most relevant chunk when scores diverge.
- **Meta-commentary** — models describe how they will answer instead of answering. Fixed by moving context to the system role and simplifying the user message to the question only.

| Model | Size | Quality | RAM |
|-------|------|---------|-----|
| TinyLlama 1.1B Q4 | 670MB | good | 2GB+ |
| TinyLlama 1.1B Q2 | 400MB | decent | 1.5GB+ |
| Qwen2 0.5B Q4 | 350MB | poor | 1GB+ |
| SmolLM2 135M Q4 | 95MB | unusable | 512MB+ |

## Performance

| Metric | Value |
|--------|-------|
| Storage per chunk | ~50 bytes + text |
| Hamming search (1k chunks) | < 5ms |
| Dot product rerank (20 candidates) | ~20ms |
| Total retrieval latency | ~50ms |
| Embedding model size | 33MB |

## Deployment Architecture

For mobile and tablet deployments, run EdgeMind on a local edge server and have devices call the API over the network:

```
Raspberry Pi / edge server
  └── llama.cpp server (inference)
  └── EdgeMind serve.py (retrieval + API)
        ↑
   local WiFi / network
        ↑
  Mobile / Tablet
  (POST to /query, display answer)
```

Mobile devices need zero ML code — just HTTP calls to the API.

## Supported Input Formats

- `.txt` — plain text
- `.pdf` — via pypdf
- `.json` — recursively flattened to text

## Design Principle

Heavy ingestion runs on powerful hardware. Edge devices only run retrieval. The knowledge base is two portable files. Retrieval is arithmetic. The model is local or configurable.

## Roadmap

- [x] Binary embedding pipeline with mean threshold quantization
- [x] Flat binary database with portable binary format
- [x] Three-stage retrieval: hamming + dot product rerank + keyword/name boost
- [x] Gap filter to prevent chunk blending in response generation
- [x] Smart chunking at sentence boundaries
- [x] Context/question separation in system/user roles
- [x] BGE retrieval-optimized embedding model
- [x] Multi-provider response generation
- [x] FastAPI service
- [x] Unified config system
- [x] LLM formatter with fallback to rule-based cleaning
- [ ] Refactor into component folders (core, ingestion, retrieval, generation)
- [ ] llama.cpp embeddings — eliminate HuggingFace dependency
- [ ] Embedding provider support (ollama, openai, cohere)
- [ ] C implementation of hamming search
- [ ] Single binary deployment
- [ ] Raspberry Pi 4 validated deployment