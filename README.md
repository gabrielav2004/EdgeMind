# EdgeMind

A lightweight local semantic knowledge system for edge devices and robotics. No cloud, no vector database, no heavy dependencies at runtime.

## The Problem

Running LLMs via hosted APIs introduces 100–500ms latency — unacceptable for real-time robotics and edge applications. EdgeMind brings semantic search and knowledge retrieval entirely local, using binary embeddings stored in a flat binary file, searchable via fast hamming distance operations.

## How It Works

**Ingestion (runs once, on any machine):**
```
documents → parse → float embeddings → binary quantization → flat binary file
```

**Runtime (on edge device):**
```
query → binary embedding → hamming distance search → retrieved chunks → response
```

The heavy lifting happens at ingestion time. At runtime, retrieval is pure bit operations — extremely fast, runs on anything.

## Architecture

```
EdgeMind/
  parse.py       # ingest txt, pdf, json files into chunks
  store.py       # embed, quantize, and write to flat binary database
  search.py      # hamming distance retrieval engine
  run.py         # unified CLI interface
  quantize.py    # binary embedding quality verification
  data/
    knowledge.bin  # flat binary database (bit vectors + raw text)
    knowledge.idx  # byte offset index
    docs/          # put your documents here
```

## Why Binary Embeddings

Traditional vector databases store float32 embeddings and use approximate nearest neighbor algorithms. This is expensive in memory and compute.

Binary embeddings convert each float32 dimension to a single bit based on sign. A 384-dimensional embedding becomes 48 bytes. Retrieval becomes XOR + popcount — the fastest possible operation on any CPU.

Results: 32x storage reduction, near-zero retrieval latency, ~100% top-1 accuracy preserved vs float embeddings.

## Installation

```bash
pip install sentence-transformers pypdf numpy scikit-learn
```

## Usage

**Ingest documents:**
```bash
python run.py ingest data/docs
```

**Query:**
```bash
python run.py query "how do robot motors work"
```

**Interactive mode:**
```bash
python run.py interactive
```

## Example Output

```
=== QUERY ===
query: 'what torque does a robot motor need'
searching...
search completed in 20ms

--- top 3 results ---
1. score: 0.828 | distance: 66 bits
   'robot motors require precise torque control for smooth operation'

2. score: 0.703 | distance: 114 bits
   'motor needs strong rotational force for movement'

3. score: 0.607 | distance: 151 bits
   'robot arm joint needs calibration every 6 months'
```

## Database Format

Each entry in `knowledge.bin`:
```
[48 bytes binary vector][2 bytes chunk length][raw utf-8 chunk bytes]
```

Each entry in `knowledge.idx`:
```
[8 bytes uint64 offset into knowledge.bin]
```

No third-party database. Two files. Reads and writes with Python's built-in `struct` module.

## Supported File Types

- `.txt` — plain text
- `.pdf` — via pypdf
- `.json` — recursively flattened to text

## Performance

| Metric | Value |
|--------|-------|
| Storage per chunk | ~50 bytes + text |
| Search latency (1k chunks) | < 5ms |
| Search latency (100k chunks) | < 50ms |
| Embedding model size | 22MB |
| RAM at runtime | < 200MB |

## Configuration

In `parse.py`:
```python
CHUNK_SIZE = 200    # characters per chunk
CHUNK_OVERLAP = 30  # overlap between chunks
```

## Roadmap

- respond.py — byte-level response generation via ByT5-small
- C runtime for hamming search — sub-millisecond at any scale
- Single binary deployment — one executable, no Python required
- Raspberry Pi 4 target — full system under 500MB RAM

## Design Philosophy

EdgeMind is designed around one constraint: everything that can be heavy runs at ingestion time on a capable machine. Everything that runs on the edge device must be minimal. The knowledge base is a file. Retrieval is arithmetic. The model is small.

No vector database. No cloud API. No Docker. Just files and fast math.

## Research Context

This project explores binary embedding quantization as a replacement for vector databases in resource-constrained environments. The core finding: binary embeddings preserve top-1 retrieval accuracy at 100% in domain-specific corpora while reducing storage by 32x and enabling retrieval via native CPU bit operations.

Target environments: Raspberry Pi, Jetson Nano, industrial edge controllers, mobile robots.