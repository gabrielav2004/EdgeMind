# EdgeMind Benchmark Results (Will Be Updated as benchmarking is done on more devices)

## Environment

| | |
|--|--|
| Platform | AWS t2.micro (constrained edge proxy) |
| CPU | 1 vCPU |
| RAM | 1GB |
| OS | Linux 6.14.0 AWS |
| Python | 3.12.3 |
| Embedding model | BAAI/bge-small-en-v1.5 |

---

## Retrieval Accuracy (Top-1)

5 queries tested against a 10-document corpus.

| Method | Top-1 | Accuracy | Avg Latency |
|--------|-------|----------|-------------|
| Float cosine (baseline) | 5/5 | 100% | 0.480ms |
| Sign binary | 5/5 | 100% | 0.321ms |
| Mean binary (EdgeMind) | 5/5 | 100% | **0.038ms** |

Mean threshold binary matches float cosine accuracy at **12x faster retrieval**.

---

## Per Query Results

| Query | Float cosine | Sign binary | Mean binary |
|-------|-------------|-------------|-------------|
| who is Mr. Karthik | rank 1 ✓ (0.723) | rank 1 ✓ (dist 101) | rank 1 ✓ (dist 102) |
| how many registrations for community day | rank 1 ✓ (0.748) | rank 1 ✓ (dist 119) | rank 1 ✓ (dist 119) |
| what is Nephele | rank 1 ✓ (0.763) | rank 1 ✓ (dist 93) | rank 1 ✓ (dist 94) |
| what model has 1.1B parameters | rank 1 ✓ (0.702) | rank 1 ✓ (dist 126) | rank 1 ✓ (dist 126) |
| what distance metric does binary search use | rank 1 ✓ (0.650) | rank 1 ✓ (dist 131) | rank 1 ✓ (dist 131) |

---

## Storage

| Format | Bytes per embedding | Compression |
|--------|-------------------|-------------|
| Float32 | 1536 bytes | 1x |
| Sign binary | 48 bytes | 32x |
| Mean binary (EdgeMind) | 48 bytes | **32x** |

Same storage footprint as sign binary — zero cost for the accuracy and speed advantage.

---

## Encoding Time

| Operation | Time |
|-----------|------|
| Corpus encoding (10 docs, cold start) | 571ms |
| Query encoding (5 queries) | 87ms |
| Subsequent query encoding (single) | ~17ms |

Encoding happens once at ingest time and once per query. Retrieval itself is sub-millisecond.

---

## Embedding Model

| | |
|--|--|
| Model | BAAI/bge-small-en-v1.5 |
| Download size | 133MB |
| Load time (cold) | ~2s |
| Load time (cached, offline) | < 1s |
| RAM usage | ~500MB |

---

## Key Findings

**Mean threshold vs sign binary:**
Mean threshold binary is 8x faster than sign binary (0.038ms vs 0.321ms) with identical accuracy. The mean threshold produces better bit distribution across the 384 dimensions — bits encode relative magnitude rather than just sign — which makes hamming distance more discriminative and faster to converge.

**Binary vs float cosine:**
Mean binary matches float cosine top-1 accuracy at 32x compression and 12x lower latency. The two-stage pipeline (binary hamming search → float dot product rerank) recovers any ranking errors from quantization before returning results.

**Edge viability:**
Full pipeline — embedding model + binary retrieval + cloud LLM — runs on a single vCPU with 1GB RAM. Add 1GB swap for headroom. Retrieval latency is sub-millisecond. The bottleneck on constrained hardware is embedding model load time on cold start, not retrieval.

---

## Reproduce

```bash
git clone https://github.com/gabrielav2004/EdgeMind
cd EdgeMind
./install.sh
pip install psutil
python benchmark.py
```
