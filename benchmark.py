import numpy as np
import time
from edgemind.core.models_cache import get_embedding_model

# ============================================
# EdgeMind Benchmark
# Compares float cosine vs sign binary vs mean threshold binary
#
# Usage:
#   1. Add your own corpus and queries below
#   2. Set correct_idx to the index of the expected answer in corpus
#   3. python benchmark.py
# ============================================

# --------------------------------------------
# EDIT THIS SECTION WITH YOUR OWN DATA
# --------------------------------------------

CORPUS = [
    "Sentence one about your topic.",
    "Sentence two about your topic.",
    "Sentence three about your topic.",
    "Sentence four about your topic.",
    "Sentence five about your topic.",
    "Sentence six about your topic.",
    "Sentence seven about your topic.",
    "Sentence eight about your topic.",
    "Sentence nine about your topic.",
    "Sentence ten about your topic.",
]

# (query text, index of correct answer in CORPUS)
QUERIES = [
    ("your first test query", 0),
    ("your second test query", 2),
    ("your third test query", 4),
    ("your fourth test query", 6),
    ("your fifth test query", 8),
]

# BGE query prefix — keep this if using BAAI/bge models
# set to "" if using a different embedding model
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------
# BENCHMARK LOGIC — no need to edit below
# --------------------------------------------

def quantize_sign(embedding):
    binary = (embedding > 0).astype(np.uint8)
    return np.packbits(binary)

def quantize_mean(embedding):
    threshold = np.mean(embedding)
    binary = (embedding > threshold).astype(np.uint8)
    return np.packbits(binary)

def hamming_distance(a, b):
    xor = np.bitwise_xor(a, b)
    return int(np.unpackbits(xor).sum())

def print_separator(char="=", width=60):
    print(char * width)

def run_benchmark():
    print_separator()
    print("EdgeMind Benchmark")
    print_separator()

    model = get_embedding_model()

    # encode corpus
    print("\nencoding corpus...")
    t0 = time.time()
    corpus_embeddings = model.encode(CORPUS, normalize_embeddings=True)
    corpus_time = (time.time() - t0) * 1000
    print(f"  {len(CORPUS)} documents encoded in {corpus_time:.1f}ms")

    # encode queries
    print("encoding queries...")
    t0 = time.time()
    prefixed = [f"{BGE_PREFIX}{q}" for q, _ in QUERIES]
    query_embeddings = model.encode(prefixed, normalize_embeddings=True)
    query_time = (time.time() - t0) * 1000
    print(f"  {len(QUERIES)} queries encoded in {query_time:.1f}ms")

    # precompute binary vectors
    sign_corpus = np.array([quantize_sign(e) for e in corpus_embeddings])
    mean_corpus = np.array([quantize_mean(e) for e in corpus_embeddings])

    print("\n" + "=" * 60)
    print("PER QUERY RESULTS")
    print_separator()

    float_top1 = 0
    sign_top1 = 0
    mean_top1 = 0

    float_times = []
    sign_times = []
    mean_times = []

    for i, (query, correct_idx) in enumerate(QUERIES):
        q_emb = query_embeddings[i]

        # float cosine
        t0 = time.time()
        float_scores = corpus_embeddings @ q_emb
        float_rank = np.argsort(float_scores)[::-1]
        float_times.append((time.time() - t0) * 1000)
        float_hit = int(correct_idx == float_rank[0])
        float_top1 += float_hit

        # sign binary
        q_sign = quantize_sign(q_emb)
        t0 = time.time()
        sign_distances = np.array([hamming_distance(q_sign, v) for v in sign_corpus])
        sign_rank = np.argsort(sign_distances)
        sign_times.append((time.time() - t0) * 1000)
        sign_hit = int(correct_idx == sign_rank[0])
        sign_top1 += sign_hit

        # mean threshold binary
        q_mean = quantize_mean(q_emb)
        t0 = time.time()
        mean_distances = np.array([hamming_distance(q_mean, v) for v in mean_corpus])
        mean_rank = np.argsort(mean_distances)
        mean_times.append((time.time() - t0) * 1000)
        mean_hit = int(correct_idx == mean_rank[0])
        mean_top1 += mean_hit

        print(f"\nquery {i+1}: '{query}'")
        print(f"  float cosine : rank {int(np.where(float_rank == correct_idx)[0][0]) + 1} {'✓' if float_hit else '✗'}  score={float_scores[correct_idx]:.3f}")
        print(f"  sign binary  : rank {int(np.where(sign_rank == correct_idx)[0][0]) + 1} {'✓' if sign_hit else '✗'}  dist={sign_distances[correct_idx]}")
        print(f"  mean binary  : rank {int(np.where(mean_rank == correct_idx)[0][0]) + 1} {'✓' if mean_hit else '✗'}  dist={mean_distances[correct_idx]}")

    total = len(QUERIES)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print_separator()
    print(f"{'Method':<22} {'Top-1':<10} {'Accuracy':<12} {'Avg Latency'}")
    print("-" * 60)
    print(f"{'Float cosine':<22} {float_top1}/{total}{'':>4} {float_top1/total*100:.0f}%{'':>8} {np.mean(float_times):.3f}ms")
    print(f"{'Sign binary':<22} {sign_top1}/{total}{'':>4} {sign_top1/total*100:.0f}%{'':>8} {np.mean(sign_times):.3f}ms")
    print(f"{'Mean binary (EdgeMind)':<22} {mean_top1}/{total}{'':>4} {mean_top1/total*100:.0f}%{'':>8} {np.mean(mean_times):.3f}ms")

    print("\n" + "=" * 60)
    print("STORAGE")
    print_separator()
    dims = len(corpus_embeddings[0])
    print(f"Embedding dimensions  : {dims}")
    print(f"Float32 per embedding : {dims * 4} bytes")
    print(f"Binary per embedding  : {dims // 8} bytes")
    print(f"Compression ratio     : {(dims * 4) / (dims // 8):.0f}x")

    print("\n" + "=" * 60)
    print("SYSTEM")
    print_separator()
    import platform, os
    print(f"Python  : {platform.python_version()}")
    print(f"OS      : {platform.system()} {platform.release()}")
    print(f"CPU     : {os.cpu_count()} cores")
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"RAM     : {ram.total // (1024**2)}MB total, {ram.available // (1024**2)}MB available")
    except ImportError:
        print("RAM     : pip install psutil for RAM info")

if __name__ == "__main__":
    run_benchmark()
