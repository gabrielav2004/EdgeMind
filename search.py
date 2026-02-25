import numpy as np
import struct
import time
import re
from config import DB_FILE, IDX_FILE, TOP_K, RERANK_CANDIDATES
from models_cache import get_embedding_model

VECTOR_BYTES = 48

STOPWORDS = {
    'how', 'many', 'what', 'is', 'are', 'the', 'a', 'an',
    'for', 'of', 'in', 'to', 'was', 'were', 'made', 'did',
    'do', 'does', 'have', 'has', 'had', 'be', 'been', 'being',
    'at', 'by', 'from', 'with', 'about', 'which', 'who', 'when'
}

def get_embedding(text):
    from config import EMBEDDING_MODEL
    if "bge" in EMBEDDING_MODEL.lower():
        text = f"Represent this sentence for searching relevant passages: {text}"
    return get_embedding_model().encode([text], normalize_embeddings=True)[0]

def quantize_to_binary(embedding):
    threshold = np.mean(embedding)
    binary = (embedding > threshold).astype(np.uint8)
    packed = np.packbits(binary)
    return packed

def load_all_vectors():
    offsets = []
    with open(IDX_FILE, 'rb') as idx:
        while True:
            data = idx.read(8)
            if not data:
                break
            offsets.append(struct.unpack('Q', data)[0])

    vectors = []
    chunks = []
    with open(DB_FILE, 'rb') as db:
        for offset in offsets:
            db.seek(offset)
            vector_bytes = db.read(VECTOR_BYTES)
            packed = np.frombuffer(vector_bytes, dtype=np.uint8).copy()
            chunk_len = struct.unpack('H', db.read(2))[0]
            chunk = db.read(chunk_len).decode('utf-8')
            vectors.append(packed)
            chunks.append(chunk)

    return np.array(vectors), chunks

def hamming_search(query_vector, vectors, chunks, top_k=RERANK_CANDIDATES):
    xor = np.bitwise_xor(query_vector, vectors)
    distances = np.array([np.unpackbits(row).sum() for row in xor])
    top_indices = np.argsort(distances)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'distance': int(distances[idx]),
            'score': 1 - (distances[idx] / (VECTOR_BYTES * 8))
        })

    return results

def extract_names(query):
    cleaned = re.sub(
        r'(?i)^(who is|who was|tell me about|what is|about)\s+', '', query
    ).strip()
    names = re.findall(r'\b[A-Z][a-z]+\b', cleaned)
    return names

def keyword_boost(query, results):
    keywords = [
        w.lower() for w in query.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]

    names = extract_names(query)
    has_names = len(names) > 0

    if has_names:
        w_semantic = 0.4
        w_keyword  = 0.3
        w_name     = 0.3
    else:
        w_semantic = 0.5
        w_keyword  = 0.5
        w_name     = 0.0

    for result in results:
        chunk_lower = result['chunk'].lower()

        keyword_score = (
            sum(1 for kw in keywords if kw in chunk_lower) /
            max(len(keywords), 1)
        )

        name_score = (
            sum(1 for name in names if name.lower() in chunk_lower) /
            max(len(names), 1)
        ) if has_names else 0.0

        boosted_score = (
            result['float_score'] * w_semantic +
            keyword_score * w_keyword +
            name_score * w_name
        )

        # never let boosting hurt a good semantic score
        result['final_score'] = max(boosted_score, result['float_score'] * 0.9)

    return sorted(results, key=lambda x: x['final_score'], reverse=True)

def rerank(query, results, top_k=TOP_K):
    if not results:
        return results

    query_embedding = get_embedding_model().encode(
        [query], normalize_embeddings=True)[0]
    chunks = [r['chunk'] for r in results]
    chunk_embeddings = get_embedding_model().encode(
        chunks, normalize_embeddings=True)

    # dot product on normalized vectors — faster than cosine, same accuracy
    scores = chunk_embeddings @ query_embedding

    for i, result in enumerate(results):
        result['float_score'] = float(scores[i])

    results = keyword_boost(query, results)

    return results[:top_k]

def search(query, top_k=TOP_K, rerank_candidates=RERANK_CANDIDATES):
    print(f"\nquery: '{query}'")
    print("searching...")

    start = time.time()
    embedding = get_embedding(query)
    query_vector = quantize_to_binary(embedding)
    vectors, chunks = load_all_vectors()
    candidates = hamming_search(query_vector, vectors, chunks, top_k=rerank_candidates)
    results = rerank(query, candidates, top_k=top_k)
    elapsed = (time.time() - start) * 1000

    print(f"search completed in {elapsed:.1f}ms")
    print(f"\n--- top {top_k} results ---")
    for i, r in enumerate(results):
        print(f"\n{i+1}. score: {r['final_score']:.3f} | float: {r['float_score']:.3f} | distance: {r['distance']} bits")
        print(f"   '{r['chunk']}'")

    return results

if __name__ == "__main__":
    search("who is Karthik")
    search("how many registrations made for community day")
    search("who is Nephele")
    search("what torque does a robot motor need")