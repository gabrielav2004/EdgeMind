import numpy as np
import struct
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

DB_FILE = "data/knowledge.bin"
IDX_FILE = "data/knowledge.idx"
VECTOR_BYTES = 48

def get_embedding(text):
    return model.encode([text], normalize_embeddings=True)[0]

def quantize_to_binary(embedding):
    binary = (embedding > 0).astype(np.uint8)
    packed = np.packbits(binary)
    return packed

def load_all_vectors():
    # load index
    offsets = []
    with open(IDX_FILE, 'rb') as idx:
        while True:
            data = idx.read(8)
            if not data:
                break
            offsets.append(struct.unpack('Q', data)[0])
    
    # load all vectors and chunks
    vectors = []
    chunks = []
    with open(DB_FILE, 'rb') as db:
        for offset in offsets:
            db.seek(offset)
            vector_bytes = db.read(VECTOR_BYTES)
            packed = np.frombuffer(vector_bytes, dtype=np.uint8)
            chunk_len = struct.unpack('H', db.read(2))[0]
            chunk = db.read(chunk_len).decode('utf-8')
            vectors.append(packed)
            chunks.append(chunk)
    
    return np.array(vectors), chunks

def hamming_search(query_vector, vectors, chunks, top_k=3):
    # XOR query against all vectors at once
    xor = np.bitwise_xor(query_vector, vectors)
    
    # count set bits per row (hamming distance)
    distances = np.array([np.unpackbits(row).sum() for row in xor])
    
    # get top_k smallest distances
    top_indices = np.argsort(distances)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'distance': int(distances[idx]),
            'score': 1 - (distances[idx] / (VECTOR_BYTES * 8))
        })
    
    return results

def search(query, top_k=3):
    print(f"\nquery: '{query}'")
    print("searching...")
    
    # embed and quantize query
    start = time.time()
    embedding = get_embedding(query)
    query_vector = quantize_to_binary(embedding)
    
    # load db and search
    vectors, chunks = load_all_vectors()
    results = hamming_search(query_vector, vectors, chunks, top_k)
    elapsed = (time.time() - start) * 1000
    
    print(f"search completed in {elapsed:.1f}ms")
    print(f"\n--- top {top_k} results ---")
    for i, r in enumerate(results):
        print(f"\n{i+1}. score: {r['score']:.3f} | distance: {r['distance']} bits")
        print(f"   '{r['chunk']}'")
    
    return results

if __name__ == "__main__":
    search("what torque does a robot motor need")
    search("how is the weather")
    search("when should i calibrate the arm")