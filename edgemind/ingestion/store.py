import numpy as np
import struct
import os
from edgemind.core.config import DB_FILE, IDX_FILE
from edgemind.core.models_cache import get_embedding_model

def get_embeddings(texts):
    return get_embedding_model().encode(texts, normalize_embeddings=True)

def quantize_to_binary(embeddings):
    binary = (embeddings > 0).astype(np.uint8)
    packed = np.packbits(binary, axis=1)
    return packed

def init_db(overwrite=False):
    os.makedirs("data", exist_ok=True)
    if overwrite:
        open(DB_FILE, 'wb').close()
        open(IDX_FILE, 'wb').close()
        print("database reset")
    else:
        if not os.path.exists(DB_FILE):
            open(DB_FILE, 'wb').close()
        if not os.path.exists(IDX_FILE):
            open(IDX_FILE, 'wb').close()
        print("database initialized")

def store_chunks(chunks):
    if not chunks:
        print("no chunks to store")
        return

    print(f"embedding {len(chunks)} chunks...")
    embeddings = get_embeddings(chunks)
    packed = quantize_to_binary(embeddings)

    with open(DB_FILE, 'ab') as db, open(IDX_FILE, 'ab') as idx:
        for i, chunk in enumerate(chunks):
            offset = db.tell()
            idx.write(struct.pack('Q', offset))

            chunk_bytes = chunk.encode('utf-8')
            chunk_len = len(chunk_bytes)

            db.write(packed[i].tobytes())
            db.write(struct.pack('H', chunk_len))
            db.write(chunk_bytes)

            print(f"stored chunk {i+1}/{len(chunks)}: '{chunk[:40]}...'")

    print(f"\ndone. {len(chunks)} chunks stored")
    print(f"db size:  {os.path.getsize(DB_FILE)} bytes")
    print(f"idx size: {os.path.getsize(IDX_FILE)} bytes")

def read_chunk(offset):
    with open(DB_FILE, 'rb') as db:
        db.seek(offset)
        vector_bytes = db.read(48)
        packed = np.frombuffer(vector_bytes, dtype=np.uint8).copy()
        chunk_len = struct.unpack('H', db.read(2))[0]
        chunk = db.read(chunk_len).decode('utf-8')
        return packed, chunk

def load_all():
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
            vector_bytes = db.read(48)
            packed = np.frombuffer(vector_bytes, dtype=np.uint8).copy()
            chunk_len = struct.unpack('H', db.read(2))[0]
            chunk = db.read(chunk_len).decode('utf-8')
            vectors.append(packed)
            chunks.append(chunk)

    return np.array(vectors), chunks

def count_chunks():
    if not os.path.exists(IDX_FILE):
        return 0
    return os.path.getsize(IDX_FILE) // 8

def verify_db():
    print("\n--- verifying database ---")
    vectors, chunks = load_all()
    print(f"total chunks stored: {len(chunks)}")
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        print(f"\nchunk {i+1}:")
        print(f"  text:         '{chunk[:50]}'")
        print(f"  vector shape: {vec.shape}")

if __name__ == "__main__":
    init_db(overwrite=True)
    chunks = [
        "the robot motor requires high torque to operate",
        "motor needs strong rotational force for movement",
        "robot arm joint needs calibration every 6 months",
    ]
    store_chunks(chunks)
    verify_db()
    print(f"\ntotal chunks in db: {count_chunks()}")
