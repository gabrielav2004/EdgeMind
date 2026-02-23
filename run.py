import sys
import os
from parse import parse_file, parse_folder
from store import init_db, store_chunks, count_chunks
from search import search
from respond import respond

def ingest(path):
    print("=== INGESTION ===")
    init_db(overwrite=True)
    
    if os.path.isdir(path):
        chunks = parse_folder(path)
    else:
        chunks = parse_file(path)
    
    if chunks:
        store_chunks(chunks)
        print(f"\n✓ ingestion complete. {count_chunks()} chunks in database")
    else:
        print("no chunks found")

def query(text):
    print("=== QUERY ===")
    results = search(text, top_k=3)
    chunks = [r['chunk'] for r in results]
    
    print("\n=== RESPONSE ===")
    answer = respond(text, chunks)
    print(f"\n{answer}")
    return answer

def interactive():
    print("=== EdgeMind ===")
    print("local semantic knowledge system")
    print("type 'quit' to exit\n")
    
    # preload model so first query is fast
    from respond import load_model
    load_model()
    
    while True:
        q = input("\nquery: ").strip()
        if q.lower() == 'quit':
            break
        if q:
            query(q)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage:")
        print("  python run.py ingest <file_or_folder>")
        print("  python run.py query <text>")
        print("  python run.py interactive")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "ingest":
        path = sys.argv[2] if len(sys.argv) > 2 else "data/docs"
        ingest(path)
    
    elif command == "query":
        text = " ".join(sys.argv[2:])
        query(text)
    
    elif command == "interactive":
        interactive()
    
    else:
        print(f"unknown command: {command}")