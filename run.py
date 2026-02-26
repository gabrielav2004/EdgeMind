import sys
import os
from config import DOCS_FOLDER

def ingest(path):
    from parse import parse_file, parse_folder
    from store import init_db, store_chunks, count_chunks
    from config import USE_LLM_FORMATTER

    print("=== INGESTION ===")
    init_db(overwrite=True)

    if os.path.isdir(path):
        chunks = parse_folder(path, use_llm=USE_LLM_FORMATTER)
    else:
        chunks = parse_file(path, use_llm=USE_LLM_FORMATTER)

    if chunks:
        store_chunks(chunks)
        print(f"\n✓ ingestion complete. {count_chunks()} chunks in database")
    else:
        print("no chunks found")

def query(text):
    from search import search
    from respond import respond

    print("=== QUERY ===")
    results = search(text)
    chunks = [r['chunk'] for r in results]

    print("\n=== RESPONSE ===")
    answer = respond(text, chunks, results=results)
    print(f"\n{answer}")
    return answer

def interactive():
    from search import search
    from respond import load_model, respond
    from models_cache import get_embedding_model

    print("=== EdgeMind ===")
    print("local semantic knowledge system")
    print("type 'quit' to exit\n")

    # preload everything once
    get_embedding_model()
    load_model()

    while True:
        q = input("\nquery: ").strip()
        if q.lower() == 'quit':
            break
        if q:
            results = search(q)
            chunks = [r['chunk'] for r in results]
            answer = respond(q, chunks, results=results)
            print(f"\n{answer}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage:")
        print("  python run.py ingest <file_or_folder>")
        print("  python run.py query <text>")
        print("  python run.py interactive")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        path = sys.argv[2] if len(sys.argv) > 2 else DOCS_FOLDER
        ingest(path)
    elif command == "query":
        print("tip: use interactive mode for faster repeated queries")
        text = " ".join(sys.argv[2:])
        query(text)
    elif command == "interactive":
        interactive()
    else:
        print(f"unknown command: {command}")