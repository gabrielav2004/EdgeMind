from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from edgemind.core.config import HOST, PORT, DOCS_FOLDER
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading models...")
    from edgemind.core.models_cache import get_embedding_model
    from edgemind.generation.respond import load_model
    get_embedding_model()
    load_model()
    print("edgemind ready")
    yield

app = FastAPI(
    title="EdgeMind",
    description="Lightweight local semantic retrieval and response API",
    version="0.1.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3
    respond: bool = True

class ChunkResult(BaseModel):
    chunk: str
    score: float
    distance: int

class QueryResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    answer: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    from edgemind.retrieval.search import search
    from edgemind.generation.respond import respond

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="query text cannot be empty")

    if not os.path.exists("data/knowledge.bin"):
        raise HTTPException(status_code=503, detail="knowledge base not found, run ingest first")

    results = search(request.text, top_k=request.top_k)
    chunks = [r['chunk'] for r in results]

    formatted = [
        ChunkResult(
            chunk=r['chunk'],
            score=r.get('float_score', r['score']),
            distance=r['distance']
        )
        for r in results
    ]

    answer = None
    if request.respond:
        answer = respond(request.text, chunks)

    return QueryResponse(
        query=request.text,
        results=formatted,
        answer=answer
    )

@app.post("/ingest")
def ingest(folder: str = DOCS_FOLDER):
    from edgemind.ingestion.parse import parse_folder
    from edgemind.ingestion.store import init_db, store_chunks, count_chunks

    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail=f"folder not found: {folder}")

    init_db(overwrite=True)
    chunks = parse_folder(folder)

    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks extracted")

    store_chunks(chunks)
    return {"status": "ok", "chunks_stored": count_chunks()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)