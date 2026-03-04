from edgemind.core.config import EMBEDDING_MODEL

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"loading embedding model ({EMBEDDING_MODEL})...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("embedding model ready")
    return _embedding_model
