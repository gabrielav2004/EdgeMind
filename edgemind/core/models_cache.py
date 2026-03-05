import os
from edgemind.core.config import EMBEDDING_MODEL, EMBEDDING_CACHE, HF_TOKEN

_embedding_model = None

def _get_local_path():
    """get local cache path for the embedding model"""
    model_name = EMBEDDING_MODEL.replace('/', '_')
    return os.path.join(EMBEDDING_CACHE, model_name)

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"loading embedding model ({EMBEDDING_MODEL})...")

        # authenticate if token provided
        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)

        local_path = _get_local_path()

        if os.path.exists(local_path):
            # load from local cache — fully offline
            print("  loading from local cache (offline mode)...")
            os.environ["HF_HUB_OFFLINE"] = "1"
            _embedding_model = SentenceTransformer(local_path)
        else:
            # download and cache locally
            print("  downloading model...")
            os.makedirs(EMBEDDING_CACHE, exist_ok=True)
            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=EMBEDDING_CACHE
            )
            # save to local path for future offline use
            _embedding_model.save(local_path)
            print(f"  model cached at {local_path}")

        print("embedding model ready")
    return _embedding_model

def download_embedding_model():
    """
    Explicitly download and cache the embedding model.
    Run this once on a powerful machine before deploying to edge.
    """
    print(f"downloading {EMBEDDING_MODEL} for offline use...")
    os.makedirs(EMBEDDING_CACHE, exist_ok=True)

    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=EMBEDDING_CACHE)

    local_path = _get_local_path()
    model.save(local_path)

    print(f"✓ model saved to {local_path}")
    print(f"  copy this folder to your edge device before deploying")
    return local_path