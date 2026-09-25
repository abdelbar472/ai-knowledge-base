import os
import threading

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is cached locally; no network check at startup

# Embeddings (lazy: model only loads when a chunk really needs embedding)
EMBEDDINGS_LOCK = threading.Lock()
_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        with EMBEDDINGS_LOCK:
            if _embeddings_instance is None:
                _embeddings_instance = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"batch_size": 32},
                )
                print(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return _embeddings_instance