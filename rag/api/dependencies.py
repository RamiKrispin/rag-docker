from functools import lru_cache

from rag.config import Settings, load_config
from rag.ingestion.embedder import get_embedder
from rag.store import ChromaStore


@lru_cache
def get_config() -> Settings:
    return load_config()


@lru_cache
def get_store() -> ChromaStore:
    return ChromaStore(get_config())


@lru_cache
def get_embedder_instance():
    return get_embedder(get_config())
