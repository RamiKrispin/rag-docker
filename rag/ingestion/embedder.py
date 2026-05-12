import logging
import time
from typing import Any

from langchain_core.embeddings import Embeddings

from rag.config import Settings

__all__ = ["get_embedder", "embed_documents_batched"]

logger = logging.getLogger("rag")


def get_embedder(config: Settings) -> Embeddings:
    provider_name = config.active.embedding_provider
    provider = config.get_embedding_provider()
    model_config = provider.models.embedding

    if provider_name == "openai":
        api_key = config.resolve_api_key("openai")
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI "
                "embeddings. Install with: pip install langchain-openai"
            )
        return OpenAIEmbeddings(
            model=model_config.name,
            openai_api_key=api_key,
            dimensions=model_config.dimensions,
        )
    elif provider_name == "gemini":
        api_key = config.resolve_api_key("gemini")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini "
                "embeddings. Install with: "
                "pip install langchain-google-genai"
            )
        return GoogleGenerativeAIEmbeddings(
            model=model_config.name,
            google_api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider_name}'. "
            f"Supported: openai, gemini"
        )


def embed_documents_batched(
    embedder: Embeddings,
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    total = len(texts)
    all_vectors: list[list[float]] = []
    start = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vectors = embedder.embed_documents(batch)
        all_vectors.extend(vectors)

        done = min(i + batch_size, total)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        logger.info(
            f"Embedded {done}/{total} chunks "
            f"({done * 100 // total}%) "
            f"- {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining"
        )

    logger.info(f"Done: {total} chunks in {time.time() - start:.1f}s")
    return all_vectors
