import functools
from dataclasses import replace

from rag.retrieval.retriever import RetrievedChunk

__all__ = ["rerank"]


def rerank(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    method: str = "cross-encoder",
    top_k: int = 5,
) -> list[RetrievedChunk]:
    if method == "none" or not chunks:
        return chunks[:top_k]

    if method == "cross-encoder":
        return _rerank_cross_encoder(question, chunks, top_k)
    elif method == "llm":
        return _rerank_llm(question, chunks, top_k)
    else:
        raise ValueError(
            f"Unknown rerank method: '{method}'. "
            f"Supported: cross-encoder, llm, none"
        )


@functools.lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _rerank_cross_encoder(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    try:
        model = _get_cross_encoder()
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for cross-encoder "
            "reranking. Install with: pip install sentence-transformers"
        )

    pairs = [[question, chunk.content] for chunk in chunks]
    scores = model.predict(pairs)

    reranked = [
        replace(chunk, score=float(score))
        for chunk, score in zip(chunks, scores)
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked[:top_k]


def _rerank_llm(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    raise NotImplementedError(
        "LLM-based reranking is not yet implemented. "
        "Use method='cross-encoder' or method='none'."
    )
