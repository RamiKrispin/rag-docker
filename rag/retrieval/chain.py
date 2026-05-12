import time
from dataclasses import dataclass, field

from rag.config import Settings
from rag.retrieval.retriever import RetrievedChunk, retrieve
from rag.retrieval.reranker import rerank

__all__ = ["QueryResponse", "Source", "QueryMetadata", "query_rag"]


@dataclass
class Source:
    file: str
    page: int
    section: str
    excerpt: str


@dataclass
class QueryMetadata:
    provider: str
    model: str
    retrieval_count: int
    latency_ms: int


@dataclass
class QueryResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    metadata: QueryMetadata | None = None


SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions \
based on the provided context from financial reports (10-Q filings).

Rules:
- Only answer based on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when stating facts (mention the document and page)
- For numerical data, be precise and include units
- If multiple documents contain relevant info, synthesize across them

Context:
{context}
"""


def query_rag(
    question: str,
    config: Settings,
    *,
    top_k: int | None = None,
    rerank_method: str | None = None,
    chat_provider: str | None = None,
) -> QueryResponse:
    start_time = time.time()

    if top_k is None:
        top_k = config.retrieval.top_k
    if rerank_method is None:
        rerank_method = (
            config.retrieval.rerank_model
            if config.retrieval.rerank
            else "none"
        )
    if chat_provider is not None:
        effective_provider = chat_provider
    else:
        effective_provider = config.active.chat_provider

    chunks = retrieve(question, config, top_k=top_k)

    if config.retrieval.rerank and rerank_method != "none":
        chunks = rerank(question, chunks, method=rerank_method,
                        top_k=top_k)

    if not chunks:
        return QueryResponse(
            answer="No relevant information found in the "
                   "ingested documents.",
            sources=[],
            metadata=QueryMetadata(
                provider=effective_provider,
                model="",
                retrieval_count=0,
                latency_ms=int((time.time() - start_time) * 1000),
            ),
        )

    context = _build_context(chunks)
    llm = _get_chat_llm(config, effective_provider)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            context=context
        )},
        {"role": "user", "content": question},
    ]

    response = llm.invoke(messages)
    answer = response.content

    sources = _extract_sources(chunks)

    latency_ms = int((time.time() - start_time) * 1000)
    provider = config.providers[effective_provider]

    return QueryResponse(
        answer=answer,
        sources=sources,
        metadata=QueryMetadata(
            provider=effective_provider,
            model=provider.models.chat.name,
            retrieval_count=len(chunks),
            latency_ms=latency_ms,
        ),
    )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page_number", "?")
        parts.append(
            f"[Source {i}: {source}, page {page}]\n"
            f"{chunk.content}\n"
        )
    return "\n---\n".join(parts)


def _extract_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    sources = []
    for chunk in chunks:
        sources.append(Source(
            file=chunk.metadata.get("source_file", "unknown"),
            page=chunk.metadata.get("page_number", 0),
            section=chunk.metadata.get("section_title", ""),
            excerpt=chunk.content[:200],
        ))
    return sources


def _get_chat_llm(config: Settings, provider_name: str):
    provider = config.providers[provider_name]
    model_config = provider.models.chat

    if provider_name == "openai":
        api_key = config.resolve_api_key("openai")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI "
                "chat. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model_config.name,
            temperature=(
                model_config.temperature
                if model_config.temperature is not None
                else 0.0
            ),
            max_tokens=model_config.max_tokens,
            openai_api_key=api_key,
        )
    elif provider_name == "anthropic":
        api_key = config.resolve_api_key("anthropic")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic "
                "chat. Install with: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=model_config.name,
            temperature=model_config.temperature or 0.0,
            max_tokens=model_config.max_tokens,
            anthropic_api_key=api_key,
        )
    elif provider_name == "gemini":
        api_key = config.resolve_api_key("gemini")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini "
                "chat. Install with: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=model_config.name,
            temperature=model_config.temperature or 0.0,
            max_tokens=model_config.max_tokens,
            google_api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unsupported chat provider: '{provider_name}'. "
            f"Supported: openai, anthropic, gemini"
        )
