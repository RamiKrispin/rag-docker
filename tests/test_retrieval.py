from unittest.mock import MagicMock, patch

import pytest

from rag.config import (
    ActiveConfig,
    ChromaDBConfig,
    ModelConfig,
    ModelsConfig,
    ProviderConfig,
    RetrievalConfig,
    Settings,
)
from rag.retrieval.retriever import RetrievedChunk, retrieve
from rag.retrieval.reranker import rerank
from rag.retrieval.chain import (
    QueryResponse,
    Source,
    _build_context,
    _extract_sources,
    query_rag,
)


def _make_settings() -> Settings:
    return Settings(
        providers={
            "openai": ProviderConfig(
                api_key_env="OPENAI_API_KEY",
                models=ModelsConfig(
                    embedding=ModelConfig(
                        name="text-embedding-3-small",
                        dimensions=1536,
                    ),
                    chat=ModelConfig(
                        name="gpt-4o",
                        temperature=0.0,
                        max_tokens=2048,
                    ),
                ),
            ),
            "anthropic": ProviderConfig(
                api_key_env="ANTHROPIC_API_KEY",
                models=ModelsConfig(
                    chat=ModelConfig(
                        name="claude-sonnet-4-20250514",
                        temperature=0.0,
                        max_tokens=2048,
                    ),
                ),
            ),
        },
        active=ActiveConfig(
            embedding_provider="openai",
            chat_provider="openai",
        ),
        retrieval=RetrievalConfig(
            top_k=5,
            rerank=True,
            rerank_model="cross-encoder",
            score_threshold=0.3,
        ),
        chromadb=ChromaDBConfig(
            host="chromadb",
            port=8000,
            collection_name="financial_reports",
        ),
    )


def _make_chunks(n: int = 5) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content=f"Revenue in Q{i} was ${i * 10}B.",
            score=0.9 - i * 0.1,
            metadata={
                "source_file": "10Q-Q1-2026.pdf",
                "page_number": i,
                "section_title": "Revenue",
                "chunk_type": "text",
            },
        )
        for i in range(1, n + 1)
    ]


class TestRetriever:
    @patch("rag.retrieval.retriever.ChromaStore")
    @patch("rag.retrieval.retriever.get_embedder")
    def test_retrieve_returns_chunks(
        self, mock_embedder_fn, mock_store_cls
    ):
        config = _make_settings()

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Revenue was $50B", "Net income $5B"]],
            "metadatas": [[
                {"source_file": "test.pdf", "page_number": 1},
                {"source_file": "test.pdf", "page_number": 2},
            ]],
            "distances": [[0.1, 0.3]],
        }
        mock_store = MagicMock()
        mock_store.collection = mock_collection
        mock_store_cls.return_value = mock_store

        results = retrieve("What was the revenue?", config, top_k=2)
        assert len(results) == 2
        assert results[0].score > results[1].score
        assert "Revenue" in results[0].content

    @patch("rag.retrieval.retriever.ChromaStore")
    @patch("rag.retrieval.retriever.get_embedder")
    def test_retrieve_filters_by_threshold(
        self, mock_embedder_fn, mock_store_cls
    ):
        config = _make_settings()
        config = config.model_copy(
            update={"retrieval": config.retrieval.model_copy(
                update={"score_threshold": 0.8}
            )}
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["High relevance", "Low relevance"]],
            "metadatas": [[
                {"source_file": "a.pdf", "page_number": 1},
                {"source_file": "b.pdf", "page_number": 1},
            ]],
            "distances": [[0.1, 0.9]],
        }
        mock_store = MagicMock()
        mock_store.collection = mock_collection
        mock_store_cls.return_value = mock_store

        results = retrieve("test", config, top_k=5)
        assert len(results) == 1
        assert results[0].content == "High relevance"

    @patch("rag.retrieval.retriever.ChromaStore")
    @patch("rag.retrieval.retriever.get_embedder")
    def test_retrieve_empty_results(
        self, mock_embedder_fn, mock_store_cls
    ):
        config = _make_settings()

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_store = MagicMock()
        mock_store.collection = mock_collection
        mock_store_cls.return_value = mock_store

        results = retrieve("test", config)
        assert results == []


class TestReranker:
    def test_rerank_none_method(self):
        chunks = _make_chunks(5)
        result = rerank("question", chunks, method="none", top_k=3)
        assert len(result) == 3

    def test_rerank_empty_chunks(self):
        result = rerank("question", [], method="cross-encoder",
                        top_k=5)
        assert result == []

    def test_rerank_invalid_method(self):
        chunks = _make_chunks(3)
        with pytest.raises(ValueError, match="Unknown rerank method"):
            rerank("question", chunks, method="invalid")

    @patch("rag.retrieval.reranker._cross_encoder_model", None)
    def test_rerank_cross_encoder(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7]

        chunks = _make_chunks(3)
        original_scores = [c.score for c in chunks]

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": MagicMock(
                CrossEncoder=MagicMock(return_value=mock_model)
            )},
        ):
            result = rerank(
                "question", chunks,
                method="cross-encoder", top_k=2,
            )

        assert len(result) == 2
        assert result[0].score == pytest.approx(0.9)
        assert result[1].score == pytest.approx(0.7)
        # Original chunks not mutated
        for chunk, orig_score in zip(chunks, original_scores):
            assert chunk.score == orig_score


class TestChain:
    def test_build_context(self):
        chunks = _make_chunks(2)
        context = _build_context(chunks)
        assert "[Source 1:" in context
        assert "[Source 2:" in context
        assert "10Q-Q1-2026.pdf" in context

    def test_extract_sources(self):
        chunks = _make_chunks(3)
        sources = _extract_sources(chunks)
        assert len(sources) == 3
        assert all(isinstance(s, Source) for s in sources)
        assert sources[0].file == "10Q-Q1-2026.pdf"

    @patch("rag.retrieval.chain._get_chat_llm")
    @patch("rag.retrieval.chain.rerank")
    @patch("rag.retrieval.chain.retrieve")
    def test_query_rag_end_to_end(
        self, mock_retrieve, mock_rerank, mock_get_llm
    ):
        config = _make_settings()

        chunks = _make_chunks(3)
        mock_retrieve.return_value = chunks
        mock_rerank.return_value = chunks[:2]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Revenue was $50B in Q1 2026."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        response = query_rag("What was the revenue?", config)
        assert isinstance(response, QueryResponse)
        assert "50B" in response.answer
        assert len(response.sources) == 2
        assert response.metadata.provider == "openai"
        assert response.metadata.model == "gpt-4o"
        assert response.metadata.latency_ms >= 0

    @patch("rag.retrieval.chain.retrieve")
    def test_query_rag_no_results(self, mock_retrieve):
        config = _make_settings()
        mock_retrieve.return_value = []

        response = query_rag("Unknown topic?", config)
        assert "No relevant information" in response.answer
        assert response.sources == []
        assert response.metadata.retrieval_count == 0
