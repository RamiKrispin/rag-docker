from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.config import (
    ActiveConfig,
    ChromaDBConfig,
    ModelsConfig,
    ModelConfig,
    ProviderConfig,
    Settings,
    load_config,
)
from rag.ingestion.pdf_parser import ParsedElement, parse_pdf
from rag.ingestion.chunker import Chunk, chunk_elements
from rag.ingestion.store import ChromaStore, _generate_id


CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
DOCS_DIR = Path(__file__).parent.parent / "docs"


def _make_test_settings() -> Settings:
    return Settings(
        providers={
            "openai": ProviderConfig(
                api_key_env="OPENAI_API_KEY",
                models=ModelsConfig(
                    embedding=ModelConfig(
                        name="text-embedding-3-small",
                        dimensions=1536,
                    ),
                    chat=ModelConfig(name="gpt-4o"),
                ),
            ),
        },
        active=ActiveConfig(
            embedding_provider="openai",
            chat_provider="openai",
        ),
        chromadb=ChromaDBConfig(
            host="chromadb",
            port=8000,
            collection_name="financial_reports",
        ),
    )


class TestPdfParser:
    def test_parse_pdf_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("nonexistent.pdf")

    @pytest.mark.skipif(
        not list(DOCS_DIR.glob("*.pdf")),
        reason="No PDF files in docs/ directory",
    )
    def test_parse_pdf_returns_elements(self):
        pdf_file = next(DOCS_DIR.glob("*.pdf"))
        result = parse_pdf(pdf_file)
        assert len(result) > 0
        assert all(isinstance(e, ParsedElement) for e in result)

    @pytest.mark.skipif(
        not list(DOCS_DIR.glob("*.pdf")),
        reason="No PDF files in docs/ directory",
    )
    def test_parse_pdf_element_structure(self):
        pdf_file = next(DOCS_DIR.glob("*.pdf"))
        result = parse_pdf(pdf_file)
        for elem in result[:10]:
            assert elem.type in ("text", "table", "heading")
            assert isinstance(elem.page, int)
            assert isinstance(elem.content, str)
            assert len(elem.content) > 0


class TestChunker:
    @pytest.fixture
    def sample_elements(self):
        return [
            ParsedElement(
                content="Introduction to the report",
                type="heading",
                page=1,
                section_title="Introduction",
            ),
            ParsedElement(
                content="This is a long paragraph. " * 100,
                type="text",
                page=1,
                section_title="Introduction",
            ),
            ParsedElement(
                content="| Col1 | Col2 |\n|---|---|\n| A | B |",
                type="table",
                page=2,
                section_title="Financials",
            ),
            ParsedElement(
                content="Short paragraph.",
                type="text",
                page=3,
                section_title="Conclusion",
            ),
        ]

    def test_recursive_chunking(self, sample_elements):
        chunks = chunk_elements(
            sample_elements, method="recursive",
            chunk_size=200, chunk_overlap=50
        )
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_by_title_chunking(self, sample_elements):
        chunks = chunk_elements(
            sample_elements, method="by_title",
            chunk_size=200, chunk_overlap=50
        )
        assert len(chunks) > 0

    def test_semantic_chunking(self, sample_elements):
        chunks = chunk_elements(
            sample_elements, method="semantic",
            chunk_size=200, chunk_overlap=50
        )
        assert len(chunks) > 0

    def test_tables_kept_intact(self, sample_elements):
        chunks = chunk_elements(
            sample_elements, method="recursive",
            chunk_size=10, chunk_overlap=2,
            keep_tables_intact=True
        )
        table_chunks = [c for c in chunks if c.type == "table"]
        assert len(table_chunks) == 1
        assert "Col1" in table_chunks[0].content

    def test_invalid_method(self, sample_elements):
        with pytest.raises(ValueError, match="chunking method"):
            chunk_elements(sample_elements, method="invalid")

    def test_chunk_metadata_preserved(self, sample_elements):
        chunks = chunk_elements(
            sample_elements, method="recursive",
            chunk_size=200, chunk_overlap=50
        )
        for chunk in chunks:
            assert isinstance(chunk.page, int)
            assert isinstance(chunk.section_title, str)
            assert chunk.type in ("text", "table", "heading")


class TestChromaStore:
    @pytest.fixture
    def mock_config(self):
        return _make_test_settings()

    @patch("rag.ingestion.store.chromadb.HttpClient")
    def test_store_init(self, mock_client_cls, mock_config):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = (
            MagicMock()
        )

        store = ChromaStore(mock_config)
        mock_client_cls.assert_called_once_with(
            host="chromadb", port=8000
        )
        mock_client.get_or_create_collection.assert_called_once_with(
            name="financial_reports"
        )

    @patch("rag.ingestion.store.chromadb.HttpClient")
    def test_store_upsert(self, mock_client_cls, mock_config):
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = (
            mock_collection
        )

        store = ChromaStore(mock_config)
        store.upsert(
            ids=["id1"],
            documents=["doc1"],
            metadatas=[{"source_file": "test.pdf"}],
        )
        mock_collection.upsert.assert_called_once()

    @patch("rag.ingestion.store.chromadb.HttpClient")
    def test_store_query(self, mock_client_cls, mock_config):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["result1"]],
            "metadatas": [[{"source_file": "test.pdf"}]],
        }
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = (
            mock_collection
        )

        store = ChromaStore(mock_config)
        results = store.query("test question", n_results=1)
        assert results["documents"][0][0] == "result1"

    @patch("rag.ingestion.store.chromadb.HttpClient")
    def test_store_ingest_chunks(self, mock_client_cls, mock_config):
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = (
            mock_collection
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [
            [0.1] * 1536, [0.2] * 1536
        ]

        store = ChromaStore(mock_config)
        chunks = [
            Chunk(content="text 1", type="text", page=1,
                  section_title="Intro", chunk_index=0),
            Chunk(content="text 2", type="text", page=2,
                  section_title="Body", chunk_index=1),
        ]
        count = store.ingest_chunks(
            chunks, mock_embedder, source_file="test.pdf"
        )
        assert count == 2
        mock_collection.upsert.assert_called_once()


class TestHelpers:
    def test_generate_id_deterministic(self):
        id1 = _generate_id("file.pdf", 0, "content")
        id2 = _generate_id("file.pdf", 0, "content")
        assert id1 == id2

    def test_generate_id_unique(self):
        id1 = _generate_id("file.pdf", 0, "content A")
        id2 = _generate_id("file.pdf", 1, "content B")
        assert id1 != id2
