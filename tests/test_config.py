import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import Settings, load_config


CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def test_load_config_from_yaml():
    config = load_config(CONFIG_PATH)
    assert isinstance(config, Settings)
    assert "openai" in config.providers
    assert "anthropic" in config.providers
    assert "gemini" in config.providers


def test_active_providers():
    config = load_config(CONFIG_PATH)
    assert config.active.embedding_provider == "openai"
    assert config.active.chat_provider == "openai"


def test_get_embedding_provider():
    config = load_config(CONFIG_PATH)
    provider = config.get_embedding_provider()
    assert provider.models.embedding is not None
    assert provider.models.embedding.name == "text-embedding-3-small"
    assert provider.models.embedding.dimensions == 1536


def test_get_chat_provider():
    config = load_config(CONFIG_PATH)
    provider = config.get_chat_provider()
    assert provider.models.chat is not None
    assert provider.models.chat.name == "gpt-4o"
    assert provider.models.chat.temperature == 0.0


def test_switch_chat_provider():
    config = load_config(CONFIG_PATH)
    config.active.chat_provider = "anthropic"
    provider = config.get_chat_provider()
    assert provider.models.chat.name == "claude-sonnet-4-20250514"


def test_chunking_defaults():
    config = load_config(CONFIG_PATH)
    assert config.chunking.method == "recursive"
    assert config.chunking.chunk_size == 1000
    assert config.chunking.chunk_overlap == 200
    assert config.chunking.keep_tables_intact is True


def test_chunking_invalid_method():
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValidationError):
        config.chunking.method = "invalid"


def test_chromadb_config():
    config = load_config(CONFIG_PATH)
    assert config.chromadb.host == "chromadb"
    assert config.chromadb.port == 8000
    assert config.chromadb.collection_name == "financial_reports"


def test_retrieval_config():
    config = load_config(CONFIG_PATH)
    assert config.retrieval.top_k == 5
    assert config.retrieval.rerank is True
    assert config.retrieval.score_threshold == 0.3


def test_resolve_api_key_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
    config = load_config(CONFIG_PATH)
    key = config.resolve_api_key("openai")
    assert key == "sk-test-key-123"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="not set"):
        config.resolve_api_key("openai")


def test_resolve_api_key_placeholder(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key_is_missing")
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="not set"):
        config.resolve_api_key("openai")


def test_invalid_provider_name():
    config = load_config(CONFIG_PATH)
    config.active.chat_provider = "nonexistent"
    with pytest.raises(ValidationError):
        Settings(
            providers=config.providers,
            active=config.active,
        )


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/path.yaml")


def test_embedding_provider_without_embedding_model():
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValidationError):
        Settings(
            providers=config.providers,
            active={"embedding_provider": "anthropic",
                    "chat_provider": "openai"},
        )
