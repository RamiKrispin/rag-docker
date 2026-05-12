import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_DEFAULT_CONFIG = Path(
    os.environ.get(
        "RAG_CONFIG_PATH",
        Path(__file__).parent.parent / "config" / "settings.yaml",
    )
)


class ModelConfig(BaseModel):
    name: str
    dimensions: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ModelsConfig(BaseModel):
    embedding: Optional[ModelConfig] = None
    chat: Optional[ModelConfig] = None


class ProviderConfig(BaseModel):
    api_key_env: str
    models: ModelsConfig


class ActiveConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    embedding_provider: str
    chat_provider: str


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    method: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    keep_tables_intact: bool = True

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"recursive", "semantic", "by_title"}
        if v not in allowed:
            raise ValueError(
                f"chunking method must be one of {allowed}, got '{v}'"
            )
        return v


class RetrievalConfig(BaseModel):
    top_k: int = 5
    rerank: bool = True
    rerank_model: str = "cross-encoder"
    score_threshold: float = 0.3


class ChromaDBConfig(BaseModel):
    host: str = "chromadb"
    port: int = 8000
    collection_name: str = "financial_reports"


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    provider: str = "langsmith"
    project_name: str = "rag-docker"


class Settings(BaseModel):
    providers: dict[str, ProviderConfig]
    active: ActiveConfig
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chromadb: ChromaDBConfig = ChromaDBConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @model_validator(mode="after")
    def validate_active_providers(self) -> "Settings":
        self.get_embedding_provider()
        self.get_chat_provider()
        return self

    def get_embedding_provider(self) -> ProviderConfig:
        name = self.active.embedding_provider
        if name not in self.providers:
            raise ValueError(
                f"Embedding provider '{name}' not found in providers"
            )
        provider = self.providers[name]
        if provider.models.embedding is None:
            raise ValueError(
                f"Provider '{name}' has no embedding model configured"
            )
        return provider

    def get_chat_provider(self) -> ProviderConfig:
        name = self.active.chat_provider
        if name not in self.providers:
            raise ValueError(
                f"Chat provider '{name}' not found in providers"
            )
        provider = self.providers[name]
        if provider.models.chat is None:
            raise ValueError(
                f"Provider '{name}' has no chat model configured"
            )
        return provider

    def resolve_api_key(self, provider_name: str) -> str:
        if provider_name not in self.providers:
            raise ValueError(
                f"Provider '{provider_name}' not found in providers"
            )
        env_var = self.providers[provider_name].api_key_env
        value = os.environ.get(env_var, "")
        if not value or value == "key_is_missing":
            raise ValueError(
                f"Environment variable '{env_var}' is not set "
                f"for provider '{provider_name}'"
            )
        return value


def load_config(path: str | Path = _DEFAULT_CONFIG) -> Settings:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Settings(**data)
