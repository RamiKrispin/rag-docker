from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_dir: str = "docs/"
    chunking_method: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    keep_tables_intact: bool = True


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    total_chunks: int
    files: list[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    chat_provider: str | None = None
    top_k: int | None = None
    rerank_method: str | None = None


class SourceResponse(BaseModel):
    file: str
    page: int
    section: str
    excerpt: str


class QueryMetadataResponse(BaseModel):
    provider: str
    model: str
    retrieval_count: int
    latency_ms: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    metadata: QueryMetadataResponse | None = None


class DocumentInfo(BaseModel):
    file: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    documents: int


class ConfigResponse(BaseModel):
    active_embedding_provider: str
    active_chat_provider: str
    chunking_method: str
    chunk_size: int
    retrieval_top_k: int
    rerank_enabled: bool
    rerank_model: str
    chromadb_host: str
    chromadb_port: int
    collection_name: str
    observability_enabled: bool


class ErrorResponse(BaseModel):
    detail: str
