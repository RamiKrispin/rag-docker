from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from rag.api.dependencies import (
    get_config,
    get_embedder_instance,
    get_store,
)
from rag.api.models import (
    ConfigResponse,
    DocumentInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QueryMetadataResponse,
    SourceResponse,
)
from rag.ingestion.chunker import chunk_elements
from rag.ingestion.pdf_parser import parse_pdf
from rag.observability.logging import setup_logging
from rag.observability.tracing import configure_tracing
from rag.retrieval.chain import query_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    config = get_config()
    configure_tracing(config)
    yield


app = FastAPI(
    title="RAG Docker API",
    description="RAG system for financial PDF reports",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        store = get_store()
        doc_count = store.count()
        chromadb_status = "connected"
    except Exception:
        chromadb_status = "disconnected"
        doc_count = 0

    status = "healthy" if chromadb_status == "connected" else "degraded"
    return HealthResponse(
        status=status,
        chromadb=chromadb_status,
        documents=doc_count,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest):
    config = get_config()
    source_dir = Path(request.source_dir).resolve()

    allowed_base = Path.cwd().resolve()
    try:
        source_dir.relative_to(allowed_base)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Source directory must be within the project",
        )

    if not source_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source directory not found: {source_dir}",
        )

    pdf_files = list(source_dir.glob("*.pdf"))
    if not pdf_files:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF files found in: {source_dir}",
        )

    store = get_store()
    embedder = get_embedder_instance()
    total_chunks = 0
    ingested_files = []

    for pdf_path in pdf_files:
        elements = parse_pdf(pdf_path)
        chunks = chunk_elements(
            elements,
            method=request.chunking_method,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            keep_tables_intact=request.keep_tables_intact,
        )
        count = store.ingest_chunks(
            chunks, embedder, source_file=pdf_path.name
        )
        total_chunks += count
        ingested_files.append(pdf_path.name)

    return IngestResponse(
        status="success",
        documents_ingested=len(ingested_files),
        total_chunks=total_chunks,
        files=ingested_files,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    config = get_config()

    try:
        response = query_rag(
            question=request.question,
            config=config,
            top_k=request.top_k,
            rerank_method=request.rerank_method,
            chat_provider=request.chat_provider,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}",
        )

    sources = [
        SourceResponse(
            file=s.file,
            page=s.page,
            section=s.section,
            excerpt=s.excerpt,
        )
        for s in response.sources
    ]

    metadata = None
    if response.metadata:
        metadata = QueryMetadataResponse(
            provider=response.metadata.provider,
            model=response.metadata.model,
            retrieval_count=response.metadata.retrieval_count,
            latency_ms=response.metadata.latency_ms,
        )

    return QueryResponse(
        answer=response.answer,
        sources=sources,
        metadata=metadata,
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    store = get_store()
    docs = store.list_documents()
    return [DocumentInfo(**doc) for doc in docs]


@app.delete("/documents/{source_file}")
def delete_document(source_file: str):
    store = get_store()
    results = store.collection.get(
        where={"source_file": source_file}
    )
    if not results["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {source_file}",
        )
    store.delete_by_source(source_file)
    return {"status": "deleted", "file": source_file}


@app.get("/config", response_model=ConfigResponse)
def get_configuration():
    config = get_config()
    return ConfigResponse(
        active_embedding_provider=config.active.embedding_provider,
        active_chat_provider=config.active.chat_provider,
        chunking_method=config.chunking.method,
        chunk_size=config.chunking.chunk_size,
        retrieval_top_k=config.retrieval.top_k,
        rerank_enabled=config.retrieval.rerank,
        rerank_model=config.retrieval.rerank_model,
        chromadb_host=config.chromadb.host,
        chromadb_port=config.chromadb.port,
        collection_name=config.chromadb.collection_name,
        observability_enabled=config.observability.enabled,
    )
