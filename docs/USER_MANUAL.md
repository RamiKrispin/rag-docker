# RAG Docker — User Manual

A RAG (Retrieval-Augmented Generation) system for financial PDF reports. Ingest 10-Q filings, query them with natural language, and get answers with source citations.

---

## Table of Contents

1. [Setup](#1-setup)
2. [PDF Ingestion](#2-pdf-ingestion)
3. [Querying Documents](#3-querying-documents)
4. [Client Interfaces](#4-client-interfaces)
5. [Configuration](#5-configuration)
6. [Evaluation](#6-evaluation)
7. [Observability](#7-observability)

---

## 1. Setup

### Prerequisites

- Docker and Docker Compose
- API keys for at least one LLM provider (OpenAI, Anthropic, or Gemini)

### Environment Variables

Set these in your shell (e.g., `~/.zshenv`):

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export CHROMA_DATA_PATH="./chroma_data"
```

### Start the System

```bash
# Start all services (dev container + ChromaDB)
docker compose up -d

# Verify ChromaDB is healthy
curl http://localhost:8000/api/v1/heartbeat

# Start the FastAPI server (inside the container)
uvicorn rag.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Optional: Start with Observability (Phoenix)

```bash
docker compose --profile observability up -d
# Phoenix UI available at http://localhost:6006
```

---

## 2. PDF Ingestion

### What Happens During Ingestion

```
PDF files in docs/
  → Docling parser (extracts text, tables, headings with page numbers)
  → Chunker (splits into manageable pieces, keeps tables intact)
  → Embedder (generates vector embeddings via OpenAI/Gemini)
  → ChromaDB (stores vectors + metadata for retrieval)
```

### Place Your PDFs

Put PDF files in the `docs/` folder:

```bash
ls docs/
# 10Q-Q1-2026-as-filed.pdf
# 10Q-Q2-2026-as-filed.pdf
# GOOG-10-Q-Q1-2026.pdf
# form-10-q.pdf
```

### Ingest via API

```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_dir": "docs/",
    "chunking_method": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "keep_tables_intact": true
  }'
```

**Response:**
```json
{
  "status": "success",
  "documents_ingested": 4,
  "total_chunks": 342,
  "files": ["10Q-Q1-2026-as-filed.pdf", "10Q-Q2-2026-as-filed.pdf", ...]
}
```

### Ingest via CLI

```bash
python clients/cli.py ingest --source-dir docs/ --method recursive
python clients/cli.py ingest --source-dir docs/ --method by_title
python clients/cli.py ingest --source-dir docs/ --method semantic
```

### Ingest via Jupyter

```python
from clients.notebook_client import RAGClient

client = RAGClient()
result = client.ingest(source_dir="docs/", chunking_method="recursive")
print(result)
```

### Chunking Methods

| Method | Best for | How it works |
|--------|----------|-------------|
| `recursive` | General use (default) | Splits by character count with overlap at paragraph/sentence boundaries |
| `by_title` | Well-structured docs | Groups content under each heading into chunks |
| `semantic` | Dense text | Groups adjacent paragraphs until chunk size is reached |

All methods keep tables intact as single chunks when `keep_tables_intact=true`.

---

## 3. Querying Documents

### How Query Works

```
Your question
  → Embed question (same embedding model used for ingestion)
  → ChromaDB similarity search (find top_k * 2 candidates)
  → Re-ranker (cross-encoder scores relevance, picks top_k)
  → LLM generation (builds prompt with context, generates answer)
  → Response with source citations
```

### Query via API

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Apple total revenue in Q1 2026?",
    "top_k": 5,
    "chat_provider": "openai",
    "rerank_method": "cross-encoder"
  }'
```

**Response:**
```json
{
  "answer": "Apple's total revenue in Q1 2026 was $95.4 billion...",
  "sources": [
    {
      "file": "10Q-Q1-2026-as-filed.pdf",
      "page": 5,
      "section": "Revenue Summary",
      "excerpt": "Net revenue for the three months ended..."
    }
  ],
  "metadata": {
    "provider": "openai",
    "model": "gpt-4o",
    "retrieval_count": 5,
    "latency_ms": 1230
  }
}
```

### Query via CLI

```bash
# Basic query
python clients/cli.py query "What was the total revenue?"

# With options
python clients/cli.py query "What were the risk factors?" --top-k 10 --provider anthropic

# Use a different reranking method
python clients/cli.py query "Compare Q1 and Q2 revenue" --rerank cross-encoder
```

### Query via Jupyter

```python
from clients.notebook_client import RAGClient

client = RAGClient()

# Ask a question
response = client.query("What was the net income?", top_k=5)
print(response.answer)

# View sources
response.show_sources()

# Rich display (renders markdown in notebook)
response
```

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `question` | (required) | Your natural language question |
| `top_k` | 5 | Number of relevant chunks to retrieve |
| `chat_provider` | from config | Override: `openai`, `anthropic`, or `gemini` |
| `rerank_method` | `cross-encoder` | `cross-encoder` (accurate) or `none` (faster) |

---

## 4. Client Interfaces

### API (FastAPI)

The central service — all other clients connect through it.

```bash
# Start the server
uvicorn rag.api.main:app --host 0.0.0.0 --port 8080 --reload

# Interactive API docs
# Open: http://localhost:8080/docs
```

**Available endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Check API + ChromaDB status |
| `POST` | `/ingest` | Ingest PDFs from a directory |
| `POST` | `/query` | Ask a question |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{file}` | Remove a document |
| `GET` | `/config` | View active configuration |

### CLI

```bash
# Check system health
python clients/cli.py health

# Ingest documents
python clients/cli.py ingest --source-dir docs/

# Ask a question
python clients/cli.py query "What was the revenue?"

# List ingested documents
python clients/cli.py docs

# View configuration
python clients/cli.py config

# Use a different API URL
python clients/cli.py --api-url http://api:8080 health
```

### Streamlit (Web UI)

```bash
streamlit run clients/streamlit_app.py
# Open: http://localhost:8501
```

Features:
- Chat-style Q&A interface
- Sidebar: provider selection, top_k slider, reranking toggle
- Document management: ingest, view ingested docs
- Expandable source citations per answer
- Health status indicator

### Jupyter Notebook

```python
from clients.notebook_client import RAGClient

# Initialize (uses http://localhost:8080 by default)
client = RAGClient()

# Or with custom URL
client = RAGClient(api_url="http://api:8080")

# Check health
client.health()

# Ingest
client.ingest(source_dir="docs/")

# Query (returns rich-display QueryResult)
response = client.query("What was the EPS?")
response  # renders markdown in notebook

# List documents
client.documents()

# Delete a document
client.delete("old-file.pdf")

# Clean up connection
client.close()

# Or use as context manager
with RAGClient() as client:
    response = client.query("What were the operating expenses?")
```

---

## 5. Configuration

### Config File

All settings are in `config/settings.yaml`:

```yaml
# Switch between providers
active:
  embedding_provider: "openai"    # or "gemini"
  chat_provider: "openai"         # or "anthropic" or "gemini"

# Chunking defaults
chunking:
  method: "recursive"             # recursive, semantic, by_title
  chunk_size: 1000
  chunk_overlap: 200
  keep_tables_intact: true

# Retrieval settings
retrieval:
  top_k: 5
  rerank: true
  rerank_model: "cross-encoder"
  score_threshold: 0.3

# Observability (opt-in)
observability:
  enabled: false
  provider: "langsmith"           # or "phoenix"
  project_name: "rag-docker"
```

### Switching Providers

**At config level** (affects all requests):
Edit `config/settings.yaml` → `active.chat_provider`

**Per request** (API override):
```bash
curl -X POST http://localhost:8080/query \
  -d '{"question": "...", "chat_provider": "anthropic"}'
```

**Via CLI**:
```bash
python clients/cli.py query "..." --provider anthropic
```

### Available Models

| Provider | Chat Model | Embedding Model |
|----------|-----------|-----------------|
| OpenAI | gpt-4o | text-embedding-3-small |
| Anthropic | claude-sonnet-4-20250514 | — (no embedding) |
| Gemini | gemini-2.5-flash | text-embedding-004 |

Note: Anthropic has no embedding model. Use OpenAI or Gemini for embeddings, and Anthropic for chat only.

---

## 6. Evaluation

Benchmark the RAG system's quality on a set of test questions.

### Run Evaluation

```bash
# Default settings
python dev/02_evaluation.py

# With options
python dev/02_evaluation.py --top-k 10 --provider openai
python dev/02_evaluation.py --method by_title
python dev/02_evaluation.py --rerank none
```

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **Retrieval Precision@K** | % of retrieved chunks that contain expected keywords |
| **Retrieval Recall@K** | % of expected keywords found in retrieved chunks |
| **Answer Relevancy** | % of expected keywords present in the answer |
| **Answer Faithfulness** | % of answer sentences grounded in source text |
| **Latency** | End-to-end response time in milliseconds |

### Comparing Configurations

```bash
# Compare chunking methods
python dev/02_evaluation.py --method recursive
python dev/02_evaluation.py --method by_title

# Compare providers
python dev/02_evaluation.py --provider openai
python dev/02_evaluation.py --provider anthropic

# Compare top_k values
python dev/02_evaluation.py --top-k 3
python dev/02_evaluation.py --top-k 10
```

### Test Set

The test questions are in `rag/evaluation/test_set.yaml`. Add your own:

```yaml
questions:
  - id: q11
    question: "What was the gross margin?"
    expected_keywords:
      - "gross"
      - "margin"
    expected_source_type: "table"
    category: "factual_numeric"
```

---

## 7. Observability

### Structured Logging

All API requests produce JSON-formatted logs with trace IDs:

```json
{"timestamp": "2026-05-08 10:30:00", "level": "INFO", "module": "chain", "message": "retrieve completed", "trace_id": "a1b2c3d4e5f6", "stage": "retrieve", "latency_ms": 120}
```

### Enable LangSmith Tracing

1. Set env var: `export LANGSMITH_API_KEY="ls-..."`
2. Edit `config/settings.yaml`:
   ```yaml
   observability:
     enabled: true
     provider: "langsmith"
     project_name: "rag-docker"
   ```
3. Restart the API server
4. View traces at [smith.langchain.com](https://smith.langchain.com)

### Enable Phoenix (Self-Hosted)

1. Start with the observability profile:
   ```bash
   docker compose --profile observability up -d
   ```
2. Edit `config/settings.yaml`:
   ```yaml
   observability:
     enabled: true
     provider: "phoenix"
   ```
3. Restart the API server
4. View traces at http://localhost:6006

---

## Quick Reference

### Common Workflows

**First-time setup:**
```bash
docker compose up -d
uvicorn rag.api.main:app --host 0.0.0.0 --port 8080 --reload
python clients/cli.py ingest --source-dir docs/
python clients/cli.py query "What was the revenue?"
```

**Add new documents:**
```bash
# Copy PDFs to docs/
cp ~/new-report.pdf docs/
# Re-ingest
python clients/cli.py ingest --source-dir docs/
```

**Switch to a different LLM:**
```bash
python clients/cli.py query "..." --provider anthropic
```

**Check what's ingested:**
```bash
python clients/cli.py docs
```

**Remove a document:**
```bash
curl -X DELETE http://localhost:8080/documents/old-file.pdf
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| ChromaDB disconnected | `docker compose up chromadb -d` |
| Docling model download fails | Check internet connection; models cache at `~/.cache/huggingface` |
| `libxcb.so.1` error | Install: `apt-get install -y libxcb1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1` |
| API key missing | Set `OPENAI_API_KEY` in your shell env, restart container |
| Empty results from query | Verify documents are ingested: `python clients/cli.py docs` |
| Slow first query | Docling/CrossEncoder models load on first use; subsequent queries are faster |
