# Test Index

This document tracks all tests created during development, organized by system component.

---

## Summary

| Component | Test File | Tests | Phase |
|-----------|-----------|-------|-------|
| Configuration System | `tests/test_config.py` | 15 | Phase 2 |
| Ingestion Pipeline | `tests/test_ingestion.py` | 15 | Phase 3 |
| Retrieval & Generation | `tests/test_retrieval.py` | 11 | Phase 4 |

**Total tests**: 41

---

## Configuration System

**File**: `tests/test_config.py`

| Test | Description | Validates |
|------|-------------|-----------|
| `test_load_config_from_yaml` | Loads config from YAML, verifies all providers present | YAML parsing, Pydantic model construction |
| `test_active_providers` | Checks active provider defaults match YAML | Active provider config |
| `test_get_embedding_provider` | Gets OpenAI embedding config, checks name + dimensions | Embedding provider retrieval |
| `test_get_chat_provider` | Gets OpenAI chat config, checks name + temperature | Chat provider retrieval |
| `test_switch_chat_provider` | Switches to Anthropic, verifies model name | Runtime provider switching |
| `test_chunking_defaults` | Verifies chunking defaults from YAML | Chunking config loading |
| `test_chunking_invalid_method` | Assigns invalid method, expects ValidationError | `validate_assignment=True` enforcement |
| `test_chromadb_config` | Checks host, port, collection name | ChromaDB config loading |
| `test_retrieval_config` | Checks top_k, rerank, score_threshold | Retrieval config loading |
| `test_resolve_api_key_success` | Sets env var, resolves key successfully | Env var resolution (happy path) |
| `test_resolve_api_key_missing` | Unsets env var, expects ValueError | Missing key detection |
| `test_resolve_api_key_placeholder` | Sets "key_is_missing" placeholder, expects ValueError | Placeholder detection |
| `test_invalid_provider_name` | Constructs Settings with nonexistent provider, expects ValidationError | Cross-field model_validator |
| `test_config_file_not_found` | Passes nonexistent path, expects FileNotFoundError | File-not-found handling |
| `test_embedding_provider_without_embedding_model` | Sets Anthropic as embedding provider (no embedding model), expects ValidationError | Missing model validation |

**Run command**:
```bash
pytest tests/test_config.py -v
```

---

## Ingestion Pipeline

**File**: `tests/test_ingestion.py`

| Test | Description | Validates |
|------|-------------|-----------|
| `TestPdfParser::test_parse_pdf_file_not_found` | Passes nonexistent path, expects FileNotFoundError | Error handling |
| `TestPdfParser::test_parse_pdf_returns_elements` | Parses a real PDF, checks elements returned | Docling integration (requires PDF + network) |
| `TestPdfParser::test_parse_pdf_element_structure` | Checks type, page, content fields on parsed elements | Element structure |
| `TestChunker::test_recursive_chunking` | Chunks sample elements with recursive method | Recursive chunker |
| `TestChunker::test_by_title_chunking` | Chunks with by_title method | By-title chunker |
| `TestChunker::test_semantic_chunking` | Chunks with semantic method | Semantic chunker |
| `TestChunker::test_tables_kept_intact` | Verifies tables are not split when keep_tables_intact=True | Table preservation |
| `TestChunker::test_invalid_method` | Passes invalid method, expects ValueError | Input validation |
| `TestChunker::test_chunk_metadata_preserved` | Checks page, section_title, type flow through chunking | Metadata propagation |
| `TestChromaStore::test_store_init` | Mocks ChromaDB client, verifies connection params | Store initialization |
| `TestChromaStore::test_store_upsert` | Mocks collection, verifies upsert called | Upsert operation |
| `TestChromaStore::test_store_query` | Mocks query results, verifies return format | Query operation |
| `TestChromaStore::test_store_ingest_chunks` | Mocks embedder + collection, ingests 2 chunks | End-to-end ingest |
| `TestHelpers::test_generate_id_deterministic` | Same input produces same ID | ID consistency |
| `TestHelpers::test_generate_id_unique` | Different input produces different IDs | ID uniqueness |

**Run command**:
```bash
pytest tests/test_ingestion.py -v
```

---

## Retrieval & Generation

**File**: `tests/test_retrieval.py`

| Test | Description | Validates |
|------|-------------|-----------|
| `TestRetriever::test_retrieve_returns_chunks` | Mocks embedder + ChromaDB, verifies chunks returned sorted by score | Basic retrieval |
| `TestRetriever::test_retrieve_filters_by_threshold` | Sets high threshold, verifies low-score chunks filtered out | Score threshold filtering |
| `TestRetriever::test_retrieve_empty_results` | Mocks empty ChromaDB response, verifies empty list returned | Empty result handling |
| `TestReranker::test_rerank_none_method` | Method="none", returns top_k without reranking | Passthrough mode |
| `TestReranker::test_rerank_empty_chunks` | Empty input, returns empty list | Edge case |
| `TestReranker::test_rerank_invalid_method` | Invalid method, expects ValueError | Input validation |
| `TestReranker::test_rerank_cross_encoder` | Mocks CrossEncoder model, verifies scoring + sorting + no mutation | Cross-encoder reranking |
| `TestChain::test_build_context` | Builds context string from chunks, verifies source references | Context formatting |
| `TestChain::test_extract_sources` | Extracts Source objects from chunks | Source extraction |
| `TestChain::test_query_rag_end_to_end` | Mocks retrieve + rerank + LLM, verifies full response structure | End-to-end query |
| `TestChain::test_query_rag_no_results` | No chunks retrieved, verifies "no information" response | Empty result path |

**Run command**:
```bash
pytest tests/test_retrieval.py -v
```

---

## FastAPI Service

**File**: `tests/test_api.py` (Phase 5 — pending)

_To be added when Phase 5 is implemented._

---

## Run All Tests

```bash
pytest tests/ -v
```
