from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import chromadb
from langchain_core.embeddings import Embeddings

from rag.config import Settings

if TYPE_CHECKING:
    from rag.ingestion.chunker import Chunk

__all__ = ["ChromaStore"]


class ChromaStore:
    def __init__(self, config: Settings):
        self._config = config
        self._client = chromadb.HttpClient(
            host=config.chromadb.host,
            port=config.chromadb.port,
        )
        self._collection = self._client.get_or_create_collection(
            name=config.chromadb.collection_name,
        )

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._collection.upsert(**kwargs)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def delete_by_source(self, source_file: str) -> None:
        results = self._collection.get(
            where={"source_file": source_file}
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    def list_documents(self) -> list[dict]:
        total = self._collection.count()
        all_metadatas: list[dict] = []
        batch_size = 100
        for offset in range(0, total, batch_size):
            batch = self._collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            all_metadatas.extend(batch["metadatas"])

        sources: dict[str, dict] = {}
        for meta in all_metadatas:
            source = meta.get("source_file", "unknown")
            if source not in sources:
                sources[source] = {"file": source, "chunks": 0}
            sources[source]["chunks"] += 1
        return list(sources.values())

    def ingest_chunks(
        self,
        chunks: list[Chunk],
        embedder: Embeddings,
        source_file: str = "",
    ) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(texts)
        timestamp = datetime.now(timezone.utc).isoformat()

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = _generate_id(source_file, i, chunk.content)
            ids.append(chunk_id)
            metadatas.append({
                "source_file": source_file or chunk.metadata.get(
                    "source_file", ""
                ),
                "page_number": chunk.page,
                "chunk_type": chunk.type,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "ingestion_timestamp": timestamp,
            })

        self.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)


def _generate_id(source: str, index: int, content: str) -> str:
    hash_input = f"{source}:{index}:{content[:100]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:32]
