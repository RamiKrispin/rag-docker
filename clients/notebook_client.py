import os
from urllib.parse import quote

import httpx


class RAGClient:
    """Helper class for interacting with the RAG API from Jupyter."""

    def __init__(self, api_url: str | None = None):
        self.api_url = api_url or os.environ.get(
            "RAG_API_URL", "http://localhost:8080"
        )
        self._client = httpx.Client(
            base_url=self.api_url, timeout=300.0
        )

    def close(self):
        """Close the HTTP connection pool."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def health(self) -> dict:
        """Check API and ChromaDB health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def ingest(
        self,
        source_dir: str = "docs/",
        chunking_method: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> dict:
        """Ingest PDF documents."""
        resp = self._client.post("/ingest", json={
            "source_dir": source_dir,
            "chunking_method": chunking_method,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })
        resp.raise_for_status()
        return resp.json()

    def query(
        self,
        question: str,
        top_k: int | None = None,
        provider: str | None = None,
        rerank_method: str | None = None,
    ) -> "QueryResult":
        """Ask a question about ingested documents."""
        payload = {"question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        if provider is not None:
            payload["chat_provider"] = provider
        if rerank_method is not None:
            payload["rerank_method"] = rerank_method

        resp = self._client.post("/query", json=payload)
        resp.raise_for_status()
        return QueryResult(resp.json())

    def documents(self) -> list[dict]:
        """List ingested documents."""
        resp = self._client.get("/documents")
        resp.raise_for_status()
        return resp.json()

    def delete(self, source_file: str) -> dict:
        """Delete a document from the collection."""
        encoded = quote(source_file, safe="")
        resp = self._client.delete(f"/documents/{encoded}")
        resp.raise_for_status()
        return resp.json()

    def config(self) -> dict:
        """Get current configuration."""
        resp = self._client.get("/config")
        resp.raise_for_status()
        return resp.json()


class QueryResult:
    """Wrapper for query results with display methods."""

    def __init__(self, data: dict):
        self._data = data
        self.answer = data.get("answer", "")
        self.sources = data.get("sources", [])
        self.metadata = data.get("metadata")

    def show_sources(self):
        """Pretty-print sources (works in Jupyter)."""
        if not self.sources:
            print("No sources.")
            return

        for i, src in enumerate(self.sources, 1):
            print(
                f"[{i}] {src['file']} (p.{src['page']}) "
                f"— {src['section']}"
            )
            excerpt = src["excerpt"][:150]
            if len(src["excerpt"]) > 150:
                excerpt += "..."
            print(f"    {excerpt}")
            print()

    def __repr__(self):
        meta = ""
        if self.metadata:
            meta = (
                f" [{self.metadata['provider']}/"
                f"{self.metadata['model']} | "
                f"{self.metadata['latency_ms']}ms]"
            )
        return (
            f"QueryResult(answer='{self.answer[:80]}...', "
            f"sources={len(self.sources)}{meta})"
        )

    def _repr_markdown_(self):
        """Rich display in Jupyter notebooks."""
        parts = [f"**Answer:** {self.answer}\n"]

        if self.sources:
            parts.append("\n**Sources:**\n")
            for i, src in enumerate(self.sources, 1):
                parts.append(
                    f"{i}. **{src['file']}** (p.{src['page']}) "
                    f"— {src['section']}\n"
                )
                parts.append(
                    f"   > {src['excerpt'][:150]}...\n"
                )

        if self.metadata:
            parts.append(
                f"\n*{self.metadata['provider']}/"
                f"{self.metadata['model']} | "
                f"{self.metadata['retrieval_count']} chunks | "
                f"{self.metadata['latency_ms']}ms*"
            )

        return "\n".join(parts)
