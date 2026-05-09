import json

import click
import httpx


DEFAULT_API_URL = "http://localhost:8080"


def _handle_response(response):
    """Check response status and extract error detail safely."""
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Unknown error")
        except Exception:
            detail = response.text[:200]
        click.echo(f"Error ({response.status_code}): {detail}",
                   err=True)
        raise SystemExit(1)
    return response


def _request(method, url, **kwargs):
    """Make HTTP request with error handling."""
    try:
        resp = getattr(httpx, method)(url, **kwargs)
    except httpx.ConnectError:
        click.echo("Error: cannot connect to API", err=True)
        raise SystemExit(1)
    except httpx.TimeoutException:
        click.echo("Error: request timed out", err=True)
        raise SystemExit(1)
    except httpx.HTTPError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)
    return resp


@click.group()
@click.option(
    "--api-url", default=DEFAULT_API_URL,
    envvar="RAG_API_URL",
    help="RAG API base URL",
)
@click.pass_context
def cli(ctx, api_url):
    """RAG Docker CLI — query financial documents."""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api_url


@cli.command()
@click.option("--source-dir", default="docs/", help="PDF directory")
@click.option(
    "--method", default="recursive",
    type=click.Choice(["recursive", "semantic", "by_title"]),
    help="Chunking method",
)
@click.option("--chunk-size", default=1000, help="Chunk size")
@click.option("--chunk-overlap", default=200, help="Chunk overlap")
@click.pass_context
def ingest(ctx, source_dir, method, chunk_size, chunk_overlap):
    """Ingest PDF documents into ChromaDB."""
    api_url = ctx.obj["api_url"]
    click.echo(f"Ingesting from {source_dir}...")

    response = _request("post", f"{api_url}/ingest", json={
        "source_dir": source_dir,
        "chunking_method": method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }, timeout=300.0)
    _handle_response(response)

    data = response.json()
    click.echo(f"Ingested {data['documents_ingested']} documents")
    click.echo(f"Total chunks: {data['total_chunks']}")
    click.echo("Files:")
    for f in data["files"]:
        click.echo(f"  - {f}")


@cli.command()
@click.argument("question")
@click.option("--top-k", default=None, type=int, help="Top K results")
@click.option(
    "--provider", default=None,
    type=click.Choice(["openai", "anthropic", "gemini"]),
    help="Chat provider override",
)
@click.option(
    "--rerank", default=None,
    type=click.Choice(["cross-encoder", "none"]),
    help="Rerank method",
)
@click.pass_context
def query(ctx, question, top_k, provider, rerank):
    """Ask a question about the ingested documents."""
    api_url = ctx.obj["api_url"]

    payload = {"question": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if provider is not None:
        payload["chat_provider"] = provider
    if rerank is not None:
        payload["rerank_method"] = rerank

    response = _request("post", f"{api_url}/query",
                        json=payload, timeout=120.0)
    _handle_response(response)

    data = response.json()

    click.echo(f"\n{data['answer']}\n")
    click.echo("--- Sources ---")
    for src in data["sources"]:
        click.echo(
            f"  [{src['file']} p.{src['page']}] "
            f"{src['section']}"
        )
        excerpt = src["excerpt"][:100]
        if len(src["excerpt"]) > 100:
            excerpt += "..."
        click.echo(f"    {excerpt}")

    if data.get("metadata"):
        meta = data["metadata"]
        click.echo(
            f"\n[{meta['provider']}/{meta['model']} | "
            f"{meta['retrieval_count']} chunks | "
            f"{meta['latency_ms']}ms]"
        )


@cli.command()
@click.pass_context
def docs(ctx):
    """List ingested documents."""
    api_url = ctx.obj["api_url"]

    response = _request("get", f"{api_url}/documents", timeout=30.0)
    _handle_response(response)

    data = response.json()
    if not data:
        click.echo("No documents ingested yet.")
        return

    click.echo(f"{'Document':<40} {'Chunks':>8}")
    click.echo("-" * 50)
    for doc in data:
        click.echo(f"{doc['file']:<40} {doc['chunks']:>8}")
    click.echo(f"\nTotal: {len(data)} documents")


@cli.command()
@click.pass_context
def health(ctx):
    """Check API and ChromaDB health."""
    api_url = ctx.obj["api_url"]

    response = _request("get", f"{api_url}/health", timeout=10.0)
    _handle_response(response)

    data = response.json()
    click.echo("API:      online")
    click.echo(f"ChromaDB: {data['chromadb']}")
    click.echo(f"Documents: {data['documents']}")


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    api_url = ctx.obj["api_url"]

    response = _request("get", f"{api_url}/config", timeout=10.0)
    _handle_response(response)
    click.echo(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    cli()
