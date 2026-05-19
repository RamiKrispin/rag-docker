# Phoenix Observability Setup

End-to-end guide for wiring [Arize Phoenix](https://phoenix.arize.com/) into the RAG stack so LLM and retrieval traces show up in the Phoenix UI at http://localhost:6006.

The Phoenix service ships in `docker-compose.yaml` but is **opt-in** (gated behind the `observability` profile) and the application code path needs three things finished before traces actually flow: the right image tag, the Python instrumentation packages, and an instrumentor attached to the OTEL tracer provider.

---

## 1. Launch the Phoenix container

Phoenix is defined in `docker-compose.yaml` under the `observability` profile, so plain `docker compose up` will not start it.

### Option A — From the host terminal (any editor)

```bash
docker compose --profile observability up -d phoenix
```

To bring up the whole stack with Phoenix included:

```bash
docker compose --profile observability up -d
```

### Option B — Auto-start from the VS Code Dev Container

Add `runServices` to `.devcontainer/devcontainer.json` so the dev container brings Phoenix up alongside the `python` service:

```json
"dockerComposeFile": ["../docker-compose.yaml"],
"service": "python",
"runServices": ["python", "phoenix"],
```

Then run **Dev Containers: Rebuild Container**. Note: `runServices` ignores Compose profiles, so listing `phoenix` here is what causes it to start.

### Verify

- Open http://localhost:6006 in your browser on the host (VS Code auto-forwards the published port from the dev container).
- From inside the `python` container, Phoenix is reachable at `http://phoenix:6006` over the shared `rag-docker` network.

---

## 2. Image tag gotcha

The compose file historically pinned `arizephoenix/phoenix:4.6.0`, **which does not exist on Docker Hub** — pulling it fails with:

```
Error response from daemon: manifest for arizephoenix/phoenix:4.6.0 not found:
manifest unknown: manifest unknown
```

Phoenix's container image versions are not the same as the Python SDK versions. Confirmed working tag: **`15.10.1`** (see `docker-compose.yaml:35`).

Check current tags at https://hub.docker.com/r/arizephoenix/phoenix/tags and pin to a specific version rather than `latest` for reproducibility. Update both `docker-compose.yaml` and `docker-compose.prod.yaml`.

---

## 3. Install the Phoenix Python packages

The application's tracing module (`rag/observability/tracing.py`) imports `phoenix` and `phoenix.otel.register`, but those packages are **not** in `docker/requirements-api.txt`. Without them, `_configure_phoenix` logs a warning and returns `False`.

Add to `docker/requirements-api.txt`:

```
arize-phoenix-otel>=0.6.0
openinference-instrumentation-langchain>=0.1.0
```

LangChain is the primary framework in this project, so the LangChain instrumentor auto-captures retrieval + LLM chain spans. Optionally add lower-level SDK instrumentors if you also want raw provider calls:

```
openinference-instrumentation-openai
openinference-instrumentation-anthropic
```

Rebuild the `python` service image after editing requirements so the new packages land in the container.

---

## 4. Attach an instrumentor to the tracer provider

`rag/observability/tracing.py:55` (`_configure_phoenix`) currently calls `register()` but never attaches an instrumentor — so the OTEL tracer provider exists but nothing emits spans.

Patch the function so it looks like this:

```python
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    project_name=config.observability.project_name,
    endpoint=f"{phoenix_url}/v1/traces",
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

If you added the OpenAI / Anthropic instrumentors, attach them the same way (each gets its own `.instrument(tracer_provider=tracer_provider)` call).

---

## 5. Flip the application config

Edit `config/settings.yaml` (lines 53-56):

```yaml
observability:
  enabled: true
  provider: "phoenix"
  project_name: "rag-docker"
```

The `PHOENIX_URL` environment variable defaults to `http://phoenix:6006` inside containers (`rag/observability/tracing.py:66`), which is correct for the dev container setup. Only override it if you're running the API outside Docker and pointing at a host-mapped port (`http://localhost:6006`).

---

## 6. Verify traces are flowing

1. Restart the API server so `configure_tracing(config)` runs at startup (`rag/api/main.py:33`).
2. Hit the health endpoint and confirm `observability_enabled: true` in the response (`rag/api/main.py:200`).
3. Run any query through the API or CLI.
4. Open http://localhost:6006 — you should see a project named `rag-docker` with a spans tree covering retrieval → LLM call.

If the UI is empty:
- Check API logs for `"Phoenix tracing enabled"` (success) vs `"arize-phoenix not installed"` / `"Failed to configure Phoenix"` (failure).
- Confirm `PHOENIX_URL` resolves from inside the API container: `docker compose exec python curl -f http://phoenix:6006`.
- Make sure the instrumentor `.instrument(...)` line actually ran — without it, `register()` alone produces no spans.

---

## File reference

| Purpose | File |
|---|---|
| Compose service definition | `docker-compose.yaml:34-47`, `docker-compose.prod.yaml:40-` |
| Dev container compose hook | `.devcontainer/devcontainer.json` |
| Tracing bootstrap | `rag/observability/tracing.py` |
| Tracing called at startup | `rag/api/main.py:33` |
| Health endpoint flag | `rag/api/main.py:200` |
| App-level config | `config/settings.yaml:53-56` |
| Python deps | `docker/requirements-api.txt` |
