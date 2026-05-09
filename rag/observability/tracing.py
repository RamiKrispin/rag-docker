import os

from rag.config import Settings
from rag.observability.logging import get_logger


_tracing_active = False


def configure_tracing(config: Settings) -> bool:
    global _tracing_active

    if not config.observability.enabled:
        return False

    provider = config.observability.provider

    if provider == "langsmith":
        _tracing_active = _configure_langsmith(config)
    elif provider == "phoenix":
        _tracing_active = _configure_phoenix(config)
    else:
        get_logger().warning(
            f"Unknown observability provider: {provider}"
        )
        _tracing_active = False

    return _tracing_active


def _configure_langsmith(config: Settings) -> bool:
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    if not api_key or api_key == "key_is_missing":
        get_logger().warning(
            "LANGSMITH_API_KEY not set, tracing disabled"
        )
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = (
        config.observability.project_name
    )

    get_logger().info(
        "LangSmith tracing enabled",
        extra={
            "extra_data": {
                "project": config.observability.project_name,
            }
        },
    )
    return True


def _configure_phoenix(config: Settings) -> bool:
    try:
        import phoenix as px
    except ImportError:
        get_logger().warning(
            "arize-phoenix not installed, tracing disabled. "
            "Install with: pip install arize-phoenix"
        )
        return False

    phoenix_url = os.environ.get(
        "PHOENIX_URL", "http://phoenix:6006"
    )

    try:
        from phoenix.otel import register
        register(
            project_name=config.observability.project_name,
            endpoint=f"{phoenix_url}/v1/traces",
        )
        get_logger().info(
            "Phoenix tracing enabled",
            extra={
                "extra_data": {
                    "project": config.observability.project_name,
                    "endpoint": phoenix_url,
                }
            },
        )
        return True
    except Exception as e:
        get_logger().warning(
            f"Failed to configure Phoenix: {e}"
        )
        return False


def is_tracing_enabled() -> bool:
    return _tracing_active
