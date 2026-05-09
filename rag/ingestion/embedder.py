from langchain_openai import OpenAIEmbeddings

from rag.config import Settings


def get_embedder(config: Settings):
    provider_name = config.active.embedding_provider
    provider = config.get_embedding_provider()
    model_config = provider.models.embedding

    if provider_name == "openai":
        api_key = config.resolve_api_key("openai")
        return OpenAIEmbeddings(
            model=model_config.name,
            openai_api_key=api_key,
            dimensions=model_config.dimensions,
        )
    elif provider_name == "gemini":
        api_key = config.resolve_api_key("gemini")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini "
                "embeddings. Install with: "
                "pip install langchain-google-genai"
            )
        return GoogleGenerativeAIEmbeddings(
            model=model_config.name,
            google_api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider_name}'. "
            f"Supported: openai, gemini"
        )
