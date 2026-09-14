"""Factory for the production LLM client (Azure AI Inference, Kimi-K2.6)."""

from __future__ import annotations

from typing import Any

from .config import Settings


def get_llm(settings: Settings) -> Any:
    """Build the Azure-hosted chat model used in production.

    Imported lazily so the rest of the package does not require the Azure SDK
    to be installed for offline test runs.
    """
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

    return AzureAIChatCompletionsModel(
        endpoint=settings.azure_endpoint,
        credential=settings.azure_api_key,
        model_name=settings.azure_model,
    )
