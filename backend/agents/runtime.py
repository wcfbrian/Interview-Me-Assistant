"""Shared OpenRouter model construction."""

from langchain_openai import ChatOpenAI

from backend.config import Settings


def create_chat_model(settings: Settings, model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
    )