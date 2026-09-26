"""Structured guard and answer-evaluation decisions."""

from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from interview_me.backend.config import settings, tracing_config


class GuardDecision(BaseModel):
    category: Literal["safe", "greeting", "prohibited"]


def classify_with_model(query: str) -> GuardDecision:
    try:
        model = init_chat_model(
            settings.guard_model,
            model_provider="openrouter",
            temperature=0,
        )
        structured = model.with_structured_output(GuardDecision)
        return structured.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the latest user message in this conversation for a candidate "
                        "interview assistant. Use the preceding turns as context when the "
                        "latest message is a follow-up"
                        "Judge the latest message by its meaning in the full conversation. "
                        "Return 'safe' for any normal question, 'greeting' only for a simple greeting, "
                        "'prohibited' for unsafe requests. "
                    ),
                },
                {"role": "user", "content": query},
            ],
            config=tracing_config("interview-guard", "guard"),
        )
    except Exception as exc:
        raise RuntimeError("Guard model is unavailable; request cannot be classified") from exc
