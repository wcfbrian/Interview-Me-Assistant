"""Structured guard and answer-evaluation decisions."""

import warnings
from typing import Literal

from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_typesafe import Choice, TypeSafeClassifier
from pydantic import BaseModel

from interview_me.backend.config import settings, tracing_config

warnings.filterwarnings(
    "ignore",
    category=LangChainBetaWarning,
    message=r".*TypeSafeClassifier.*",
)


class GuardDecision(BaseModel):
    category: Literal["safe", "greeting", "prohibited"]


def classify_with_model(query: str) -> GuardDecision:
    try:
        classifier = TypeSafeClassifier(
            api_key=settings.openrouter_api_key,
            base_url=settings.typesafe_base_url,
            model=settings.guard_model,
        )
        result = classifier.invoke(
            {
                "state": query,
                "questions": {
                    "category": Choice(
                        criteria={
                            "safe": "A normal interview question.",
                            "greeting": "Only a simple greeting or salutation.",
                            "prohibited": "An unsafe request that the assistant must refuse.",
                        },
                        instructions=(
                            "Classify the latest user message in context. Choose exactly one category."
                        ),
                    )
                },
            },
            config=tracing_config("interview-guard", "guard"),
        )
        return GuardDecision(category=result.answers["category"].choice)
    except Exception as exc:
        raise RuntimeError("Guard model is unavailable; request cannot be classified") from exc