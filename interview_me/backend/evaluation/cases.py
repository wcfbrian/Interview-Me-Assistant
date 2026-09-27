"""Typed, reviewable inputs for interview-assistant evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


GuardRoute = Literal["safe", "greeting", "prohibited"]
Scenario = Literal["interview", "follow_up", "robustness", "safety"]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class EvaluationCase(BaseModel):
    """One reviewed evaluation scenario grounded in approved candidate data."""

    case_id: str = Field(min_length=1)
    scenario: Scenario
    conversation: list[ConversationTurn] = Field(min_length=1)
    expected_guard: GuardRoute
    expected_sources: list[str] = Field(default_factory=list)
    factual_claims: list[str] = Field(default_factory=list)
    criteria: str = Field(min_length=1)
    expected_response: str | None = None
    latency_budget_ms: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def requires_a_final_user_turn(self) -> "EvaluationCase":
        if self.conversation[-1].role != "user":
            raise ValueError("conversation must end with a user turn")
        if self.expected_guard == "safe" and not self.expected_sources:
            raise ValueError("safe cases must name approved expected_sources")
        if self.expected_guard != "safe" and not self.expected_response:
            raise ValueError("non-safe cases must specify the fixed expected_response")
        return self

    @property
    def prompt(self) -> str:
        return self.conversation[-1].content


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate a curated JSON evaluation dataset."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(row) for row in rows]