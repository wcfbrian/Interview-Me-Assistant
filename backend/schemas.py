"""Typed contracts shared across agent nodes."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

DecisionKind = Literal["relevant", "off_topic", "greeting"]


class GuardResult(BaseModel):
    is_relevant: bool
    is_greeting: bool = False
    reason: str = Field(default="")


class JudgeResult(BaseModel):
    approved: bool
    reason: str = Field(default="")


class CandidateProfile(BaseModel):
    """Allowlisted professional facts extracted from a resume."""

    name: str = ""
    summary: str = ""
    education: list[str] = Field(default_factory=list)
    work_history: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    highlighted_projects: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Source:
    source_type: Literal["resume", "github"]
    name: str
    locator: str


@dataclass(frozen=True)
class Evidence:
    text: str
    source: Source
    score: float = 0.0


@dataclass
class AgentState:
    """State passed through the workflow; only main_history is conversational memory."""

    query: str
    main_history: list[dict[str, str]] = field(default_factory=list)
    guard_decision: DecisionKind | None = None
    evidence: list[Evidence] = field(default_factory=list)
    answer: str | None = None
    response: str | None = None
    approved: bool | None = None
    provenance: list[Source] = field(default_factory=list)