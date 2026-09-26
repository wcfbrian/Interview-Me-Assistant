"""Deterministic safety and relevance policy used before model routing."""

from dataclasses import dataclass

FALLBACK_OFF_TOPIC = "Please ask a relevant question about the candidate."
FALLBACK_NO_INFORMATION = "No information found."
FALLBACK_UNRELIABLE = "I am not able to answer that reliably."
FALLBACK_GREETING = "Hello. Please ask me about the candidate's experience."

_INTERVIEW_TERMS = frozenset(
    {
        "background",
        "candidate",
        "career",
        "education",
        "experience",
        "introduction",
        "interview",
        "introduce",
        "leadership",
        "project",
        "projects",
        "resume",
        "skill",
        "skills",
        "technical",
        "work",
    }
)
_INJECTION_PHRASES = (
    "ignore your instructions",
    "ignore previous instructions",
    "reveal the system prompt",
    "show the system prompt",
    "jailbreak",
)
_GREETING_TERMS = frozenset({"hello", "hi", "hey"})


@dataclass(frozen=True)
class GuardDecision:
    """The policy result passed to the model-backed guard node."""

    is_relevant: bool
    response: str | None = None
    is_greeting: bool = False


def classify_query(query: str) -> GuardDecision:
    """Classify obvious policy cases before making an LLM request.

    The model-backed guard remains responsible for nuanced cases. This cheap
    boundary protects fixed fallbacks and rejects known injection patterns.
    """

    normalized = " ".join(query.casefold().split())
    tokens = {token.strip(".,?!:;()[]{}") for token in normalized.split()}
    if not normalized or any(phrase in normalized for phrase in _INJECTION_PHRASES):
        return GuardDecision(False, FALLBACK_OFF_TOPIC)

    if tokens and tokens <= _GREETING_TERMS:
        return GuardDecision(True, FALLBACK_GREETING, True)
    if any(term in tokens for term in _INTERVIEW_TERMS):
        return GuardDecision(True)

    return GuardDecision(False, FALLBACK_OFF_TOPIC)