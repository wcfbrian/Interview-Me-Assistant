import pytest

from backend.graph import build_graph
from backend.policy import FALLBACK_NO_INFORMATION, FALLBACK_UNRELIABLE
from backend.schemas import GuardResult


class FakeSettings:
    guard_model = "unused"
    openrouter_api_key = "unused"
    openrouter_base_url = "https://example.invalid"
    request_timeout_seconds = 1.0


@pytest.mark.asyncio
async def test_relevant_query_without_evidence_returns_no_information() -> None:
    async def guard(_: str):
        return GuardResult(is_relevant=True)

    graph = build_graph(settings=FakeSettings(), guard=guard)

    result = await graph.ainvoke({"query": "What projects have you worked on?"})

    assert result["response"] == FALLBACK_NO_INFORMATION


@pytest.mark.asyncio
async def test_generated_answer_is_not_released_without_judge_approval() -> None:
    async def retrieve(_: str):
        return []

    async def generate(*_args):
        return "Unsupported answer"

    async def guard(_: str):
        return GuardResult(is_relevant=True)

    graph = build_graph(
        settings=FakeSettings(), retrieve=retrieve, generate=generate, guard=guard
    )

    result = await graph.ainvoke({"query": "Tell me about your project experience."})

    assert result["response"] == FALLBACK_NO_INFORMATION
    assert result["response"] != FALLBACK_UNRELIABLE


@pytest.mark.asyncio
async def test_greeting_skips_retrieval() -> None:
    async def guard(_: str):
        return GuardResult(is_relevant=True, is_greeting=True, reason="Hello.")

    async def retrieve(_: str):
        raise AssertionError("greetings must not retrieve evidence")

    graph = build_graph(settings=FakeSettings(), guard=guard, retrieve=retrieve)

    result = await graph.ainvoke({"query": "Hi"})

    assert result["response"] == "Hello."