import asyncio
from types import SimpleNamespace
from typing import cast

import eval.eval_agent as eval_agent
from eval.eval_agent import EvaluationDataset, LLMTestCase, build_test_cases


def test_build_test_cases_uses_cache_when_goldens_change(monkeypatch, tmp_path):
    cached_test_case = LLMTestCase(
        input="Old prompt",
        actual_output="Cached answer",
        expected_output="Old expected answer",
    )
    dataset = cast(
        EvaluationDataset,
        SimpleNamespace(
            goldens=[
                SimpleNamespace(
                    input="Updated prompt",
                    expected_output="Updated expected answer",
                )
            ]
        ),
    )

    monkeypatch.setattr(eval_agent, "TEST_CASE_CACHE_PATH", tmp_path / "cache.jsonl")
    eval_agent._save_test_cases([cached_test_case])
    monkeypatch.setattr(
        "eval.eval_agent.build_graph",
        lambda: (_ for _ in ()).throw(AssertionError("graph should not be built")),
    )

    test_cases = asyncio.run(build_test_cases(dataset))

    assert len(test_cases) == 1
    assert test_cases[0].input == "Old prompt"
    assert test_cases[0].actual_output == "Cached answer"


def test_build_test_cases_rerun_bypasses_cache(monkeypatch):
    golden = SimpleNamespace(
        input="Tell me about your experience",
        expected_output="Expected answer",
        context=["Expected context"],
        additional_metadata={"route": "resume"},
        comments=None,
        name="experience",
    )
    dataset = cast(EvaluationDataset, SimpleNamespace(goldens=[golden]))

    class FakeGraph:
        async def ainvoke(self, payload):
            assert payload["messages"][0]["content"] == golden.input
            return {
                "messages": [SimpleNamespace(content="Generated answer")],
                "context": ["Retrieved evidence"],
            }

    monkeypatch.setattr(
        "eval.eval_agent._load_cached_test_cases",
        lambda _: (_ for _ in ()).throw(AssertionError("cache should be bypassed")),
    )
    monkeypatch.setattr("eval.eval_agent.build_graph", lambda: FakeGraph())
    saved = []
    monkeypatch.setattr("eval.eval_agent._save_test_cases", lambda cases: saved.extend(cases))

    test_cases = asyncio.run(build_test_cases(dataset, rerun=True))

    assert test_cases[0].actual_output == "Generated answer"
    assert test_cases[0].retrieval_context == ["Retrieved evidence"]
    assert saved == test_cases