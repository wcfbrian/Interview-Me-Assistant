"""Local DeepEval metrics for curated interview-assistant cases."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain.chat_models import init_chat_model

from interview_me.backend.graph.projects import _retrieve_project_evidence
from interview_me.backend.artifacts import load_profile, load_projects
from interview_me.backend.config import settings
from interview_me.backend.evaluation.adapter import run_case
from interview_me.backend.evaluation.cases import EvaluationCase, load_cases


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the project's configured OpenRouter model."""

    def __init__(self, model_name: str) -> None:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for DeepEval runs")
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return init_chat_model(
            self.model_name,
            model_provider="openrouter",
            temperature=0,
        )

    def generate(self, prompt: str, schema=None):
        model = self.model.with_structured_output(schema) if schema else self.model
        response = model.invoke(prompt)
        return response if schema else str(response.content)

    async def a_generate(self, prompt: str, schema=None):
        model = self.model.with_structured_output(schema) if schema else self.model
        response = await model.ainvoke(prompt)
        return response if schema else str(response.content)

    def get_model_name(self) -> str:
        return self.model_name


def approved_contexts(case: EvaluationCase) -> list[str]:
    """Resolve the approved artifact text named by a reviewed case."""
    profile = load_profile(settings.profile_path)
    projects = {project["name"]: project for project in load_projects(settings.project_path)}
    contexts: list[str] = []

    for source in case.expected_sources:
        prefix, _, name = source.partition(":")
        if prefix == "resume" and name in profile:
            contexts.append(f"{source}: {profile[name]}")
        elif prefix == "project" and name in projects:
            project = projects[name]
            contexts.append(f"{source}: {project.get('readme') or project.get('summary', '')}")
        else:
            raise ValueError(f"Case {case.case_id} names an unknown source: {source}")
    return contexts


def retrieval_contexts(case: EvaluationCase) -> list[str]:
    """Capture actual project retrieval output for retrieval metrics when relevant."""
    if not any(source.startswith("project:") for source in case.expected_sources):
        return []
    payload = json.loads(_retrieve_project_evidence.invoke({"query": case.prompt}))
    return [f"{item['source']}: {item['text']}" for item in payload.get("evidence", [])]


def _score(metric, test_case: LLMTestCase) -> dict[str, Any]:
    metric.measure(test_case)
    return {
        "score": metric.score,
        "passed": metric.is_successful(),
        "reason": metric.reason,
    }


def _quality_scores(result, judge: OpenRouterJudge, contexts: list[str], retrieved: list[str]) -> dict[str, dict[str, Any]]:
    expected_output = "\n".join(result.case.factual_claims) or result.case.criteria
    test_case = LLMTestCase(
        input=result.case.prompt,
        actual_output=result.answer,
        expected_output=expected_output,
        context=contexts,
        retrieval_context=retrieved or contexts,
    )
    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(model=judge, async_mode=False),
        "faithfulness": FaithfulnessMetric(model=judge, async_mode=False),
        "interview_style": GEval(
            name="InterviewStyle",
            criteria=result.case.criteria,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=judge,
            async_mode=False,
        ),
    }
    if retrieved:
        metrics["contextual_precision"] = ContextualPrecisionMetric(model=judge, async_mode=False)
        metrics["contextual_recall"] = ContextualRecallMetric(model=judge, async_mode=False)
    return {name: _score(metric, test_case) for name, metric in metrics.items()}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _route_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("safe", "greeting", "prohibited")
    metrics: dict[str, Any] = {
        "accuracy": mean(row["route"] == row["expected_guard"] for row in rows),
        "confusion_matrix": {
            expected: dict(Counter(row["route"] for row in rows if row["expected_guard"] == expected))
            for expected in labels
        },
    }
    for label in labels:
        true_positive = sum(row["route"] == label and row["expected_guard"] == label for row in rows)
        predicted = sum(row["route"] == label for row in rows)
        actual = sum(row["expected_guard"] == label for row in rows)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return metrics


async def evaluate(cases: list[EvaluationCase], judge: OpenRouterJudge) -> dict[str, Any]:
    """Run the bot and live DeepEval metrics for each curated case."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = await run_case(case)
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "scenario": case.scenario,
            "expected_guard": case.expected_guard,
            "route": result.route,
            "answer": result.answer,
            "latency_ms": result.latency_ms,
            "fixed_response_matches": result.answer == case.expected_response if case.expected_response else None,
        }
        if result.route == "safe":
            contexts = approved_contexts(case)
            retrieved = retrieval_contexts(case)
            row["sources"] = case.expected_sources
            row["quality"] = _quality_scores(result, judge, contexts, retrieved)
        rows.append(row)

    latencies = [row["latency_ms"] for row in rows]
    quality_scores: dict[str, list[float]] = {}
    for row in rows:
        for metric, metric_result in row.get("quality", {}).items():
            quality_scores.setdefault(metric, []).append(metric_result["score"])
    robustness_rows = [row for row in rows if row["scenario"] in {"robustness", "safety"}]
    return {
        "model": settings.evaluation_model,
        "case_count": len(rows),
        "route_metrics": _route_metrics(rows),
        "quality_averages": {name: mean(scores) for name, scores in quality_scores.items()},
        "robustness_pass_rate": mean(row["route"] == row["expected_guard"] for row in robustness_rows) if robustness_rows else None,
        "latency_ms": {"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95), "max": max(latencies, default=0.0)},
        "results": rows,
    }


def run(dataset_path: Path, report_path: Path) -> dict[str, Any]:
    """Evaluate a curated dataset and persist a machine-readable report."""
    report = asyncio.run(evaluate(load_cases(dataset_path), OpenRouterJudge(settings.evaluation_model)))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run local DeepEval interview quality metrics.")
    parser.add_argument("--dataset", type=Path, default=Path("interview_me/backend/evaluation/datasets/interview_cases.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluations/deepeval-report.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.report), indent=2))


if __name__ == "__main__":
    main()