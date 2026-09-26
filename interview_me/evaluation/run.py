"""Run deterministic routing metrics without making model calls."""

import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from interview_me.backend.graph import guard_route


def _classification_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    labels = {row["expected_guard"] for row in rows}
    metrics = {}
    predictions = [guard_route({"messages": [HumanMessage(content=row["query"])]}) for row in rows]
    for label in sorted(labels):
        true_positive = sum(prediction == label and row["expected_guard"] == label for prediction, row in zip(predictions, rows))
        predicted = predictions.count(label)
        actual = sum(row["expected_guard"] == label for row in rows)
        metrics[f"{label}_precision"] = true_positive / predicted if predicted else 0.0
        metrics[f"{label}_recall"] = true_positive / actual if actual else 0.0
    return metrics


def run(dataset_path: Path) -> dict[str, float]:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    metrics = _classification_metrics(rows)
    predictions = [guard_route({"messages": [HumanMessage(content=row["query"])]}) for row in rows]
    metrics["guard_accuracy"] = sum(prediction == row["expected_guard"] for prediction, row in zip(predictions, rows)) / len(rows)
    metrics["p50_latency_ms"] = (time.perf_counter() - started) * 1000 / len(rows)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(Path("interview_me/evaluation/datasets/interview_cases.json")), indent=2))