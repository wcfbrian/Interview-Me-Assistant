"""Deterministic metrics for guard, retrieval, quality, and latency reports."""

from statistics import mean


def binary_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, float]:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have the same length.")
    tp = sum(actual and guess for actual, guess in zip(expected, predicted, strict=True))
    fp = sum(not actual and guess for actual, guess in zip(expected, predicted, strict=True))
    fn = sum(actual and not guess for actual, guess in zip(expected, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def latency_summary(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds:
        return {"count": 0, "mean": 0.0, "p95": 0.0}
    ordered = sorted(milliseconds)
    index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {"count": len(ordered), "mean": mean(ordered), "p95": ordered[index]}