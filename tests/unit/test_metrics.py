from evaluations.metrics import binary_metrics, latency_summary


def test_binary_metrics_reports_precision_recall_and_f1() -> None:
    result = binary_metrics([True, True, False], [True, False, False])

    assert result == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3}


def test_latency_summary_reports_p95() -> None:
    result = latency_summary([10, 20, 30, 40])

    assert result["count"] == 4
    assert result["p95"] == 40