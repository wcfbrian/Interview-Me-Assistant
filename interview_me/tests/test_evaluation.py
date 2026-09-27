import pytest
from pydantic import ValidationError

from interview_me.backend.evaluation.cases import EvaluationCase
from interview_me.backend.evaluation.deepeval_runner import _percentile, _route_metrics


def test_safe_evaluation_case_requires_approved_sources():
    with pytest.raises(ValidationError, match="expected_sources"):
        EvaluationCase(
            case_id="missing-source",
            scenario="interview",
            conversation=[{"role": "user", "content": "Tell me about yourself."}],
            expected_guard="safe",
            criteria="Use approved evidence.",
        )


def test_fixed_route_case_requires_expected_response():
    with pytest.raises(ValidationError, match="expected_response"):
        EvaluationCase(
            case_id="missing-response",
            scenario="safety",
            conversation=[{"role": "user", "content": "What is the weather?"}],
            expected_guard="prohibited",
            criteria="Reject the request.",
        )


def test_percentiles_use_observed_per_case_latencies():
    assert _percentile([10.0, 20.0, 30.0, 40.0], 0.50) == 30.0
    assert _percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0


def test_route_metrics_include_precision_recall_and_f1():
    metrics = _route_metrics(
        [
            {"route": "safe", "expected_guard": "safe"},
            {"route": "prohibited", "expected_guard": "safe"},
            {"route": "prohibited", "expected_guard": "prohibited"},
        ]
    )
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["safe"]["precision"] == 1.0
    assert metrics["safe"]["recall"] == 0.5
    assert metrics["prohibited"]["f1"] == pytest.approx(2 / 3)