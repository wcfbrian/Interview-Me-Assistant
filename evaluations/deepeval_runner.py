"""Optional DeepEval runner using Jev as the evaluation-only LLM judge."""

from backend.config import get_settings


def evaluation_model_name() -> str:
    """Keep production judging separate from the evaluation model."""

    return get_settings().evaluation_model