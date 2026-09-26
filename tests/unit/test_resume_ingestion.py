import pytest

from backend.ingestion.resume import extract_resume_text


def test_missing_resume_path_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        extract_resume_text("does-not-exist.pdf")