import pytest

from src.backend.artifacts import (
    load_profile,
    load_projects,
    validate_data_artifacts,
)
from src.backend.config import (
    PROJECT_BM25_INDEX_FILENAME,
    PROFILE_FILENAME,
    PROJECTS_FILENAME,
)


# Verifies JSON artifact loaders preserve profile fields and project records using temporary JSON files.
def test_artifacts_preserve_only_json_profile_and_projects(tmp_path):
    profile_path = tmp_path / "profile.json"
    projects_path = tmp_path / "projects.json"
    profile_path.write_text('{"skills": "Python"}', encoding="utf-8")
    projects_path.write_text('[{"name": "demo", "readme": "RAG"}]', encoding="utf-8")

    assert load_profile(profile_path) == {"skills": "Python"}
    assert load_projects(projects_path)[0]["name"] == "demo"


def test_validate_data_artifacts_reports_all_missing_files(tmp_path):
    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma"

    with pytest.raises(RuntimeError) as error:
        validate_data_artifacts(
            data_dir / PROFILE_FILENAME,
            data_dir / PROJECTS_FILENAME,
            chroma_dir / PROJECT_BM25_INDEX_FILENAME,
        )

    message = str(error.value)
    assert "profile.json" in message
    assert "projects.json" in message
    assert PROJECT_BM25_INDEX_FILENAME in message
    assert "uv run python -m src.backend.ingest --regenerate-json" in message


def test_validate_data_artifacts_accepts_ingested_files(tmp_path):
    data_dir = tmp_path / "data"
    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir(parents=True)
    (data_dir / PROFILE_FILENAME).write_text("{}", encoding="utf-8")
    (data_dir / PROJECTS_FILENAME).write_text("[]", encoding="utf-8")
    (chroma_dir / PROJECT_BM25_INDEX_FILENAME).touch()

    validate_data_artifacts(
        data_dir / PROFILE_FILENAME,
        data_dir / PROJECTS_FILENAME,
        chroma_dir / PROJECT_BM25_INDEX_FILENAME,
    )