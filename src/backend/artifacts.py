"""Load approved, local candidate artifacts without inventing missing facts."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_profile(path: Path) -> dict[str, str]:
    profile = load_json(path, {})
    if not isinstance(profile, dict):
        raise ValueError("profile.json must contain a JSON object")
    return {str(key): str(value) for key, value in profile.items()}


def load_projects(path: Path) -> list[dict[str, str]]:
    projects = load_json(path, [])
    if not isinstance(projects, list):
        raise ValueError("projects.json must contain a JSON array")
    return [item for item in projects if isinstance(item, dict)]


def validate_data_artifacts(
    profile_path: Path,
    project_path: Path,
    project_bm25_index_path: Path,
) -> None:
    required_paths = (profile_path, project_path, project_bm25_index_path)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(
            f"Interview data is not initialized. Missing required artifacts: {missing}. "
            "Run `uv run python -m src.backend.ingest --regenerate-json` to create them."
        )