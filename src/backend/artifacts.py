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