"""Build runtime candidate artifacts from resume and GitHub source inputs."""

import argparse
from pathlib import Path

from src.backend.config import settings
from .ingest_project import (
    get_project_bm25,
    get_project_vector_store,
    load_projects,
    pipeline,
)
from .ingest_resume import (
    CandidateProfile,
    extract_profile,
    extract_resume_text,
)


def ingest(input_resume: Path, output_dir: Path, regenerate_json: bool = False) -> None:
    """Build runtime profile.json and projects.json from real source artifacts."""
    profile_path = output_dir / "profile.json"
    projects_path = output_dir / "projects.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if regenerate_json or not profile_path.exists():
        profile = extract_profile(extract_resume_text(input_resume))
        profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    if regenerate_json or not projects_path.exists():
        if not settings.github_username:
            raise ValueError(
                f"{projects_path} does not exist and GITHUB_USERNAME is not configured"
            )
        pipeline(settings.github_username, projects_path, regenerate_json=True)
    else:
        pipeline(settings.github_username, projects_path, regenerate_json=False)


def main() -> None:
    """Run ingestion from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-resume", type=Path, default=settings.resume_path)
    parser.add_argument("--output-dir", type=Path, default=settings.data_dir)
    parser.add_argument(
        "--regenerate-json",
        "--regenerate",
        dest="regenerate_json",
        action="store_true",
        help="regenerate profile.json and projects.json from their source inputs",
    )
    args = parser.parse_args()
    ingest(args.input_resume, args.output_dir, regenerate_json=args.regenerate_json)
    print(
        f"Extracted profile and indexed {len(load_projects(settings.project_path))} "
        f"projects to {args.output_dir}"
    )
    print(f"Profile fields: {', '.join(CandidateProfile.model_fields)}")


__all__ = [
    "CandidateProfile",
    "extract_profile",
    "extract_resume_text",
    "get_project_bm25",
    "get_project_vector_store",
    "ingest",
    "load_projects",
    "main",
    "pipeline",
]