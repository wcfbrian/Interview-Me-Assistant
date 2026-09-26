"""Extract an approved profile from the resume PDF and validate project data."""

import argparse
import json
from pathlib import Path

from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from pypdf import PdfReader

from interview_me.backend.config import settings


class CandidateProfile(BaseModel):
    """Facts exposed to the resume subagent."""

    self_introduction: str
    working_background: str
    education: str
    skills: str
    highlighted_project: str


def extract_resume_text(resume_path: Path) -> str:
    """Extract text from every page without modifying the source PDF."""
    reader = PdfReader(str(resume_path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError(f"No text could be extracted from {resume_path}")
    return text


def extract_profile(resume_text: str) -> CandidateProfile:
    """Convert resume text into the closed runtime profile schema."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    structured_model = model.with_structured_output(CandidateProfile)
    return structured_model.invoke([
        {
            "role": "system",
            "content": (
                "Extract only facts explicitly present in the resume. Do not infer "
                "or embellish. Return concise first-person interview-ready text for "
                "each field. If a field is absent, write 'No information found.'"
            ),
        },
        {"role": "user", "content": resume_text},
    ])


def load_projects(projects_path: Path) -> list[dict[str, str]]:
    with projects_path.open(encoding="utf-8") as handle:
        projects = json.load(handle)
    if not isinstance(projects, list):
        raise ValueError("projects.json must contain a JSON array")
    return [project for project in projects if isinstance(project, dict)]


def ingest(resume_path: Path, projects_path: Path, output_dir: Path) -> CandidateProfile:
    """Build runtime profile.json and projects.json from real source artifacts."""
    profile = extract_profile(extract_resume_text(resume_path))
    projects = load_projects(projects_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "projects.json").write_text(json.dumps(projects, indent=2), encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=settings.resume_path)
    parser.add_argument("--projects", type=Path, default=settings.project_index_path)
    parser.add_argument("--output", type=Path, default=settings.data_dir)
    args = parser.parse_args()
    profile = ingest(args.resume, args.projects, args.output)
    print(f"Extracted profile and indexed {len(load_projects(args.projects))} projects to {args.output}")
    print(f"Profile fields: {', '.join(profile.model_fields)}")


if __name__ == "__main__":
    main()