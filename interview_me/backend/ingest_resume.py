"""Extract an approved candidate profile from a resume PDF."""

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
