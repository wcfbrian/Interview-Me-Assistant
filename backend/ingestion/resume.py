"""Resume PDF extraction with a typed profile boundary."""

from pathlib import Path

from pypdf import PdfReader


def extract_resume_text(path: str | Path) -> str:
    """Extract text from a text-based PDF and fail clearly for empty scans."""

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError("Resume PDF contains no extractable text.")
    return text