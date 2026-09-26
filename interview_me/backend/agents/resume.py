import json

from langchain.tools import tool

from interview_me.backend.artifacts import load_profile
from interview_me.backend.config import settings


_FIELD_TERMS = {
    "self_introduction": ("introduce", "about you", "tell me about yourself"),
    "working_background": ("experience", "work", "background", "career"),
    "education": ("education", "degree", "university", "school"),
    "skills": ("skill", "technology", "technical", "programming", "stack"),
    "highlighted_project": ("highlighted project", "favorite project", "project"),
}


@tool
def get_general_info(query: str) -> str:
    """Return only approved resume facts relevant to a general interview query."""
    profile = load_profile(settings.profile_path)
    normalized_query = query.lower()
    matches = [
        {"source": f"resume:{field}", "text": profile[field]}
        for field, terms in _FIELD_TERMS.items()
        if field in profile and any(term in normalized_query for term in terms)
    ]
    return json.dumps({"status": "found" if matches else "not_found", "evidence": matches})

