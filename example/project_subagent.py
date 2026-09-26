import json
import re

from langchain.tools import tool

from backend.project.artifacts import load_projects
from backend.project.config import settings


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


@tool
def get_project_detail(query: str) -> str:
    """Return project summaries whose approved README text matches the query."""
    query_terms = _terms(query)
    ranked = []
    for project in load_projects(settings.project_path):
        text = " ".join(str(project.get(key, "")) for key in ("name", "summary", "readme"))
        score = len(query_terms & _terms(text))
        if score:
            ranked.append((score, project))
    ranked.sort(key=lambda item: item[0], reverse=True)
    evidence = [
        {
            "source": f"project:{project.get('name', 'unknown')}",
            "text": project.get("readme") or project.get("summary", ""),
            "score": score,
        }
        for score, project in ranked[:3]
    ]
    return json.dumps({"status": "found" if evidence else "not_found", "evidence": evidence})