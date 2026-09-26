import json
from langchain.tools import tool

from interview_me.backend.artifacts import load_projects
from interview_me.backend.config import settings
from interview_me.backend.ingest_project import get_project_vector_store


@tool
def get_project_detail(query: str) -> str:
    """Return semantically relevant project evidence from the Chroma index."""
    projects_by_name = {
        str(project.get("name", "unknown")): project
        for project in load_projects(settings.project_path)
    }
    matches = get_project_vector_store().similarity_search_with_relevance_scores(query, k=3)
    evidence = [
        {
            "source": f"project:{document.metadata.get('name', 'unknown')}",
            "text": projects_by_name.get(
                document.metadata.get("name", ""),
                {"readme": document.page_content},
            ).get("readme") or document.page_content,
            "score": score,
        }
        for document, score in matches
    ]
    return json.dumps({"status": "found" if evidence else "not_found", "evidence": evidence})