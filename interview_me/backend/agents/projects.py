import json
from functools import lru_cache

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from interview_me.backend.artifacts import load_projects
from interview_me.backend.config import settings, tracing_config
from interview_me.backend.ingest_project import get_project_vector_store


@tool("retrieve_project_evidence")
def _retrieve_project_evidence(query: str) -> str:
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


@lru_cache(maxsize=1)
def build_project_agent():
    """Build the project subagent that owns semantic project retrieval."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[_retrieve_project_evidence],
        system_prompt=(
            "You are the project evidence subagent. Always call the retrieval tool. "
            "Return its JSON result unchanged. Do not invent, summarize, or add facts."
        ),
    )


@tool
def get_project_detail(query: str) -> str:
    """Delegate project questions to the semantic project evidence subagent."""
    response = build_project_agent().invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=tracing_config("interview-project-agent", "project-agent"),
    )
    return str(response["messages"][-1].content)