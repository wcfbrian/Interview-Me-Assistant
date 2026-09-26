import json
from functools import lru_cache

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from interview_me.backend.artifacts import load_profile
from interview_me.backend.config import settings, tracing_config


@tool("retrieve_resume_evidence")
def _retrieve_resume_evidence(query: str) -> str:
    """Return the approved resume facts for the resume subagent to select from."""
    profile = load_profile(settings.profile_path)
    matches = [
        {"source": f"resume:{field}", "text": profile[field]}
        for field in profile
    ]
    return json.dumps({"query": query, "evidence": matches})


@lru_cache(maxsize=1)
def build_resume_agent():
    """Build the resume subagent that owns resume evidence retrieval."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[_retrieve_resume_evidence],
        system_prompt=(
            "You are the resume evidence subagent. Always call the retrieval tool. "
            "Use the user's query to select relevant evidence from the tool result. "
            "Return only JSON with status and evidence fields. Copy evidence text "
            "exactly. Do not invent, summarize, or add facts."
        ),
    )


@tool
def get_general_info(query: str) -> str:
    """Delegate general interview questions to the resume evidence subagent."""
    response = build_resume_agent().invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=tracing_config("interview-resume-agent", "resume-agent"),
    )
    return str(response["messages"][-1].content)

