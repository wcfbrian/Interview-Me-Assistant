from functools import lru_cache

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from interview_me.backend.agents.projects import get_project_detail
from interview_me.backend.agents.resume import get_general_info
from interview_me.backend.config import settings


@lru_cache(maxsize=1)
def _build_main_agent():
    """Build the supervisor that owns tool selection and answer composition."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[get_general_info, get_project_detail],
        system_prompt=(
            "You are the main interview agent."
            "Choose the resume tool for general background questions and the project tool for project details."
            "If user ask for random project, call the resume tool first. Only call the project tool if looking for detail or project not in resume."
            "Use only evidence returned by those tools. Merge evidence internally and compose a concise first-person answer in natural language. "
            "If no evidence supports the question, "
            "answer exactly: I am sorry. I am not sure. Please ask another question."
        ),
    )
