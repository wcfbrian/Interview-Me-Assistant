from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model

from src.backend.config import settings
from src.backend.graph.projects import get_project_detail
from src.backend.graph.resume import (
    ProfileContext,
    get_education,
    get_highlighted_project,
    get_name,
    get_self_introduction,
    get_skills,
    get_working_background,
)


MAIN_AGENT_PROMPT = (
    "You are the main interview agent. "
    "Call the relevant resume field tool for general background questions and the project retrieval tool "
    "for project details. If the user asks for a random project, call get_highlighted_project first. "
    "Only call get_project_detail when looking for detail or when the project is not in the resume. "
    "Answer concisely in first person tone using only returned evidence. "
    "If no evidence supports the question, answer exactly: I am sorry. I am not sure. Please ask another question."
)


@lru_cache(maxsize=1)
def build_main_agent():
    """Build the interview agent that owns retrieval and answer composition."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    resume_tools = [
        get_name,
        get_self_introduction,
        get_working_background,
        get_education,
        get_skills,
        get_highlighted_project,
    ]
    return create_agent(
        model=model,
        tools=[*resume_tools, get_project_detail],
        middleware=[
            ToolCallLimitMiddleware(
                tool_name=tool.name,
                run_limit=1,
                exit_behavior="continue",
            )
            for tool in resume_tools
        ],
        context_schema=ProfileContext,
        system_prompt=MAIN_AGENT_PROMPT,
    )