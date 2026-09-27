from functools import lru_cache

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from src.backend.config import settings
from src.backend.graph.projects import get_project_detail
from src.backend.graph.resume import get_general_info


MAIN_AGENT_PROMPT = (
    "You are the main interview agent. "
    "Choose the resume tool for general background questions and the project tool for project details. "
    "If the user asks for a random project, call the resume tool first. Only call the project tool when "
    "looking for detail or when the project is not in the resume. "
    "Answer concisely in first person using only returned evidence. "
    "If no evidence supports the question, answer exactly: I am sorry. I am not sure. Please ask another question."
)


@lru_cache(maxsize=1)
def _build_main_agent():
    """Build the supervisor that owns tool selection and answer composition."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[get_general_info, get_project_detail],
        system_prompt=MAIN_AGENT_PROMPT,
    )