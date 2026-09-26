from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage

from interview_me.backend.agents.projects import get_project_detail
from interview_me.backend.agents.resume import get_general_info
from interview_me.backend.config import settings


def build_main_agent():
    """Build the supervisor that owns tool selection and answer composition."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[get_general_info, get_project_detail],
        system_prompt=(
            "You are the main interview agent. Choose the resume tool for general "
            "background questions and the project tool for project details. Use only "
            "evidence returned by those tools. Merge evidence internally and compose "
            "a concise first-person answer. If no evidence supports the question, "
            "answer exactly: No information found."
        ),
    )


def run_main_agent(messages: list[BaseMessage]) -> AIMessage:
    """Run the supervisor tool loop and return only its final answer message."""
    response = build_main_agent().invoke({"messages": messages})
    return next(
        message
        for message in reversed(response["messages"])
        if isinstance(message, AIMessage)
    )
