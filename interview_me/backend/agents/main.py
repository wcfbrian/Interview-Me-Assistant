from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from interview_me.backend.agents.projects import get_project_detail
from interview_me.backend.agents.resume import get_general_info
from interview_me.backend.config import settings, tracing_config


def _build_main_agent():
    """Build the supervisor that owns tool selection and answer composition."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return create_agent(
        model=model,
        tools=[get_general_info, get_project_detail],
        system_prompt=(
            "You are the main interview agent. Choose the resume tool for general "
            "background questions and the project tool for project details. If user ask for random project, call the resume tool first. Only call the project tool if looking for detail or project not in resume. "
            "Use only evidence returned by those tools. Merge evidence internally and compose "
            "a concise first-person answer. If no evidence supports the question, "
            "answer exactly: I am not sure. Please rephrase your question."
        ),
    )


def run_main_agent(messages: list[BaseMessage]) -> AIMessage:
    """Run the supervisor tool loop and return only its final answer message."""
    response = _build_main_agent().invoke(
        {"messages": messages},
        config=tracing_config("interview-main-agent", "main-agent"),
    )
    return next(
        message
        for message in reversed(response["messages"])
        if isinstance(message, AIMessage)
    )


async def stream_main_agent(messages: list[BaseMessage]):
    """Yield supervisor text chunks as they are generated."""
    stream = _build_main_agent().astream(
        {"messages": messages},
        config=tracing_config("interview-main-agent", "main-agent"),
        stream_mode="messages",
    )
    try:
        async for chunk, _ in stream:
            if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
                if chunk.content:
                    yield chunk.content
    finally:
        await stream.aclose()
