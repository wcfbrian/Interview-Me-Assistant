import json
from functools import lru_cache
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool

from src.backend.artifacts import load_profile
from src.backend.config import settings, tracing_config


class ProfileContext(TypedDict):
    name: str
    self_introduction: str
    working_background: str
    education: str
    skills: str
    highlighted_project: str


def _profile_field_tool(field: str, runtime: ToolRuntime[ProfileContext]) -> str:
    return json.dumps({"source": f"resume:{field}", "text": runtime.context[field]})


@tool
def get_name(runtime: ToolRuntime[ProfileContext]) -> str:
    """Get the approved candidate name."""
    return _profile_field_tool("name", runtime)


@tool
def get_self_introduction(runtime: ToolRuntime[ProfileContext]) -> str:
    """Get the approved self-introduction."""
    return _profile_field_tool("self_introduction", runtime)


@tool
def get_working_background(runtime: ToolRuntime[ProfileContext]) -> str:
    """Get the approved working background."""
    return _profile_field_tool("working_background", runtime)


@tool
def get_education(runtime: ToolRuntime[ProfileContext]) -> str:
    """Get the approved education history."""
    return _profile_field_tool("education", runtime)


@tool
def get_skills(runtime: ToolRuntime[ProfileContext]) -> str:
    """Get the approved skills."""
    return _profile_field_tool("skills", runtime)


@tool
def get_highlighted_project(runtime: ToolRuntime[ProfileContext]) -> str:
    """Get the approved highlighted project experience."""
    return _profile_field_tool("highlighted_project", runtime)


@lru_cache(maxsize=1)
def build_resume_agent():
    """Build the resume subagent with typed profile field tools."""
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    tools = [
        get_name,
        get_self_introduction,
        get_working_background,
        get_education,
        get_skills,
        get_highlighted_project,
    ]
    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            ToolCallLimitMiddleware(
                tool_name=tool.name,
                run_limit=1,
                exit_behavior="continue",
            )
            for tool in tools
        ],
        context_schema=ProfileContext,
        system_prompt=(
            "You are the resume evidence subagent. Always call the field tool or "
            "tools relevant to the user's query.  Use only evidence returned by "
            "those tools. Return related content only. Do not invent, summarize, or add facts."
        ),
    )


@tool
def get_general_info(query: str) -> str:
    """Delegate general interview questions to the resume evidence subagent."""
    response = build_resume_agent().invoke(
        {"messages": [{"role": "user", "content": query}]},
        context=load_profile(settings.profile_path),
        config=tracing_config("interview-resume-agent", "resume-agent"),
    )
    return str(response["messages"][-1].content)