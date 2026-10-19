from typing import TypedDict

from langchain.tools import ToolRuntime, tool


class ProfileContext(TypedDict):
    name: str
    self_introduction: str
    working_background: str
    education: str
    skills: str
    highlighted_project: str


def _profile_field_tool(field: str, runtime: ToolRuntime[ProfileContext]) -> str:
    label = field.replace("_", " ").title()
    return (
        f"## {label}\n"
        f"**Source:** `resume:{field}`\n\n"
        f"{runtime.context[field]}"
    )


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