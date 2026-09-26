"""Stateless evidence subagents used by the main agent."""

from backend.knowledge import LocalEvidenceStore
from backend.schemas import Evidence


async def search_resume(query: str, store: LocalEvidenceStore) -> list[Evidence]:
    """Search resume evidence without receiving or retaining chat history."""

    return [item for item in await store.search(query) if item.source.source_type == "resume"]


async def search_projects(query: str, store: LocalEvidenceStore) -> list[Evidence]:
    """Search project evidence without receiving or retaining chat history."""

    return [item for item in await store.search(query) if item.source.source_type == "github"]