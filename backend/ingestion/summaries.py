"""GitHub README summarization for the searchable project index."""

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from backend.agents.runtime import create_chat_model
from backend.config import Settings
from backend.ingestion.github import RepositoryReadme

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Summarize this project README in fewer than 500 words. Preserve project
name, purpose, technologies, implementation details, and measurable outcomes only
when stated. Do not invent missing details.""",
        ),
        ("human", "Project: {name}\nREADME:\n{text}"),
    ]
)


async def summarize_readmes(
    repositories: list[RepositoryReadme], settings: Settings, output: str | Path
) -> None:
    model = create_chat_model(settings, settings.main_model)
    chain = _PROMPT | model
    summaries = []
    for repository in repositories:
        text = repository.readme_path.read_text(errors="replace")
        response = await chain.ainvoke({"name": repository.name, "text": text})
        summaries.append(
            {"name": repository.name, "summary": response.content, "path": str(repository.readme_path)}
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summaries, indent=2))