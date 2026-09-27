"""Download public, non-fork repository READMEs into the project index."""

import base64
import json
import pickle
import re
import shutil
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, field_validator
from rank_bm25 import BM25Okapi

from ..config import settings

BM25_INDEX_FILENAME = "projects.bm25.pkl"


class ProjectSummary(BaseModel):
    summary: str = Field(..., description="A concise summary of the project. Limit to 300 words.")

    @field_validator("summary")
    @classmethod
    def word_limit(cls, value: str) -> str:
        value = value.strip()
        if len(value.split()) >= 300:
            raise ValueError("summary must contain fewer than 300 words")
        return value


@lru_cache(maxsize=1)
def _summary_agent():
    model = init_chat_model(settings.main_model, model_provider="openrouter")
    return model.with_structured_output(ProjectSummary)


def summarize_readme(readme: str) -> str:
    response = _summary_agent().invoke([
        {
            "role": "system",
            "content": (
                "Summarize a GitHub README for project retrieval. Include the "
                "project's purpose, important technologies, and notable capabilities."
                "Use only facts in the README. Keep the summary below 300 words."
                "Format:\nSummary: <summary text>"
            ),
        },
        {"role": "user", "content": readme},
    ])
    return response.summary


def load_projects(projects_path: Path) -> list[dict[str, str]]:
    with projects_path.open(encoding="utf-8") as handle:
        projects = json.load(handle)
    if not isinstance(projects, list):
        raise ValueError("projects.json must contain a JSON array")
    return [project for project in projects if isinstance(project, dict)]


def _project_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


def get_project_vector_store() -> Chroma:
    return Chroma(
        collection_name="projects",
        embedding_function=_project_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def reset_project_vector_store() -> Chroma:
    """Remove the dedicated project Chroma database and create a fresh one."""
    get_project_bm25.cache_clear()
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    return get_project_vector_store()


def _project_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_project_bm25(projects: list[dict[str, str]]) -> dict | None:
    if not projects:
        return None
    corpus = [
        " ".join(str(project.get(field, "")) for field in ("name", "readme"))
        for project in projects
    ]
    tokenized_corpus = [_project_tokens(document) for document in corpus]
    return {
        "bm25": BM25Okapi(tokenized_corpus),
        "project_ids": [str(project["project_id"]) for project in projects],
    }


@lru_cache(maxsize=1)
def get_project_bm25() -> dict | None:
    """Return the cached BM25 index for projects, if it exists."""
    index_path = settings.chroma_dir / BM25_INDEX_FILENAME
    if not index_path.exists():
        return None
    with index_path.open("rb") as handle:
        return pickle.load(handle)


def index_projects(projects_path: Path) -> int:
    """Embed project summaries and persist them in the Chroma project index."""
    projects = load_projects(projects_path)
    project_ids = [str(project.get("project_id", "")) for project in projects]
    if not all(project_ids) or len(set(project_ids)) != len(project_ids):
        raise ValueError(
            "Each project must have a unique project_id; run project ingestion with "
            "--regenerate-json."
        )
    vector_store = reset_project_vector_store()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    with (settings.chroma_dir / BM25_INDEX_FILENAME).open("wb") as handle:
        pickle.dump(_build_project_bm25(projects), handle)
    if not projects:
        return 0
    documents = [
        Document(
            page_content=str(project.get("summary", "")),
            metadata={
                "project_id": str(project["project_id"]),
                "name": str(project.get("name", "unknown")),
                "source_url": str(project.get("source_url", "")),
            },
        )
        for project in projects
    ]
    vector_store.add_documents(
        documents,
        ids=project_ids,
    )
    return len(documents)


def _get_json(url: str, token: str = "") -> object:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "interview-me"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def pipeline(username: str, output: Path, regenerate_json: bool = False, token: str = "") -> None:
    """Fetch, summarize, and embed public GitHub repositories for a given user."""
    if not output.exists() or regenerate_json:
        print(f"Fetching public repositories for @{username}...")
        repositories = _get_json(f"https://api.github.com/users/{username}/repos", token)
        if not isinstance(repositories, list):
            raise ValueError("GitHub repositories response was not a list")
        print(f"Processing {len(repositories)} repositories...")
        projects = []
        for repository in repositories:
            if repository.get("fork") or repository.get("archived"):
                print(f"[skip] {repository.get('name', 'unknown')}: fork or archived")
                continue
            try:
                readme = _get_json(repository["contents_url"].split("{", 1)[0] + "/README.md", token)
                content = base64.b64decode(readme["content"]).decode("utf-8", errors="replace")
            except Exception as exc:
                print(f"[skip] {repository.get('name', 'unknown')}: README unavailable ({exc})")
                continue
            projects.append({
                "project_id": f"github-{repository['id']}",
                "name": repository["name"],
                "summary": summarize_readme(content),
                "readme": content,
                "source_url": repository["html_url"],
            })
            print(f"[done] project: {repository['name']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(projects, indent=2), encoding="utf-8")
        print(f"Completed: Writing summarized project to {output}...")

    print("Starting indexing projects...")
    index_projects(output)
    print(f"Completed: indexed projects to {output}")