"""Download public, non-fork repository READMEs into the project index."""

import base64
import json
import pickle
import re
import shutil
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

from ..config import settings


def split_project_readme(project: dict[str, str]) -> list[Document]:
    """Split a project README into heading-aware, size-limited passages."""
    readme = str(project.get("readme", "")).strip()
    if not readme:
        return []

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
        ],
        strip_headers=True,
    )
    section_documents = markdown_splitter.split_text(readme)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks: list[Document] = []
    project_id = str(project["project_id"])
    for section in section_documents:
        for chunk in text_splitter.split_documents([section]):
            if not chunk.page_content.strip():
                continue
            chunk_index = len(chunks)
            chunk_id = f"{project_id}:chunk:{chunk_index}"
            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "project_id": project_id,
                        "name": str(project.get("name", "unknown")),
                        "source_url": str(project.get("source_url", "")),
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        **{
                            key: str(value)
                            for key, value in chunk.metadata.items()
                            if key != "start_index"
                        },
                    },
                )
            )
    return chunks


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


@lru_cache(maxsize=1)
def get_project_vector_store() -> Chroma:
    return Chroma(
        collection_name="projects",
        embedding_function=_project_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def reset_project_vector_store() -> Chroma:
    """Remove the dedicated project Chroma database and create a fresh one."""
    get_project_bm25.cache_clear()
    get_project_vector_store.cache_clear()
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    return get_project_vector_store()


def _project_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_project_bm25(projects: list[dict[str, str]]) -> dict | None:
    documents = [
        (project, chunk)
        for project in projects
        for chunk in split_project_readme(project)
    ]
    if not documents:
        return None
    corpus = [
        f"{project.get('name', '')} {chunk.page_content}"
        for project, chunk in documents
    ]
    tokenized_corpus = [_project_tokens(document) for document in corpus]
    return {
        "bm25": BM25Okapi(tokenized_corpus),
        "chunk_ids": [chunk.metadata["chunk_id"] for _, chunk in documents],
    }


@lru_cache(maxsize=1)
def get_project_bm25() -> dict | None:
    """Return the cached BM25 index for projects, if it exists."""
    index_path = settings.project_bm25_index_path
    if not index_path.exists():
        return None
    with index_path.open("rb") as handle:
        return pickle.load(handle)


def index_projects(projects_path: Path) -> int:
    """Embed README chunks and persist them in the Chroma project index."""
    projects = load_projects(projects_path)
    project_ids = [str(project.get("project_id", "")) for project in projects]
    if not all(project_ids) or len(set(project_ids)) != len(project_ids):
        raise ValueError(
            "Each project must have a unique project_id; run project ingestion with "
            "--regenerate-json."
        )
    vector_store = reset_project_vector_store()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    with settings.project_bm25_index_path.open("wb") as handle:
        pickle.dump(_build_project_bm25(projects), handle)
    if not projects:
        return 0
    documents = [
        chunk
        for project in projects
        for chunk in split_project_readme(project)
    ]
    if not documents:
        return 0
    vector_store.add_documents(
        documents,
        ids=[document.metadata["chunk_id"] for document in documents],
    )
    return len(documents)


def _get_json(url: str, token: str = "") -> object:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "interview-me"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def pipeline(username: str, output: Path, regenerate_json: bool = False, token: str = "") -> None:
    """Fetch public GitHub repositories and index their README chunks."""
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
                "readme": content,
                "source_url": repository["html_url"],
            })
            print(f"[done] project: {repository['name']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(projects, indent=2), encoding="utf-8")
        print(f"Completed: Writing project READMEs to {output}...")

    print("Starting indexing projects...")
    index_projects(output)
    print(f"Completed: indexed projects to {output}")