"""Environment-backed configuration for local and container runs."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROFILE_FILENAME = "profile.json"
PROJECTS_FILENAME = "projects.json"
PROJECT_BM25_INDEX_FILENAME = "projects.bm25.pkl"


@dataclass(frozen=True)
class Settings:
    main_model: str = os.getenv("MAIN_MODEL", "deepseek/deepseek-v4.1-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    langsmith_tracing: bool = os.getenv(
        "LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")
    ).lower() == "true"
    langsmith_project: str = os.getenv(
        "LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "interview-me")
    )
    guard_model: str = os.getenv("GUARD_MODEL", "~typesafe/jev-latest")
    typesafe_base_url: str = os.getenv(
        "TYPESAFE_BASE_URL", "https://openrouter.ai/api"
    )
    evaluation_model: str = os.getenv("EVALUATION_MODEL", "typesafe/jev-1.13")
    github_username: str = os.getenv("GITHUB_USERNAME", "")
    data_dir: Path = Path(os.getenv("DATA_DIR", "src/backend/data"))
    chroma_dir: Path = Path(
        os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            os.getenv("CHROMA_DIR", "src/backend/data/chroma"),
        )
    )
    max_project_searches: int = int(os.getenv("MAX_PROJECT_SEARCHES", "2"))
    max_readme_fetches: int = int(os.getenv("MAX_README_FETCHES", "2"))
    profile_filename: str = os.getenv("PROFILE_FILENAME", PROFILE_FILENAME)
    projects_filename: str = os.getenv("PROJECTS_FILENAME", PROJECTS_FILENAME)
    project_bm25_index_filename: str = os.getenv(
        "PROJECT_BM25_INDEX_FILENAME", PROJECT_BM25_INDEX_FILENAME
    )

    @property
    def resume_path(self) -> Path:
        return Path(os.getenv("RESUME_PATH", str(self.data_dir / "resume.pdf")))

    @property
    def profile_path(self) -> Path:
        return self.data_dir / self.profile_filename

    @property
    def project_path(self) -> Path:
        return self.data_dir / self.projects_filename

    @property
    def project_bm25_index_path(self) -> Path:
        return self.chroma_dir / self.project_bm25_index_filename


settings = Settings()


def tracing_config(run_name: str, component: str) -> dict:
    """Return consistent LangSmith metadata for an agent invocation."""
    return {
        "run_name": run_name,
        "tags": ["interview-me", component],
        "metadata": {
            "component": component,
            "langsmith_project": settings.langsmith_project,
        },
    }