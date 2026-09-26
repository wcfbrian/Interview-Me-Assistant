"""Environment-backed runtime configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration shared by the backend and Chainlit entrypoint."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = Field(min_length=1)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    main_model: str = "deepseek/deepseek-v4.1-flash"
    guard_model: str = "z-ai/glm-5.3-flash"
    evaluation_model: str = "typesafe/jev-1.13"
    embedding_model: str = "openai/text-embedding-3-small"
    github_username: str = Field(min_length=1)
    github_token: str | None = None
    resume_path: str = "data/resume.pdf"
    data_dir: str = "data"
    chroma_persist_directory: str = "data/chroma"
    chainlit_auth_username: str | None = None
    chainlit_auth_password: str | None = None
    request_timeout_seconds: float = 30.0
    max_project_searches: int = 2
    max_readme_fetches: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()