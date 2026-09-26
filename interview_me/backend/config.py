"""Environment-backed configuration for local and container runs."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    main_model: str = os.getenv("MAIN_MODEL", "deepseek/deepseek-v4.1-flash")
    guard_model: str = os.getenv("GUARD_MODEL", "z-ai/glm-5.3-flash")
    evaluation_model: str = os.getenv("EVALUATION_MODEL", "typesafe/jev-1.13")
    github_username: str = os.getenv("GITHUB_USERNAME", "")
    data_dir: Path = Path(os.getenv("DATA_DIR", "interview_me/backend/data"))
    max_project_searches: int = int(os.getenv("MAX_PROJECT_SEARCHES", "2"))
    max_readme_fetches: int = int(os.getenv("MAX_README_FETCHES", "2"))

    @property
    def resume_path(self) -> Path:
        return Path(os.getenv("RESUME_PATH", str(self.data_dir / "resume.pdf")))

    @property
    def profile_path(self) -> Path:
        return self.data_dir / "profile.json"

    @property
    def project_index_path(self) -> Path:
        return self.data_dir / "projects.json"


settings = Settings()