"""Build validated local resume and GitHub evidence artifacts."""

import asyncio
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.github import sync_public_readmes
from backend.ingestion.profile import extract_candidate_profile
from backend.ingestion.resume import extract_resume_text
from backend.ingestion.summaries import summarize_readmes
from backend.knowledge import LocalEvidenceStore
from backend.retrieval.chroma import ChromaVectorStore


async def ingest() -> None:
    settings = get_settings()
    root = Path(settings.data_dir)
    profile = await extract_candidate_profile(extract_resume_text(settings.resume_path), settings)
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    (extracted / "candidate.json").write_text(profile.model_dump_json(indent=2))

    repositories = sync_public_readmes(
        settings.github_username, root / "github" / "readmes", settings.github_token
    )
    await summarize_readmes(repositories, settings, root / "github" / "summaries.json")
    store = LocalEvidenceStore(str(root))
    ChromaVectorStore(settings).upsert(store.project_documents)


if __name__ == "__main__":
    asyncio.run(ingest())