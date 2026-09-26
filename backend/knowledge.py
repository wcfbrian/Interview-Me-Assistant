"""Local evidence store built from validated ingestion artifacts."""

import json
from pathlib import Path

from backend.retrieval.chroma import ChromaVectorStore
from backend.retrieval.hybrid import HybridRetriever, SearchDocument
from backend.schemas import Evidence, Source


class LocalEvidenceStore:
    """Search cached resume and GitHub summaries without conversational memory."""

    def __init__(self, data_dir: str, vector_store: ChromaVectorStore | None = None):
        root = Path(data_dir)
        self.resume_documents = self._load_resume(root / "extracted" / "candidate.json")
        self.project_documents = self._load_projects(root / "github" / "summaries.json")
        self.resume_search = HybridRetriever(self.resume_documents)
        self.project_search = HybridRetriever(self.project_documents)
        self.vector_store = vector_store

    @staticmethod
    def _load_resume(path: Path) -> list[SearchDocument]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        documents = []
        for section, value in payload.items():
            if isinstance(value, list):
                text = "; ".join(str(item) for item in value)
            else:
                text = str(value)
            if text.strip():
                documents.append(SearchDocument(section, text, str(path)))
        return documents

    @staticmethod
    def _load_projects(path: Path) -> list[SearchDocument]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        return [
            SearchDocument(item["name"], item["summary"], item["path"])
            for item in payload
        ]

    async def search(self, query: str) -> list[Evidence]:
        vector_scores = self.vector_store.similarity_scores(query) if self.vector_store else None
        results = self.resume_search.search(query, limit=3, vector_scores=vector_scores)
        results += self.project_search.search(query, limit=3)
        return [
            Evidence(
                text=result.document.summary,
                source=Source(
                    "resume" if result.document in self.resume_documents else "github",
                    result.document.name,
                    result.document.locator,
                ),
                score=result.score,
            )
            for result in sorted(results, key=lambda item: item.score, reverse=True)[:3]
        ]