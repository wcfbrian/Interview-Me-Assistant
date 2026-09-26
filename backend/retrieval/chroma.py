"""Chroma vector index backed by the configured OpenRouter embeddings model."""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.config import Settings
from backend.retrieval.hybrid import SearchDocument


class ChromaVectorStore:
    def __init__(self, settings: Settings, collection_name: str = "interview-projects"):
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        self._store = Chroma(
            collection_name=collection_name,
            persist_directory=settings.chroma_persist_directory,
            embedding_function=embeddings,
        )

    def upsert(self, documents: list[SearchDocument]) -> None:
        if not documents:
            return
        self._store.add_texts(
            [document.text for document in documents],
            metadatas=[
                {"name": document.name, "locator": document.locator}
                for document in documents
            ],
            ids=[document.locator for document in documents],
        )

    def similarity_scores(self, query: str, limit: int = 3) -> dict[str, float]:
        matches = self._store.similarity_search_with_score(query, k=limit)
        return {
            match.metadata["locator"]: 1.0 / (1.0 + float(distance))
            for match, distance in matches
        }