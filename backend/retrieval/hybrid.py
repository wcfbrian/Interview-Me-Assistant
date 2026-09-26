"""Small deterministic hybrid ranker used by resume and project retrieval."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SearchDocument:
    name: str
    summary: str
    locator: str

    @property
    def text(self) -> str:
        return f"{self.name} {self.summary}"


@dataclass(frozen=True)
class SearchResult:
    document: SearchDocument
    score: float


class HybridRetriever:
    """Combine normalized lexical and optional vector scores.

    The vector score provider is injected by the Chroma adapter so this class
    stays deterministic and testable without an embedding API.
    """

    def __init__(self, documents: list[SearchDocument]):
        self.documents = documents
        self._tokenized = [_TOKEN_RE.findall(document.text.casefold()) for document in documents]
        self._bm25 = BM25Okapi(self._tokenized) if documents else None

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        vector_scores: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        if not self.documents:
            return []
        lexical = self._bm25.get_scores(_TOKEN_RE.findall(query.casefold()))
        max_lexical = max(lexical) or 1.0
        vectors = vector_scores or {}
        scored = []
        for index, document in enumerate(self.documents):
            lexical_score = float(lexical[index]) / max_lexical
            vector_score = max(0.0, min(1.0, vectors.get(document.locator, 0.0)))
            score = 0.6 * lexical_score + 0.4 * vector_score
            scored.append(SearchResult(document, score))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:limit]