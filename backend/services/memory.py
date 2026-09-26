"""Short-term main-agent conversation memory."""

from collections.abc import Awaitable, Callable

SummaryFunction = Callable[[list[dict[str, str]]], Awaitable[str]]


class MainAgentMemory:
    """Retain recent turns and summarize older context above a token budget."""

    def __init__(self, summarize: SummaryFunction, token_limit: int = 5000):
        self._summarize = summarize
        self._token_limit = token_limit
        self._messages: list[dict[str, str]] = []
        self._summary: str | None = None

    def messages(self) -> list[dict[str, str]]:
        prefix = [{"role": "system", "content": self._summary}] if self._summary else []
        return prefix + list(self._messages)

    async def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if self._token_count() <= self._token_limit:
            return
        older = self._messages[:-12]
        self._messages = self._messages[-12:]
        if older:
            self._summary = await self._summarize(older)

    def _token_count(self) -> int:
        content = " ".join(item["content"] for item in self._messages)
        return max(1, len(content) // 4)