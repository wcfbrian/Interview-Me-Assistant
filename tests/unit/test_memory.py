import pytest

from backend.services.memory import MainAgentMemory


@pytest.mark.asyncio
async def test_memory_keeps_recent_six_turns_after_compaction() -> None:
    summaries = []

    async def summarize(messages):
        summaries.append(messages)
        return "Earlier interview context."

    memory = MainAgentMemory(summarize=summarize, token_limit=20)
    for index in range(8):
        await memory.add("user", f"Question {index} " + ("long " * 10))
        await memory.add("assistant", f"Answer {index}")

    messages = memory.messages()

    assert summaries
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "Answer 7"
    assert len(messages) <= 13