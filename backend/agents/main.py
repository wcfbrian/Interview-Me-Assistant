"""Memory-aware main answer agent."""

from langchain.agents import create_agent

from backend.agents.runtime import create_chat_model
from backend.config import Settings
from backend.schemas import Evidence


def create_answer_generator(settings: Settings):
    model = create_chat_model(settings, settings.main_model)
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You answer as the candidate in a job interview. Use only the supplied "
            "evidence. Do not infer missing facts. Answer in concise first-person "
            "prose. If evidence does not support the answer, say exactly: "
            "No information found."
        ),
    )

    async def generate(
        query: str, evidence: list[Evidence], history: list[dict[str, str]]
    ) -> str:
        evidence_text = "\n".join(
            f"[{item.source.source_type}:{item.source.name}] {item.text}" for item in evidence
        )
        messages = [*history, {"role": "user", "content": query}]
        messages.append(
            {
                "role": "user",
                "content": f"Evidence to ground the answer:\n{evidence_text}",
            }
        )
        result = await agent.ainvoke({"messages": messages})
        return result["messages"][-1].content

    return generate