"""Production answer judge using the configured guard model."""

from langchain_core.prompts import ChatPromptTemplate

from backend.agents.runtime import create_chat_model
from backend.config import Settings
from backend.schemas import JudgeResult

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Judge whether an answer responds to the interviewer's question and
is fully supported by the supplied evidence. Reject invented details, unsupported
claims, and answers that do not address the question.""",
        ),
        (
            "human",
            "Question: {query}\nEvidence:\n{evidence}\nCandidate answer:\n{answer}",
        ),
    ]
)


async def judge_answer(
    query: str, answer: str, evidence: str, settings: Settings
) -> JudgeResult:
    model = create_chat_model(settings, settings.guard_model)
    return await (
        _PROMPT | model.with_structured_output(JudgeResult)
    ).ainvoke({"query": query, "answer": answer, "evidence": evidence})