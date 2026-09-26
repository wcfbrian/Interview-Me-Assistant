"""LLM-assisted, privacy-preserving resume profile extraction."""

from langchain_core.prompts import ChatPromptTemplate

from backend.agents.runtime import create_chat_model
from backend.config import Settings
from backend.schemas import CandidateProfile

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract only professional candidate facts from the resume into the
provided schema. Exclude email, phone, address, IDs, compensation, protected traits,
and other sensitive personal details. Do not infer facts not present in the text.""",
        ),
        ("human", "Resume text:\n{text}"),
    ]
)


async def extract_candidate_profile(text: str, settings: Settings) -> CandidateProfile:
    model = create_chat_model(settings, settings.main_model)
    structured = model.with_structured_output(CandidateProfile)
    return await (_PROMPT | structured).ainvoke({"text": text})