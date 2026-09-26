import pytest

from backend.agents.subagents import search_projects, search_resume
from backend.knowledge import LocalEvidenceStore


@pytest.mark.asyncio
async def test_subagents_filter_evidence_without_history(tmp_path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (extracted / "candidate.json").write_text('{"skills": ["Python"]}')
    store = LocalEvidenceStore(str(tmp_path))

    resume_results = await search_resume("Python", store)
    project_results = await search_projects("Python", store)

    assert resume_results and resume_results[0].source.source_type == "resume"
    assert project_results == []