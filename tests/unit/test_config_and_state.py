from backend.schemas import AgentState, Evidence, Source


def test_subagents_do_not_have_a_memory_field() -> None:
    assert "main_history" in AgentState.__dataclass_fields__
    assert "subagent_history" not in AgentState.__dataclass_fields__


def test_evidence_keeps_source_identity() -> None:
    evidence = Evidence(
        text="Built a hybrid retrieval system.",
        source=Source("github", "retrieval-demo", "README.md"),
        score=0.91,
    )

    assert evidence.source.source_type == "github"
    assert evidence.source.name == "retrieval-demo"