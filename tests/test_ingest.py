import json
from types import SimpleNamespace

from src.backend.ingest import ingest
from src.backend.ingest.ingest_resume import CandidateProfile


def _profile() -> CandidateProfile:
    return CandidateProfile(
        name="Candidate",
        self_introduction="I build software.",
        working_background="Engineering",
        education="Computer science",
        skills="Python",
        highlighted_project="A project",
    )


# Verifies ingestion creates a missing project artifact by substituting extraction and project pipeline dependencies.
def test_ingest_generates_missing_project_index(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    project_path = output_dir / "projects.json"
    monkeypatch.setattr("src.backend.ingest.extract_resume_text", lambda _: "resume")
    monkeypatch.setattr("src.backend.ingest.extract_profile", lambda _: _profile())
    monkeypatch.setattr(
        "src.backend.ingest.settings",
        SimpleNamespace(github_username="candidate"),
    )

    def fake_sync(username, output, regenerate_json=False, token=""):
        assert username == "candidate"
        output.write_text('[{"name": "demo"}]', encoding="utf-8")
        return 1

    monkeypatch.setattr("src.backend.ingest.pipeline", fake_sync)

    ingest(tmp_path / "resume.pdf", output_dir)

    assert json.loads(project_path.read_text()) == [{"name": "demo"}]


# Verifies regeneration replaces an existing project artifact by invoking ingestion with the regeneration flag.
def test_ingest_regenerates_existing_project_index(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    project_path = output_dir / "projects.json"
    project_path.write_text('[{"name": "old"}]', encoding="utf-8")
    monkeypatch.setattr("src.backend.ingest.extract_resume_text", lambda _: "resume")
    monkeypatch.setattr("src.backend.ingest.extract_profile", lambda _: _profile())
    monkeypatch.setattr(
        "src.backend.ingest.settings",
        SimpleNamespace(github_username="candidate"),
    )

    def fake_sync(username, output, regenerate_json=False, token=""):
        output.write_text('[{"name": "new"}]', encoding="utf-8")
        return 1

    monkeypatch.setattr("src.backend.ingest.pipeline", fake_sync)

    ingest(tmp_path / "resume.pdf", output_dir, regenerate_json=True)

    assert json.loads(project_path.read_text()) == [{"name": "new"}]