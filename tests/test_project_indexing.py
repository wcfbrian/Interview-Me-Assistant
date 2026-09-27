import base64
import json

import pytest

from src.backend.ingest.ingest_project import ProjectSummary, index_projects, pipeline


# Verifies summaries below the limit are accepted by constructing a 299-word project summary.
def test_project_summary_allows_fewer_than_300_words():
    assert len(ProjectSummary(summary="word " * 299).summary.split()) == 299


# Verifies the summary limit rejects 300 words by asserting the Pydantic validation error.
def test_project_summary_rejects_300_words():
    with pytest.raises(ValueError, match="fewer than 300 words"):
        ProjectSummary(summary="word " * 300)


# Verifies indexing writes summary content and project identity into Chroma by capturing added documents in a fake store.
def test_index_projects_embeds_summaries_with_project_metadata(monkeypatch, tmp_path):
    project_path = tmp_path / "projects.json"
    project_path.write_text(
        json.dumps([{
            "project_id": "github-1",
            "name": "demo",
            "summary": "A retrieval project",
            "readme": "Full approved README",
            "source_url": "https://github.com/example/demo",
        }]),
        encoding="utf-8",
    )

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

    vector_store = FakeVectorStore()
    monkeypatch.setattr(
        "src.backend.ingest.ingest_project.reset_project_vector_store",
        lambda: vector_store,
    )

    assert index_projects(project_path) == 1
    assert vector_store.documents[0].page_content == "A retrieval project"
    assert vector_store.documents[0].metadata["project_id"] == "github-1"
    assert vector_store.documents[0].metadata["name"] == "demo"
    assert vector_store.ids == ["github-1"]


# Verifies GitHub ingestion persists a stable project ID by faking repository and README API responses.
def test_project_pipeline_persists_github_project_id(monkeypatch, tmp_path):
    project_path = tmp_path / "projects.json"

    def fake_get_json(url, token=""):
        if url.endswith("/repos"):
            return [{
                "id": 123,
                "name": "demo",
                "fork": False,
                "archived": False,
                "contents_url": "https://api.github.com/repos/example/demo/contents{+path}",
                "html_url": "https://github.com/example/demo",
            }]
        return {"content": base64.b64encode(b"# Demo").decode("ascii")}

    monkeypatch.setattr("src.backend.ingest.ingest_project._get_json", fake_get_json)
    monkeypatch.setattr("src.backend.ingest.ingest_project.summarize_readme", lambda _: "Demo")
    monkeypatch.setattr("src.backend.ingest.ingest_project.index_projects", lambda _: 1)

    pipeline("candidate", project_path)

    assert json.loads(project_path.read_text(encoding="utf-8")) == [{
        "project_id": "github-123",
        "name": "demo",
        "summary": "Demo",
        "readme": "# Demo",
        "source_url": "https://github.com/example/demo",
    }]