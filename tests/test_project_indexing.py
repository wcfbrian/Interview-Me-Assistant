import base64
from dataclasses import replace
import json

from src.backend.config import settings
from src.backend.ingest.ingest_project import (
    index_projects,
    pipeline,
    split_project_readme,
)


def test_split_project_readme_preserves_markdown_sections_and_context():
    chunks = split_project_readme({
        "project_id": "github-1",
        "name": "demo",
        "source_url": "https://github.com/example/demo",
        "readme": "# Demo\n\nIntro.\n\n## Retrieval\n\nUses BM25 and vectors.",
    })

    assert len(chunks) == 2
    assert chunks[0].page_content.startswith("# Demo")
    assert chunks[1].page_content.startswith("## Retrieval")
    assert chunks[1].metadata["h1"] == "Demo"
    assert chunks[1].metadata["h2"] == "Retrieval"
    assert chunks[1].metadata["project_id"] == "github-1"
    assert chunks[1].metadata["source_url"] == "https://github.com/example/demo"
    assert chunks[0].metadata["chunk_id"] == "github-1:chunk:0"
    assert chunks[1].metadata["chunk_id"] == "github-1:chunk:1"


def test_split_project_readme_limits_chunk_size_and_skips_empty_content():
    chunks = split_project_readme({
        "project_id": "github-1",
        "readme": "# Long section\n\n" + ("retrieval " * 500),
    })

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
    assert split_project_readme({"project_id": "empty", "readme": ""}) == []


def test_index_projects_embeds_readme_chunks_with_project_metadata(monkeypatch, tmp_path):
    project_path = tmp_path / "projects.json"
    monkeypatch.setattr(
        "src.backend.ingest.ingest_project.settings",
        replace(settings, chroma_dir=tmp_path / "chroma"),
    )
    project_path.write_text(
        json.dumps([{
            "project_id": "github-1",
            "name": "demo",
            "readme": "# Demo\n\nA retrieval project",
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
    assert vector_store.documents[0].page_content.startswith("# Demo")
    assert "A retrieval project" in vector_store.documents[0].page_content
    assert vector_store.documents[0].metadata["project_id"] == "github-1"
    assert vector_store.documents[0].metadata["name"] == "demo"
    assert vector_store.ids == ["github-1:chunk:0"]


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
    monkeypatch.setattr("src.backend.ingest.ingest_project.index_projects", lambda _: 1)

    pipeline("candidate", project_path)

    assert json.loads(project_path.read_text(encoding="utf-8")) == [{
        "project_id": "github-123",
        "name": "demo",
        "readme": "# Demo",
        "source_url": "https://github.com/example/demo",
    }]