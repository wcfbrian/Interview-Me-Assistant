from src.backend.artifacts import load_profile, load_projects


# Verifies JSON artifact loaders preserve profile fields and project records using temporary JSON files.
def test_artifacts_preserve_only_json_profile_and_projects(tmp_path):
    profile_path = tmp_path / "profile.json"
    projects_path = tmp_path / "projects.json"
    profile_path.write_text('{"skills": "Python"}', encoding="utf-8")
    projects_path.write_text('[{"name": "demo", "readme": "RAG"}]', encoding="utf-8")

    assert load_profile(profile_path) == {"skills": "Python"}
    assert load_projects(projects_path)[0]["name"] == "demo"