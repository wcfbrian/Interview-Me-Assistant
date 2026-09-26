"""Incremental GitHub README synchronization."""

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class RepositoryReadme:
    name: str
    html_url: str
    readme_path: Path
    sha: str


def sync_public_readmes(
    username: str,
    destination: str | Path,
    token: str | None = None,
) -> list[RepositoryReadme]:
    """Download changed non-fork public READMEs and preserve a manifest cache."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        f"https://api.github.com/users/{username}/repos",
        headers=headers,
        params={"type": "public", "per_page": 100, "sort": "updated"},
        timeout=30,
    )
    response.raise_for_status()
    repositories = []
    for repository in response.json():
        if repository.get("fork"):
            continue
        readme_response = requests.get(
            f"https://api.github.com/repos/{username}/{repository['name']}/readme",
            headers=headers,
            timeout=30,
        )
        if readme_response.status_code == 404:
            continue
        readme_response.raise_for_status()
        payload = readme_response.json()
        name = repository["name"]
        path = root / f"{name}.md"
        if previous.get(name, {}).get("sha") != payload["sha"]:
            content = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
            path.write_text(content)
        repositories.append(
            RepositoryReadme(name, repository["html_url"], path, payload["sha"])
        )

    manifest = {
        item.name: {"html_url": item.html_url, "path": str(item.readme_path), "sha": item.sha}
        for item in repositories
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return repositories