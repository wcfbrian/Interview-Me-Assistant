"""Download public, non-fork repository READMEs into the project artifact index."""

import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

from interview_me.backend.config import settings


def _get_json(url: str, token: str = "") -> object:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "interview-me"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def sync(username: str, output: Path, token: str = "") -> int:
    print(f"[1/3] Fetching public repositories for @{username}...")
    repositories = _get_json(f"https://api.github.com/users/{username}/repos?per_page=100", token)
    if not isinstance(repositories, list):
        raise ValueError("GitHub repositories response was not a list")
    print(f"[2/3] Processing {len(repositories)} repositories...")
    projects = []
    for repository in repositories:
        if repository.get("fork") or repository.get("archived"):
            print(f"[skip] {repository.get('name', 'unknown')}: fork or archived")
            continue
        try:
            readme = _get_json(repository["contents_url"].split("{", 1)[0] + "/README.md", token)
            content = base64.b64decode(readme["content"]).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"[skip] {repository.get('name', 'unknown')}: README unavailable ({exc})")
            continue
        words = content.split()
        projects.append({
            "name": repository["name"],
            "summary": " ".join(words[:500]),
            "readme": content,
            "source_url": repository["html_url"],
        })
        print(f"[done] project: {repository['name']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(projects, indent=2), encoding="utf-8")
    print(f"[3/3] Completed: indexed {len(projects)} projects to {output}")
    return len(projects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", nargs="?", default=settings.github_username)
    parser.add_argument("--output", type=Path, default=settings.project_index_path)
    parser.add_argument("--token", default="")
    args = parser.parse_args()
    if not args.username:
        parser.error("GITHUB_USERNAME is not set; provide a username or add it to .env")
    sync(args.username, args.output, args.token)