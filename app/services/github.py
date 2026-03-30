"""GitHub service — fetches PR diffs via the GitHub API."""

from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"


async def fetch_pr_diff(pr_ref: str) -> dict[str, Any]:
    """Fetch a PR diff from GitHub.

    Args:
        pr_ref: PR reference in format "owner/repo/pull/123" or full URL.

    Returns:
        Dict with keys: diff, title, author, language
    """
    # Normalize input — accept both "owner/repo/pull/123" and full URLs
    pr_ref = pr_ref.replace("https://github.com/", "")
    parts = pr_ref.strip("/").split("/")

    if len(parts) < 4 or parts[2] != "pull":
        raise ValueError(f"Invalid PR reference: {pr_ref}. Expected format: owner/repo/pull/123")

    owner, repo, _, pr_number = parts[0], parts[1], parts[2], parts[3]

    headers = {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    async with httpx.AsyncClient() as client:
        # Fetch PR metadata
        pr_url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
        pr_resp = await client.get(pr_url, headers=headers)
        pr_resp.raise_for_status()
        pr_data = pr_resp.json()

        # Fetch the diff
        diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
        diff_resp = await client.get(pr_url, headers=diff_headers)
        diff_resp.raise_for_status()

    logger.info(
        "github.pr_fetched",
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        diff_size=len(diff_resp.text),
    )

    return {
        "diff": diff_resp.text,
        "title": pr_data.get("title", ""),
        "author": pr_data.get("user", {}).get("login", ""),
        "language": None,  # Could detect from file extensions in the diff
    }
