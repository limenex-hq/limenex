"""Fetch real-world Claude Code Skill fixtures pinned in real-skills.json.

Reads real-skills.json, downloads each skill from its upstream GitHub
repository at the pinned commit SHA, extracts the skill subdirectory
into skills-real/<name>/, and verifies that SKILL.md is present.

Usage
-----
    python tests/analyzer/fixtures/claude_code/fetch_real_skills.py
    python tests/analyzer/fixtures/claude_code/fetch_real_skills.py --force
    python tests/analyzer/fixtures/claude_code/fetch_real_skills.py --only python_binance

Exit codes
----------
0  all skills fetched (or skipped as already present)
1  at least one skill failed
2  usage error (missing manifest, unknown --only target, missing httpx)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import httpx
except ImportError:
    print(
        "fetch_real_skills: httpx is required. "
        "Install with: pip install limenex[analyzer]",
        file=sys.stderr,
    )
    sys.exit(2)


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "real-skills.json"
TARGET_DIR = HERE / "skills-real"

_GITHUB_REPO_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+?)/?$")


@dataclass
class SkillEntry:
    name: str
    owner: str
    repo: str
    commit_sha: str
    path_in_repo: str


def _parse_manifest(path: Path) -> list[SkillEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[SkillEntry] = []
    for raw in data["skills"]:
        m = _GITHUB_REPO_RE.match(raw["repo_url"])
        if not m:
            raise ValueError(
                f"{raw['name']}: only github.com repos supported, got {raw['repo_url']!r}"
            )
        entries.append(
            SkillEntry(
                name=raw["name"],
                owner=m.group(1),
                repo=m.group(2),
                commit_sha=raw["commit_sha"],
                path_in_repo=raw["path_in_repo"].strip("/"),
            )
        )
    return entries


def _tarball_url(entry: SkillEntry) -> str:
    return f"https://github.com/{entry.owner}/{entry.repo}/archive/{entry.commit_sha}.tar.gz"


def _download_tarball(url: str, client: httpx.Client) -> bytes:
    r = client.get(url, follow_redirects=True, timeout=60.0)
    r.raise_for_status()
    return r.content


def _extract_skill(tarball_bytes: bytes, entry: SkillEntry, dest: Path) -> int:
    """Extract files under <archive_root>/<path_in_repo>/ into dest.

    Returns the number of files written. Raises ValueError if the archive
    contains no files under path_in_repo.
    """
    want_prefix = f"{entry.path_in_repo}/"
    written = 0
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError(f"{entry.name}: empty tarball")
        # The archive root is like "repo-<sha>/..."; identify it from the first member.
        archive_root = members[0].name.split("/", 1)[0]
        prefix = f"{archive_root}/{want_prefix}"
        for m in members:
            if not m.isfile():
                continue
            if not m.name.startswith(prefix):
                continue
            rel = m.name[len(prefix) :]
            if not rel:
                continue
            # Defensive: refuse absolute paths or traversal in archive.
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError(f"{entry.name}: unsafe path in archive: {m.name!r}")
            out_path = dest / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            out_path.write_bytes(src.read())
            written += 1
    if written == 0:
        raise ValueError(
            f"{entry.name}: no files found under {want_prefix!r} in archive "
            f"(bad path_in_repo, or SHA predates the path)"
        )
    return written


def _verify(dest: Path, entry: SkillEntry) -> None:
    if not (dest / "SKILL.md").is_file():
        raise ValueError(
            f"{entry.name}: extraction did not produce a SKILL.md at skill root"
        )


def _fetch_one(entry: SkillEntry, client: httpx.Client, force: bool) -> str:
    dest = TARGET_DIR / entry.name
    if dest.exists() and (dest / "SKILL.md").is_file() and not force:
        return "skip"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tarball = _download_tarball(_tarball_url(entry), client)
    n = _extract_skill(tarball, entry, dest)
    _verify(dest, entry)
    return f"ok ({n} files)"


def run(entries: Iterable[SkillEntry], force: bool) -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client() as client:
        for entry in entries:
            try:
                status = _fetch_one(entry, client, force)
                print(f"  {entry.name:<28}  {status}")
            except Exception as e:
                failures += 1
                print(f"  {entry.name:<28}  FAIL: {e}", file=sys.stderr)
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch skills even if already present.",
    )
    p.add_argument(
        "--only",
        metavar="NAME",
        action="append",
        help="Fetch only the named skill(s); may be passed multiple times.",
    )
    args = p.parse_args(argv)

    if not MANIFEST_PATH.is_file():
        print(f"manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    all_entries = _parse_manifest(MANIFEST_PATH)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in all_entries if e.name in wanted]
        missing = wanted - {e.name for e in entries}
        if missing:
            print(f"unknown skill name(s): {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        entries = all_entries

    print(f"fetching {len(entries)} skill(s) into {TARGET_DIR}")
    return run(entries, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
