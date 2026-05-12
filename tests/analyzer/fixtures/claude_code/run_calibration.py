"""Run the analyzer against a skills directory and write per-skill JSON reports.

Outputs are written to _runs/<skills_dir>_<timestamp>/<skill_name>.json.

Usage
-----
    python tests/analyzer/fixtures/claude_code/run_calibration.py skills
    python tests/analyzer/fixtures/claude_code/run_calibration.py skills-real
    python tests/analyzer/fixtures/claude_code/run_calibration.py skills --only log-archiver
    python tests/analyzer/fixtures/claude_code/run_calibration.py skills --concurrency 3

Environment
-----------
    LIMENEX_ANALYZER_BASE_URL    default: https://openrouter.ai/api/v1
    LIMENEX_ANALYZER_MODEL       default: openai/gpt-4o-mini
    LIMENEX_ANALYZER_API_KEY     required
    LIMENEX_ANALYZER_TEMPERATURE default: 0.0

A .env file at the repository root is loaded automatically if python-dotenv
is installed (see the [dev] extra in pyproject.toml).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(HERE / ".env", override=False)
except ImportError:
    pass

try:
    from limenex.analyzer.harness import (
        AnalyzeSkillError,
        analyze_skill,
        discover_skills,
    )
    from limenex.analyzer.providers import OpenAICompatProvider
except ImportError as e:
    print(
        f"run_calibration: failed to import limenex.analyzer ({e}). "
        "Install with: pip install -e .[analyzer]",
        file=sys.stderr,
    )
    sys.exit(2)


RUNS_ROOT = HERE / "_runs"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0


def _load_provider() -> OpenAICompatProvider:
    api_key = os.environ.get("LIMENEX_ANALYZER_API_KEY")
    if not api_key:
        print(
            "run_calibration: LIMENEX_ANALYZER_API_KEY is not set.",
            file=sys.stderr,
        )
        sys.exit(2)
    base_url = os.environ.get("LIMENEX_ANALYZER_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LIMENEX_ANALYZER_MODEL", DEFAULT_MODEL)
    try:
        temperature = float(
            os.environ.get("LIMENEX_ANALYZER_TEMPERATURE", str(DEFAULT_TEMPERATURE))
        )
    except ValueError:
        print(
            "run_calibration: LIMENEX_ANALYZER_TEMPERATURE must be a float.",
            file=sys.stderr,
        )
        sys.exit(2)
    return OpenAICompatProvider(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def _serialise(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


async def _run_one(skill, provider, out_dir, semaphore):
    async with semaphore:
        try:
            exposure = await analyze_skill(skill, provider, framework="claude_code")
        except AnalyzeSkillError as e:
            return skill.name, None, str(e)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{skill.name}.json"
        out_path.write_text(
            json.dumps(exposure, default=_serialise, indent=2),
            encoding="utf-8",
        )
        return skill.name, exposure, None


async def _run_all(skills, provider, out_dir, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_run_one(s, provider, out_dir, semaphore) for s in skills]
    try:
        return await asyncio.gather(*tasks)
    finally:
        await provider.aclose()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "skills_dir",
        help="Directory under this fixtures folder to analyse (e.g. 'skills' or 'skills-real').",
    )
    p.add_argument("--only", action="append", help="Analyse only these skill name(s).")
    p.add_argument("--concurrency", type=int, default=3, help="Max concurrent analyses.")
    args = p.parse_args(argv)

    skills_root = (HERE / args.skills_dir).resolve()
    if not skills_root.is_dir():
        print(f"run_calibration: not a directory: {skills_root}", file=sys.stderr)
        return 2

    skills = discover_skills(skills_root)
    if args.only:
        wanted = set(args.only)
        skills = [s for s in skills if s.name in wanted]
        missing = wanted - {s.name for s in skills}
        if missing:
            print(f"run_calibration: unknown skills: {sorted(missing)}", file=sys.stderr)
            return 2
    if not skills:
        print("run_calibration: no skills discovered", file=sys.stderr)
        return 2

    provider = _load_provider()
    fp = provider.fingerprint()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_ROOT / f"{args.skills_dir}_{timestamp}"

    print(f"model={fp.model}  temperature={fp.temperature}")
    print(f"analysing {len(skills)} skill(s) into {out_dir}")

    results = asyncio.run(_run_all(skills, provider, out_dir, args.concurrency))

    failures = 0
    for name, exposure, err in results:
        if err is not None:
            failures += 1
            print(f"  {name:<28}  FAIL: {err}", file=sys.stderr)
        else:
            n = len(exposure.findings) if hasattr(exposure, "findings") else "?"
            print(f"  {name:<28}  ok ({n} findings)")

    print(f"done. outputs in {out_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())