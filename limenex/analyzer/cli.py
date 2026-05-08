"""Limenex analyzer CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="limenex-analyze",
        description=(
            "Consultative LLM-based static analyser for Claude Code / "
            "MCP skill files. Inspects SKILL.md files against the Limenex "
            "governance taxonomy and emits a structured report."
        ),
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())