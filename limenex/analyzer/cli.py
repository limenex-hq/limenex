"""Limenex analyzer CLI entry point.

Running this CLI requires the ``analyzer`` optional dependencies
(``httpx``, ``rich``). Install with::

    pip install "limenex[analyzer]"

If the dependencies are missing, the CLI writes an install hint to
stderr and exits 1 before any analyzer logic runs.
"""

from __future__ import annotations

import argparse
import sys

_MISSING_DEPS_MESSAGE = (
    "The limenex analyzer requires optional dependencies that are not installed.\n"
    "\n"
    "Install them with:\n"
    "\n"
    '    pip install "limenex[analyzer]"\n'
)


def _check_dependencies() -> None:
    """Verify analyzer optional dependencies are importable.

    Exits with status 1 and a helpful install hint on stderr if either
    ``httpx`` or ``rich`` is missing.
    """
    missing: list[str] = []
    try:
        import httpx  # noqa: F401
    except ImportError:
        missing.append("httpx")
    try:
        import rich  # noqa: F401
    except ImportError:
        missing.append("rich")
    if missing:
        sys.stderr.write(_MISSING_DEPS_MESSAGE)
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    _check_dependencies()
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
