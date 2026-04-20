"""CLI entry point for `python -m limenex`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .core.policy import DeterministicPolicy, PolicyConfig, SemanticPolicy
from .core.policy_store import LocalFilePolicyStore

_DEFAULT_POLICY_PATH = ".limenex/policies.yaml"


def _count_policies(configs: dict[str, PolicyConfig]) -> tuple[int, int, int]:
    total = deterministic = semantic = 0
    for cfg in configs.values():
        for policy in cfg.policies:
            total += 1
            if isinstance(policy, DeterministicPolicy):
                deterministic += 1
            elif isinstance(policy, SemanticPolicy):
                semantic += 1
    return total, deterministic, semantic


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(
            f"✗ Invalid — policy file not found at '{path}'. "
            f"Create the file or pass a path as the first argument "
            f"(e.g. `python -m limenex validate config/policies.yaml`).",
            file=sys.stderr,
        )
        return 1
    try:
        store = LocalFilePolicyStore(path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"✗ Invalid — {exc}", file=sys.stderr)
        return 1

    skills = store._configs
    total, deterministic, semantic = _count_policies(skills)
    print(
        f"✓ Valid — {len(skills)} skill{'s' if len(skills) != 1 else ''}, "
        f"{total} polic{'ies' if total != 1 else 'y'} "
        f"({deterministic} deterministic, {semantic} semantic)"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m limenex",
        description="Limenex CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate",
        help="Validate a policies.yaml file",
        description=(
            "Load the policies file via LocalFilePolicyStore and report "
            "skill and policy counts, or a clear error message pointing "
            "to the exact problem."
        ),
    )
    validate.add_argument(
        "path",
        nargs="?",
        default=_DEFAULT_POLICY_PATH,
        help=f"Path to policies.yaml (default: {_DEFAULT_POLICY_PATH})",
    )
    validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
