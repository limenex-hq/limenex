"""
filesystem.py — Filesystem skills for Limenex.

Governed execution functions for local filesystem actions. Each skill is
obtained via a factory that binds it to a PolicyEngine instance at
application startup.

Skill IDs (reference these in .limenex/policies.yaml):
    filesystem.delete  —  delete a file or empty directory
    filesystem.write   —  write text content to a file
    filesystem.move    —  move or rename a file or directory

Policy guidance:
    String path parameters (filepath, src, dst) now support exact-string
    DeterministicPolicy checks using in/not_in operators. This enables
    path allowlists and blocklists without routing to SemanticPolicy.

    Matching is exact and case-sensitive. Limenex does not normalise,
    resolve, or canonicalise paths before comparison. For subtree or
    prefix-based path governance, normalise or extract the relevant
    path component before calling the skill and use in/not_in on the
    extracted value.

"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from limenex.core.engine import PolicyEngine

__all__ = [
    "DELETE_SKILL_ID",
    "WRITE_SKILL_ID",
    "MOVE_SKILL_ID",
    "make_delete",
    "make_write",
    "make_move",
]

DELETE_SKILL_ID: str = "filesystem.delete"
WRITE_SKILL_ID: str = "filesystem.write"
MOVE_SKILL_ID: str = "filesystem.move"


def make_delete(engine: PolicyEngine) -> Callable:
    """Return a governed delete skill bound to engine.

    Call once at application startup. The returned callable is sync and safe
    to reuse across calls.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        delete(agent_id, filepath) -> None
    """

    @engine.governed(DELETE_SKILL_ID, agent_id_param="agent_id")
    def delete(agent_id: str, filepath: str) -> None:
        """Governed skill: delete a file or empty directory.

        Evaluates all policies registered under DELETE_SKILL_ID before
        executing the local filesystem action. The delete never occurs on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            filepath (str): Supports exact-string DeterministicPolicy checks
                via in/not_in operators. Example: allow deletion only for
                specific paths using DeterministicPolicy(
                    operator="in",
                    values=frozenset({"/tmp/allowed.txt"}),
                    param="filepath",
                ).
                Matching is exact and case-sensitive; paths are compared as
                provided. For subtree or prefix-based rules, extract or
                normalise the relevant path component before calling the skill
                and use in/not_in on the extracted value.

            frequency / count: Use a numeric DeterministicPolicy with
                param=None to govern how often delete is called
                (e.g. daily delete count).

        Args:
            agent_id:  The agent initiating this delete. Used by the engine
                       to resolve policy state.
            filepath:  Path to delete. Supports exact-string in/not_in policy
                       checks and is passed directly to the local filesystem
                       operation with no canonicalisation by Limenex.

        Raises:
            BlockedError:        Policy verdict is BLOCK. No delete occurs.
            EscalationRequired:  Policy verdict is ESCALATE. No delete occurs.
            FileNotFoundError:   Path does not exist.
            IsADirectoryError:   Path is a non-empty directory.
            PermissionError:     OS denies deletion.
        """
        path = Path(filepath)
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()

    return delete


def make_write(engine: PolicyEngine) -> Callable:
    """Return a governed write skill bound to engine.

    Call once at application startup. The returned callable is sync and safe
    to reuse across calls.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        write(agent_id, filepath, content) -> None
    """

    @engine.governed(WRITE_SKILL_ID, agent_id_param="agent_id")
    def write(agent_id: str, filepath: str, content: str) -> None:
        """Governed skill: write text content to a file.

        Evaluates all policies registered under WRITE_SKILL_ID before
        executing the local filesystem action. The write never occurs on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            filepath (str): Supports exact-string DeterministicPolicy checks
                via in/not_in operators. Example: restrict writes to an
                approved path allowlist using DeterministicPolicy(
                    operator="in",
                    values=frozenset({"/workspace/output.txt"}),
                    param="filepath",
                ).
                Matching is exact and case-sensitive; paths are compared as
                provided. For subtree or prefix-based rules, extract or
                normalise the relevant path component before calling the skill
                and use in/not_in on the extracted value.

            content (str): Not a good fit for DeterministicPolicy. Use
                SemanticPolicy for rules about what may be written.

            frequency / count: Use a numeric DeterministicPolicy with
                param=None to govern how often write is called
                (e.g. daily write count).

        Args:
            agent_id:  The agent initiating this write. Used by the engine
                       to resolve policy state.
            filepath:  Path to write. Supports exact-string in/not_in policy
                       checks and is passed directly to the local filesystem
                       operation with no canonicalisation by Limenex.
            content:   Text content to write. Forwarded directly to the file.

        Raises:
            BlockedError:        Policy verdict is BLOCK. No write occurs.
            EscalationRequired:  Policy verdict is ESCALATE. No write occurs.
            FileNotFoundError:   Parent directory does not exist.
            PermissionError:     OS denies writing.
        """
        Path(filepath).write_text(content, encoding="utf-8")

    return write


def make_move(engine: PolicyEngine) -> Callable:
    """Return a governed move skill bound to engine.

    Call once at application startup. The returned callable is sync and safe
    to reuse across calls.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        move(agent_id, src, dst) -> None
    """

    @engine.governed(MOVE_SKILL_ID, agent_id_param="agent_id")
    def move(agent_id: str, src: str, dst: str) -> None:
        """Governed skill: move or rename a file or directory.

        Evaluates all policies registered under MOVE_SKILL_ID before
        executing the local filesystem action. The move never occurs on
        BLOCK or ESCALATE verdicts.

        Policy dimensions:
            src (str), dst (str): Both parameters support exact-string
                DeterministicPolicy checks via in/not_in operators. Example:
                allow moves only from a known source set by targeting
                param="src", or block moves into sensitive destinations by
                targeting param="dst". Matching is exact and case-sensitive;
                paths are compared as provided. For subtree or prefix-based
                rules, extract or normalise the relevant path component before
                calling the skill and use in/not_in on the extracted value.

            frequency / count: Use a numeric DeterministicPolicy with
                param=None to govern how often move is called.

        Args:
            agent_id:  The agent initiating this move. Used by the engine
                       to resolve policy state.
            src:       Source path. Supports exact-string in/not_in policy
                       checks and is passed directly to shutil.move.
            dst:       Destination path. Supports exact-string in/not_in
                       policy checks and is passed directly to shutil.move.

        Raises:
            BlockedError:        Policy verdict is BLOCK. No move occurs.
            EscalationRequired:  Policy verdict is ESCALATE. No move occurs.
            FileNotFoundError:   Source path does not exist.
            PermissionError:     OS denies the move.
        """
        shutil.move(src, dst)

    return move
