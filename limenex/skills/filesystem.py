"""
filesystem.py — Filesystem skills for Limenex.

Governed execution functions for filesystem actions. Each skill is obtained
via a factory that binds it to a PolicyEngine instance at application startup.
All skills use stdlib directly — no executor injection.

Skill IDs (reference these in .limenex/policies.yaml):
    filesystem.delete  —  delete a file
    filesystem.write   —  write content to a file
    filesystem.move    —  move or rename a file
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

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        delete(agent_id, filepath) -> None
    """

    @engine.governed(DELETE_SKILL_ID, agent_id_param="agent_id")
    def delete(agent_id: str, filepath: str) -> None:
        """Governed skill: delete a file on behalf of an agent.

        Evaluates all policies registered under DELETE_SKILL_ID before
        performing the deletion. The file is never deleted on BLOCK or
        ESCALATE verdicts.

        Policy dimensions:
            filepath (str): Cannot be used as DeterministicPolicy.param —
                string values are not numeric. Use SemanticPolicy for
                path-based rules (e.g. "Do not allow deletion of files
                outside /tmp/"). Use DeterministicPolicy for frequency
                limits (e.g. max N deletions per hour — non-projective,
                no param required).

        Governance timing: state is recorded after the operation completes.
        If the filesystem operation raises, state is not advanced.

        Args:
            agent_id: The agent initiating this deletion. Used by the engine
                to resolve and record policy state.
            filepath: Absolute or relative path to the file to delete.

        Returns:
            None

        Raises:
            BlockedError: Policy verdict is BLOCK. File was not deleted.
            EscalationRequired: Policy verdict is ESCALATE. File was not deleted.
            FileNotFoundError: filepath does not exist.
        """
        Path(filepath).unlink()

    return delete


def make_write(engine: PolicyEngine) -> Callable:
    """Return a governed write skill bound to engine.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        write(agent_id, filepath, content) -> None
    """

    @engine.governed(WRITE_SKILL_ID, agent_id_param="agent_id")
    def write(agent_id: str, filepath: str, content: str) -> None:
        """Governed skill: write content to a file on behalf of an agent.

        Evaluates all policies registered under WRITE_SKILL_ID before
        performing the write. The file is never written on BLOCK or
        ESCALATE verdicts.

        Policy dimensions:
            filepath (str): Cannot be used as DeterministicPolicy.param —
                string values are not numeric. Path-based governance (e.g.
                "Do not allow writes outside the project directory") must
                use SemanticPolicy.
            content (str): Cannot be used as DeterministicPolicy.param.
                Content-based governance (e.g. "Do not write files containing
                credentials") must use SemanticPolicy.
            Frequency/volume limits (e.g. max N writes per hour): Use
                DeterministicPolicy without param — non-projective count check.

        Governance timing: state is recorded after the operation completes.
        If the filesystem operation raises, state is not advanced.

        Args:
            agent_id: The agent initiating this write. Used by the engine
                to resolve and record policy state.
            filepath: Absolute or relative path to the file to write.
                Created if it does not exist; overwritten if it does.
            content: UTF-8 string content to write to the file.

        Returns:
            None

        Raises:
            BlockedError: Policy verdict is BLOCK. File was not written.
            EscalationRequired: Policy verdict is ESCALATE. File was not written.
            OSError: filepath is not writable or the parent directory
                does not exist.
        """
        Path(filepath).write_text(content, encoding="utf-8")

    return write


def make_move(engine: PolicyEngine) -> Callable:
    """Return a governed move skill bound to engine.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        move(agent_id, src, dst) -> None
    """

    @engine.governed(MOVE_SKILL_ID, agent_id_param="agent_id")
    def move(agent_id: str, src: str, dst: str) -> None:
        """Governed skill: move or rename a file on behalf of an agent.

        Evaluates all policies registered under MOVE_SKILL_ID before
        performing the move. The file is never moved on BLOCK or
        ESCALATE verdicts.

        Policy dimensions:
            src (str): Cannot be used as DeterministicPolicy.param — string
                values are not numeric. Use SemanticPolicy for path-based
                rules (e.g. "Do not allow moves out of the working directory").
            dst (str): Same constraint as src.
            Frequency limits: Use DeterministicPolicy without param —
                non-projective count check.

        Governance timing: state is recorded after the operation completes.
        If the filesystem operation raises, state is not advanced.

        Args:
            agent_id: The agent initiating this move. Used by the engine
                to resolve and record policy state.
            src: Path to the source file.
            dst: Destination path. If dst is a directory, the file is moved
                into it preserving the filename.

        Returns:
            None

        Raises:
            BlockedError: Policy verdict is BLOCK. File was not moved.
            EscalationRequired: Policy verdict is ESCALATE. File was not moved.
            FileNotFoundError: src does not exist.
            shutil.Error: The move operation fails (e.g. src and dst resolve to
            the same path).

        """
        shutil.move(src, dst)

    return move
