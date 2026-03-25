"""
limenex/skills/__init__.py — Public exports for the Limenex skills library.

Factories and skill ID constants for all built-in skills. Import from here
in application code:

    from limenex.skills import make_charge, CHARGE_SKILL_ID

Community contributors building new skills should import ReturnT directly
from limenex.skills._types rather than from this module.
"""

from __future__ import annotations

from limenex.skills.comms import SEND_SKILL_ID, make_send
from limenex.skills.filesystem import (
    DELETE_SKILL_ID,
    MOVE_SKILL_ID,
    WRITE_SKILL_ID,
    make_delete,
    make_move,
    make_write,
)
from limenex.skills.finance import (
    CHARGE_SKILL_ID,
    SPEND_SKILL_ID,
    make_charge,
    make_spend,
)
from limenex.skills.web import POST_SKILL_ID, make_post

__all__ = [
    # Finance
    "CHARGE_SKILL_ID",
    "SPEND_SKILL_ID",
    "make_charge",
    "make_spend",
    # Filesystem
    "DELETE_SKILL_ID",
    "WRITE_SKILL_ID",
    "MOVE_SKILL_ID",
    "make_delete",
    "make_write",
    "make_move",
    # Comms
    "SEND_SKILL_ID",
    "make_send",
    # Web
    "POST_SKILL_ID",
    "make_post",
]
