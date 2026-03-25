"""
_types.py — Shared type aliases for the Limenex skills library.

Import ReturnT from here in all skill files — do not define a local TypeVar.
"""

from __future__ import annotations

from typing import TypeVar

__all__ = ["ReturnT"]

ReturnT = TypeVar("ReturnT")
