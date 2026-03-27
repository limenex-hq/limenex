"""
_exceptions.py — Skills-layer exceptions for Limenex.
"""

from __future__ import annotations

__all__ = ["UnregisteredExecutorError"]


class UnregisteredExecutorError(Exception):
    """Raised when a discriminator key has no registered executor in the registry.

    Occurs at skill call time when the discriminator argument (e.g. service,
    channel, provider) does not match any key in the registry supplied at
    factory time. The skill was not executed.

    Attributes:
        skill_id: The skill that was called (e.g. "finance.spend").
        key:      The discriminator value that was not found in the registry
                  (e.g. "aws", "whatsapp", "stripe").

    Example:
        try:
            await spend(agent_id="agent-1", service="aws", amount_usd=50.0)
        except UnregisteredExecutorError as exc:
            print(exc.skill_id)  # "finance.spend"
            print(exc.key)       # "aws"
    """

    def __init__(self, skill_id: str, key: str) -> None:
        self.skill_id = skill_id
        self.key = key
        super().__init__(
            f"No executor registered for key '{key}' in skill '{skill_id}'. "
            f"Register it in the registry dict supplied to the skill factory."
        )
