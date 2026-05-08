"""Limenex Skill Risk Analyzer.

Consultative LLM-based audit tool for SKILL.md files. Inspects skills
against the Limenex governance taxonomy (finance / filesystem / comm / web)
and produces a structured report suitable for policy authoring.

The analyzer is consultative only: the runtime policy engine never reads its output.
"""

__all__: list[str] = []