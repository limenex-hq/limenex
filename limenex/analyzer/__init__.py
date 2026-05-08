"""Limenex Skill Risk Analyzer.

Consultative LLM-based audit tool for SKILL.md files. Inspects skills
against the Limenex governance taxonomy (finance / filesystem / comm / web)
and produces a structured report suitable for policy authoring.

The analyzer is consultative only: the runtime policy engine never reads
its output.

Running the analyzer CLI (``limenex-analyze``) requires the ``analyzer``
optional dependencies. Install with::

    pip install "limenex[analyzer]"

The report schema (``limenex.analyzer.schema``) is importable without
extras, so downstream consumers can validate analyzer output without
installing the full analyzer toolchain.
"""

__all__: list[str] = []