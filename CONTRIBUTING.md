# Contributing to Limenex

Limenex is an agentic AI execution governance layer. Its purpose is simple: intercept consequential agent actions before execution, evaluate them against policies, and either allow, block, or escalate them.

We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code. Right now, the most valuable contributions are improvements to the core library, documentation, tests, and especially **new governed skills**.

This guide explains how to contribute well, with detailed guidance on adding new skills to the library. Please read it before opening a pull request.

## Table of Contents

- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Reporting Issues](#reporting-issues)
- [Before You Start a Larger Change](#before-you-start-a-larger-change)
- [Contributing Skills](#contributing-skills)
  - [Design Principles](#design-principles)
  - [Naming and Placement](#naming-and-placement)
  - [Function Signature Rules](#function-signature-rules)
  - [Async vs Sync](#async-vs-sync)
- [Required Implementation Patterns](#required-implementation-patterns)
  - [Pattern A: Executor-Injected Skills](#pattern-a-executor-injected-skills)
  - [Pattern B: Direct Local Skills](#pattern-b-direct-local-skills)
- [Docstring Requirements](#docstring-requirements)
- [Testing Requirements](#testing-requirements)
- [Pull Request Checklist](#pull-request-checklist)
- [Good Skill Ideas](#good-skill-ideas)
- [What Not to Submit](#what-not-to-submit)
- [Questions](#questions)
- [Licensing](#licensing)

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/limenex-hq/limenex.git
   cd limenex
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the package in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run the test suite to verify your setup:
   ```bash
   pytest
   ```

## How to Contribute

1. Fork the repository on GitHub.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes.
4. Ensure all tests pass and code style is clean:
   ```bash
   pytest
   black --check .
   ruff check .
   ```
5. Open a pull request against `main` with a clear description of your changes.

## Code Style

- **black** for code formatting
- **ruff** for linting
- Line length: **88** characters
- All style rules are enforced by CI — please ensure your code passes before opening a PR.

## Reporting Issues

Please use [GitHub Issues](https://github.com/limenex-hq/limenex/issues) to report bugs. When filing an issue, include:

- Your Python version
- Your operating system
- A minimal reproducible example

## Before You Start a Larger Change

Before starting work on a new skill category, a change to the policy model, a public API change, or a major documentation restructure, please open an issue or start a discussion first. This helps us align on scope early and avoids wasted effort.

For small bug fixes and documentation tweaks, a direct pull request is fine.

---

## Contributing Skills

This is the main section of this guide. Everything below explains how to design, implement, test, and submit a new governed skill.

### Design Principles

Limenex is opinionated about what a skill is and how it should be built. All contributed skills must preserve these principles.

#### 1. Govern the action category, not the vendor

Skills are named after what they do, not which SDK or service they call. The risk Limenex governs is the action itself — vendor-specific behavior belongs inside the executor, not the skill identity.

Good:
- `finance.charge`
- `comms.send`
- `filesystem.delete`

Bad:
- `stripe_charge`
- `send_slack_message`
- `top_up_openai_credits`

#### 2. Narrow scope is mandatory

A skill wraps one tightly-scoped, consequential action. If you cannot write a clear, bounded policy against a function, it is too broad to be a Limenex skill.

Good:
- Send a message
- Charge a payment
- Delete a file

Bad:
- Process a transaction (auth, capture, refund are separate risks)
- Send and log a message (sending is consequential, logging is not)

#### 3. The agent call surface stays plain

The agent-facing function accepts only plain data parameters relevant to the action. Executors are never passed at call time — they are bound once at application startup through the skill factory.

#### 4. Deterministic governance should be easy

When designing a skill, prefer signatures that map cleanly to policy:

- **Numeric parameters** (`amount`, `amount_usd`, `estimated_cost_usd`) → good for deterministic threshold checks
- **String parameters** (`filepath`, `recipient`, `channel`, `region`) → good for exact-string allowlists/blocklists via `in`/`not_in`
- **Content-heavy parameters** (message body, payload) → usually better governed with `SemanticPolicy`

Limenex does not normalize, parse, or extract structure from strings. If a caller needs domain-level or prefix-based matching, they should pre-process the value before passing it to the skill.

### Naming and Placement

- Function names should be short and action-generic: `send`, `post`, `delete`.
- The governed `skill_id` must use `<category>.<action>` namespacing: `comms.send`, `finance.charge`.
- Add the skill to the matching module under `limenex/skills/` when a category already exists.
- Create a new category module only when the action category is genuinely new.
- Export the factory function and the skill ID constant from `limenex/skills/__init__.py`.

### Function Signature Rules

- `agent_id: str` is always the first parameter.
- Remaining parameters are plain domain data only.
- Do not expose executor callables in the call-time signature.
- Reuse `ReturnT` from `limenex.skills._types` for executor-backed skills. Do not define a local `TypeVar`.

### Async vs Sync

- **Executor-injected skills** → `async def`. This matches finance, comms, and web skills.
- **Stdlib-only local action skills** → may be synchronous. This matches filesystem skills.

---

## Required Implementation Patterns

### Pattern A: Executor-Injected Skills

Use this for skills that dispatch to one of several backends or vendors (e.g. `finance.charge`, `comms.send`, `web.post`).

The factory binds the skill to a `PolicyEngine` and accepts a `registry` dict keyed by the discriminator parameter (e.g. `provider`, `service`, `channel`, `destination`).

```python
from __future__ import annotations

import asyncio
from typing import Callable

from limenex.core.engine import PolicyEngine
from limenex.skills._exceptions import UnregisteredExecutorError
from limenex.skills._types import ReturnT

MY_SKILL_ID: str = "category.action"


def make_action(engine: PolicyEngine, registry: dict[str, Callable]) -> Callable:
    """Return a governed action skill bound to engine.

    Call once at application startup. The returned callable is safe to
    reuse across concurrent async tasks.

    Args:
        engine:   The PolicyEngine instance to bind this skill to.
        registry: Mapping of backend name to executor callable.
                  Executors receive (arg1=arg1, arg2=arg2).
                  agent_id and backend are never forwarded.
                  Sync and async callables are both supported.

    Returns:
        An async callable with signature:
        action(agent_id, backend, arg1, arg2) -> ReturnT
    """

    @engine.governed(MY_SKILL_ID, agent_id_param="agent_id")
    async def _governed(
        agent_id: str, backend: str, arg1: str, arg2: float
    ) -> None:
        pass

    async def action(
        agent_id: str,
        backend: str,
        arg1: str,
        arg2: float,
    ) -> ReturnT:
        """Governed skill: <describe the action>.

        <Docstring body — see Docstring Requirements below.>
        """
        if backend not in registry:
            raise UnregisteredExecutorError(MY_SKILL_ID, backend)

        await _governed(
            agent_id=agent_id,
            backend=backend,
            arg1=arg1,
            arg2=arg2,
        )
        
        executor = registry[backend]
        if asyncio.iscoroutinefunction(executor):
            return await executor(arg1=arg1, arg2=arg2)
        return executor(arg1=arg1, arg2=arg2)

    return action
```

**Key points:**

- `_governed` exists only for governance — it is the decorated inner function.
- The outer function is the real callable exposed to the application.
- The executor is captured from the factory closure, never passed by the agent.
- The executor must never appear in the kwargs seen by the policy engine.
- If the registry key is missing, raise `UnregisteredExecutorError` **before** governance runs.
- Governance state is recorded after governance passes but **before** the executor runs. Executor failure does not roll back recorded state.

### Pattern B: Direct Local Skills

Use this for narrow local actions with no executor injection (e.g. `filesystem.delete`, `filesystem.write`, `filesystem.move`).

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

from limenex.core.engine import PolicyEngine

MY_SKILL_ID: str = "category.action"


def make_action(engine: PolicyEngine) -> Callable:
    """Return a governed action skill bound to engine.

    Call once at application startup. The returned callable is sync
    and safe to reuse across calls.

    Args:
        engine: The PolicyEngine instance to bind this skill to.

    Returns:
        A sync callable with signature:
        action(agent_id, filepath) -> None
    """

    @engine.governed(MY_SKILL_ID, agent_id_param="agent_id")
    def action(agent_id: str, filepath: str) -> None:
        """Governed skill: <describe the action>.

        <Docstring body — see Docstring Requirements below.>
        """
        Path(filepath).write_text("example", encoding="utf-8")

    return action
```

**Key points:**

- The decorator is applied directly to the function body — no two-layer split.
- Prefer standard library implementations when possible.
- Do not add a registry when there is no real need for executor injection.
- State is recorded **after** the stdlib operation completes. If the operation raises, state is not advanced.

---

## Docstring Requirements

Every contributed skill must have a thorough docstring. The docstring is part of the public API and part of the teaching surface of the project.

At minimum, document:

- What the skill does and its `skill_id`
- Which parameters are governance dimensions and what policy types they suit:
  - Numeric parameters → `DeterministicPolicy` threshold checks
  - String parameters → `DeterministicPolicy` `in`/`not_in` checks
  - Content parameters → `SemanticPolicy`
- What gets forwarded to the executor and what does not
- That the executor is never called on `BLOCK` or `ESCALATE`
- Whether governance state is recorded before or after the actual side effect

See `limenex/skills/finance.py` and `limenex/skills/comms.py` for reference examples.

---

## Testing Requirements

Every new skill must include tests. Add them to `tests/test_skills.py` or a new test file if introducing a new category.

### Required Coverage

- **ALLOW** path — skill executes and returns the executor's return value
- **BLOCK** path — `BlockedError` raised, executor never called
- **ESCALATE** path — `EscalationRequired` raised, executor never called
- **Registry miss** — `UnregisteredExecutorError` raised before governance runs (executor-injected skills only)
- **Sync executor dispatch** — executor called correctly when sync
- **Async executor dispatch** — executor called correctly when async

### Test Rules

- **Mocks only.** Never call live external services in tests.
- **Filesystem tests** must use `tmp_path`.
- **Executor strip validation:** add a test proving the executor is never present in the kwargs seen by the engine. Use `CapturingPolicyEngine` (a `PolicyEngine` subclass that overrides `evaluate()` to capture kwargs) and assert `"executor"` is absent from every captured entry. This pattern is already used in `tests/test_skills.py` and should be the standard for all community-contributed skill tests.
- **Assert executor not called** on non-ALLOW verdicts.
- **Assert return value** — the skill returns whatever the executor returns on ALLOW.

---

## Pull Request Checklist

Review this checklist before opening a PR.

### For All PRs

- [ ] Change is focused and clearly scoped
- [ ] Code is formatted (`black`) and linted (`ruff`) locally
- [ ] All tests pass locally (`pytest`)
- [ ] Relevant documentation is updated
- [ ] `CHANGELOG.md` is updated if appropriate

### For New Skills (in addition to the above)

- [ ] Skill governs a narrow, consequential action
- [ ] Skill is action-category based, not vendor-branded
- [ ] `skill_id` uses `<category>.<action>` naming
- [ ] `agent_id: str` is the first parameter
- [ ] Call-time signature contains only plain data parameters
- [ ] Executor callables are injected at factory time, not call time
- [ ] Implementation follows the correct pattern (A or B above)
- [ ] Docstring explains governance-relevant parameters clearly
- [ ] Tests cover ALLOW, BLOCK, and ESCALATE verdict paths
- [ ] Tests verify executor is not called on non-ALLOW verdicts
- [ ] Tests verify executor-related objects are absent from engine kwargs
- [ ] No live external API calls are made in tests

---

## Good Skill Ideas

The skills library focuses on horizontal action categories that apply across industries. The current set covers finance, filesystem, comms, and web. If you see a broadly applicable risk category that is missing, open an issue with the proposed skill signature and example policies — we'd love to discuss it.

Domain-specific skills (e.g. trading, healthcare, legal) are best built in your own codebase using the same patterns documented above. The decorator and engine support any skill — the community library focuses on broadly applicable actions.

## What Not to Submit

Please do not open PRs for:

- Broad "god skills" that wrap many unrelated actions
- Vendor-specific wrappers presented as core skills (e.g. `stripe.charge` instead of `finance.charge`)
- Skills with no obvious policy surface
- Skills that hide governance-relevant parameters inside opaque blobs
- Large refactors mixed together with unrelated feature work

## Questions

If you are unsure whether something should be a skill, open an issue with:

1. The action you want to govern
2. Why it is consequential
3. The proposed function signature
4. The proposed `skill_id`
5. Example deterministic and semantic policies you'd expect users to write

That is usually enough to review the fit quickly.

## Licensing

This project is licensed under Apache 2.0. By submitting a pull request, you certify that your contribution is your own original work and that you have the right to submit it under the project's license, in accordance with the [Developer Certificate of Origin](https://developercertificate.org/).