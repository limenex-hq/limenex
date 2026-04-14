---
title: "Your AI Agent Just Spent $3,000. Nobody Told It to Stop."
date: 2026-04-20
description: "Limenex is an open-source, deterministic stateful governance layer for AI agents and agentic systems."
---

**Limenex*** is an open-source, deterministic stateful governance layer for AI agents and agentic systems.*

<!-- more -->

---

Agentic AI has been the talk of the town. In January 2026, OpenClaw went from zero to 180,000 GitHub stars in two weeks --- the fastest-adopted AI agent in history. It can manage your email, handle payments through Stripe, operate on your filesystem, and take autonomous actions across your digital life.

Users soon realized it might be a little "too powerful". Within weeks, security researchers found 341 malicious skills in ClawHub, making up 12% of the entire registry. A critical RCE vulnerability (CVE-2026-25253) allowed full instance takeover from a single malicious link. Over 40,000 instances were found exposed on the public internet with no authentication. In China, users who had rushed to install OpenClaw were paying to have it removed after serious and unexpected financial and security consequences. Damages were done, and some were irreversible.

The incidents show us how capable agentic AI is and what the potential can be. Though capability without governance is a recipe for disaster --- as we have seen with the OpenClaw fiasco.

---

***The industry knows the problem. Progress has been made, but gaps remain.***

**Prompt-level enforcement.** Claude has `CLAUDE.md`. OpenAI has instruction hierarchy. Both pin rules in the model's context. This approach is vulnerable to prompt injection and degrades in effectiveness as the context window grows. The UK's National Cyber Security Centre puts it plainly --- LLMs are "inherently confusable deputies."

**Code-level governance.** LangChain offers middleware that can gate tool calls before execution. Microsoft's Agent Governance Toolkit is a dedicated policy engine that evaluates function parameters at runtime. Both operate outside the token stream, so prompt injection can't bypass them. But both are stateless by design: each call is evaluated in isolation with no memory of previous actions. This leaves a practical gap --- they can block a $500 charge, but they wouldn't catch a 50th $10 charge.

---

***How Limenex approaches the problem***

Limenex is an open-source, deterministic governance layer for AI agents. The core philosophy is intent-execution separation: the model decides what it *wants* to do, but a policy engine that sits entirely outside the model determines whether it's *allowed* to. AI agents are like employees at your organisation --- they can think and plan freely, but policies draw the line between what they can execute unilaterally and what requires sign-off. What actions are allowed, what requires human approval, and what is never permitted --- defined in config, not in a prompt.

Six core design principles guide our architecture.

**Govern actions, not intent.** Limenex operates on the skill call itself --- the skill name, the parameters, the values. A prompt injection can manipulate what an agent intends to do. It can't change the fact that `amount_usd=600` gets evaluated against a policy before the executor fires.

**Skills, not functions.** Not every agent action needs governance --- only the ones that carry real risk. A skill is a tightly-scoped, vendor-agnostic wrapper around a single consequential action: charging a payment, deleting a file, sending a message. Skills are named after what they do, not which SDK executes it. `finance.charge` is the skill --- Stripe, Square, or Braintree is just the executor you inject. Limenex governs whether it runs.

**Stateful by default.** The engine natively supports cumulative state. When a new action comes in, it projects whether executing it would breach a limit defined in your policies --- before it executes. A stateful policy like "block if cumulative spend exceeds $100" doesn't require custom logic.

**Three verdicts.** Every policy evaluation returns ALLOW, BLOCK, or ESCALATE. BLOCK is a hard stop --- no override, no approval path. ESCALATE pauses the action and routes it to a human. The policy author decides which is appropriate.

**Policies are data, not code.** Rules are defined in YAML, not embedded into your agent logic. Changing a spending cap or tightening a file deletion policy means updating a few lines in a file, instead of touching code and redeploying.

**Policies can be deterministic, or semantic.** Hard, rule-based checks like spending limits, path allowlists, per-call caps, are deterministic. For natural language rules that require reasoning and judgement, e.g. "escalate if the email tone is aggressive", semantic policies are natively supported, with evaluation done by a separate LLM you provide.

---

***Limenex in practice***

Let's look at an example. Suppose you have a research agent that can top up its own OpenAI API credits --- a common pattern in production agentic systems. Here's what the tool looks like without governance, and what changes when you wire it through Limenex.

**Before: naive top-ups (no governance)**

```python
# BEFORE: unguarded "top up OpenAI credits" helper

async def naive_top_up_openai(amount_usd: float) -> None:
    print(f"[naive] Topped up OpenAI credits by ${amount_usd:.2f}")

async def naive_scenario() -> None:
    print("=== BEFORE: naive top-ups ===")
    await naive_top_up_openai(40.0)   # looks fine
    await naive_top_up_openai(600.0)  # just as easy

await naive_scenario()
```

In this scenario, there are no limits and no approval path. The agent can top up $40 or $600 without any issues. You could hardcode an `if amount_usd > 500` check --- that works for one function in one service. Limenex solves this across all your agent's skills, with rules defined in YAML that anyone can change without redeploying.

**After: Limenex-governed top-ups**

We define two deterministic policies for `finance.spend`:

1. **BLOCK** any single top-up over $500 --- a **stateless** per-call check (no approval path).
2. **ESCALATE** when cumulative spend for this agent would exceed $100 --- a **stateful** cumulative check (human review).

```yaml
finance.spend:
  policies:
    # Hard-block any single top-up over $500 — no approval path
    - type: deterministic
      dimension: openai_single_topup_usd
      operator: lt
      value: 500.0
      param: amount_usd
      breach_verdict: BLOCK
      stateful: false

    # Escalate if cumulative spend would exceed $100 — human review
    - type: deterministic
      dimension: openai_cumulative_spend_usd
      operator: lt
      value: 100.0
      param: amount_usd
      breach_verdict: ESCALATE
```

Note that policy order matters --- the engine evaluates in sequence and stops on the first non-ALLOW verdict, so hard safety rules go first.

Set up the engine and create the governed skill:

```python
from pathlib import Path
from limenex.core.engine import BlockedError, EscalationRequired, PolicyEngine
from limenex.core.policy_store import LocalFilePolicyStore
from limenex.core.stores import LocalFileStateStore
from limenex.skills.finance import make_spend

# Set up Limenex engine and the governed finance.spend skill
policy_store = LocalFilePolicyStore(policies_path)
state_store = LocalFileStateStore(limenex_dir / "state_langgraph.json")
engine = PolicyEngine(policy_store=policy_store, state_store=state_store)

# Payment executor — this would be your real OpenAI billing call in production
async def openai_top_up_executor(amount_usd: float) -> None:
    print(f"[governed] Approved OpenAI top-up of ${amount_usd:.2f}", flush=True)

spend = make_spend(engine, registry={"openai": openai_top_up_executor})

# This is what you'd expose to your LLM agent as a tool — plain data parameters only:
# async def top_up_openai(amount_usd: float) -> None: ...
AGENT_ID = "research-agent-1"

async def top_up_openai(amount_usd: float) -> None:
    # agent_id is application-layer context, not agent input
    await spend(agent_id=AGENT_ID, service="openai", amount_usd=amount_usd)
```

Optionally, wrap it as a LangGraph tool:

```python
from langchain_core.tools import tool

@tool
async def top_up_openai_credits(amount_usd: float) -> str:
    \"\"\"Top up OpenAI API credits. Governed by Limenex — subject to
    per-call and cumulative spend limits.\"\"\"
    try:
        await spend(agent_id=AGENT_ID, service="openai", amount_usd=amount_usd)
        return f"Topped up ${amount_usd:.2f} successfully."
    except BlockedError:
        return f"Blocked — ${amount_usd:.2f} exceeds the single transaction limit."
    except EscalationRequired:
        return f"Escalated — ${amount_usd:.2f} requires human approval."
```

Pass `top_up_openai_credits` to a `ToolNode` and bind it to your agent's model. The agent sees a plain function with `amount_usd: float`. Governance is invisible to the agent and to your graph definition.

**What happens at runtime**

Let's imagine the agent makes four calls:

1. Top up $40 (allowed).
2. Top up $30 (allowed, cumulative now $70).
3. Top up $50 (would take cumulative to $120) → ESCALATE.
4. Top up $600 (fails the single-top-up limit) → BLOCK.

| Call | Amount | Cumulative | Verdict | Why |
|------|--------|-----------|---------|-----|
| 1 | $40 | $40 | ALLOW | Under both limits |
| 2 | $30 | $70 | ALLOW | Still under $100 cumulative |
| 3 | $50 | would be $120 | ESCALATE | $70 + $50 > $100 — routed to human |
| 4 | $600 | — | BLOCK | $600 > $500 — hard stop |

Call 3 raises `EscalationRequired`. Your application catches it, stores the paused action, notifies a human, and re-invokes or rejects based on their decision. Call 4 never reaches the executor. A retry loop burning $3,000 hits ESCALATE at $100 and stops --- not because the agent chose to, but because the policy engine made it.

---

***What Limenex governs***

Limenex currently ships governed skills across four categories:

| Category | Skills | Example policy |
|----------|--------|---------------|
| **Finance** | `finance.charge`, `finance.spend` | Block if single charge > $500; escalate if cumulative spend > $100 |
| **Filesystem** | `filesystem.delete`, `filesystem.write`, `filesystem.move` | Block if filepath not in allowed directories |
| **Comms** | `comms.send` | Escalate if recipient not in approved contacts; semantic check on message tone |
| **Web** | `web.post` | Block if destination not in allowlisted endpoints |

If a function is too broad to attach a clear, bounded policy to, it shouldn't be a skill. Skills are intentionally the smallest meaningful unit of consequential action. The library focuses on horizontal categories that apply across industries — domain-specific skills are best built in your own codebase using the same patterns and engine.

---

Agentic AI is powerful — but that power needs guardrails to be safe and useful in production. Limenex is our little contribution to making that happen.

***Get started***

```bash
pip install limenex
```

Define your policies in YAML. Wire up the engine. Wrap your tool. That's it.

→ [GitHub](https://github.com/limenex-hq/limenex)

→ [Quickstart guide](https://limenex.dev)

→ [Full LangGraph example notebook](https://github.com/limenex-hq/limenex/blob/main/examples/langgraph_example.ipynb)

If this solves a problem you've been thinking about, star the repo — it helps other developers find it.

---
