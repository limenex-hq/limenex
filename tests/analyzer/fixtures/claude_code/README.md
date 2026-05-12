# Claude Code analyzer fixtures

This directory holds the fixtures used to calibrate and regression-test
the Limenex Skill Risk Analyzer against Claude Code Skills. There are
two fixture sets with different purposes.

## Layout

```
claude_code/
├── real-skills.json            # Manifest of real skills (fetched, not vendored)
├── fixture_generation_prompt.md
├── skills/                     # Synthetic fixtures — authored, committed
├── expected/                   # Expected analyzer output for each synthetic skill
├── stories/                    # Narrative rationale for each synthetic skill
└── skills-real/                # Real-world skills — fetched on demand, gitignored
```

## Synthetic fixtures (`skills/`, `expected/`, `stories/`)

Handcrafted skills with known ground-truth analyzer output. Each skill in
`skills/<name>/` has a matching `expected/<name>.json` (the target analyzer
output) and `stories/<name>.md` (the intent and edge cases the fixture is
meant to exercise). These are the primary regression inputs — the analyzer
is expected to match `expected/*.json` within a small semantic tolerance.

Used by the A.8.1 calibration pass. Committed.

## Real-world fixtures (`skills-real/`)

Real Claude Code Skills sourced from public repositories, pinned by commit
SHA in `real-skills.json`. Fetched on demand by `scripts/fetch_real_skills.py`.

- Not vendored. Avoids redistributing third-party content and propagating
  upstream NOTICE / licence requirements into this repo.
- No expected-output fixtures. Analyzer output against these skills is the
  subject of a calibration judge pass, not a target to match.
- Reproducibility is via pinned SHAs in the manifest.

Used by the A.8.3 calibration pass. Gitignored.

### Fetching real skills

```
python scripts/fetch_real_skills.py
```

Reads `real-skills.json`, clones each upstream repo at its pinned SHA,
copies the skill subdirectory into `skills-real/<name>/`, and verifies
that a `SKILL.md` is present at the skill root. Re-runnable and
idempotent.

## Running the analyzer against fixtures

Synthetic pass (ground-truth regression):

```
python -m limenex.analyzer analyze tests/analyzer/fixtures/claude_code/skills
```

Real-skill pass (calibration, no ground truth):

```
python -m limenex.analyzer analyze tests/analyzer/fixtures/claude_code/skills-real
```

The two passes are kept separate because synthetic skills are
ground-truth-matched and real skills are judge-adjudicated — mixing them
in one report conflates two different quality signals.
