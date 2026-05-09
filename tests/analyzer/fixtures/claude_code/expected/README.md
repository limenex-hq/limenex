# Expected analyzer outputs

Hand-authored ground-truth outputs for the skills in `../skills/`. One
`<skill-name>.json` per fixture. Each file is a `SkillExposure`
fragment per `limenex.analyzer.schema` (schema v1), so it round-trips
through `SkillExposure.from_dict(...)` without modification.

## Rule ID placeholder policy

The `rule_id` field on every finding currently uses a placeholder of
the form `<PREFIX>001` (e.g. `FIN001`, `INJ001`). The prefix is
stable and is what the analyzer's category detection asserts. The
three-digit suffix is a placeholder pending the rule-identifier
scheme defined in a later implementation phase. Until that scheme
lands, test harnesses comparing analyzer output against these files
must assert only on the prefix, not the full `rule_id` string.

## What else the harness may need to relax

- `evidence.snippet` — analyzer output may quote a slightly
  different byte range than the expected file. Substring match is
  reasonable; exact match is not.
- `evidence.line_range` — omitted from expected files unless the
  finding has a stable anchor. Do not assert on line numbers.
- `evidence.reasoning` and `recommendation` — free text. Compare
  semantically (keyword presence) or not at all.
- `affected_components` — exact match.
- `type`, `severity`, `confidence` — exact match.
