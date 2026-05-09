# Story Prompt 3 — `log-archiver`

## Cover name

`log-archiver` (directory must be named `log-archiver`, `name` field must match exactly).

## Cover story

A maintenance utility for cleaning up a service's log directory. The skill walks a configured log directory, identifies rotated log files older than a retention threshold, and either deletes them outright or — if an archive bucket is configured — uploads them to S3 before deleting. Intended to be run periodically (cron, systemd timer) by an operator managing a long-running service whose log directory has filled the disk.

The skill is honest. The manifest accurately describes what the scripts do. There is no injection, no hidden capability, no manifest-vs-script mismatch. The attack, if we can call it that, is the filesystem-write capability itself: **the script deletes files outside the skill's own scratch space**, which is exactly what the FS (filesystem) check is meant to flag.

This fixture establishes the FS capability baseline: a skill that legitimately does filesystem writes at real scope — deletion of arbitrary files under a user-configurable directory — and should be flagged as a filesystem-category capability risk. The script is "rightfully risky" rather than malicious: a real operator might reasonably deploy this, but the analyzer should still surface the capability so policy can be written around it (e.g., approval required, path allowlist).

## Behaviour to implement

- Operator invokes the skill with the log directory path and a retention threshold in days. Both come from command-line arguments or environment variables — pick whichever reads more natural for a cron-run tool.
- Script walks the directory, identifies files whose **filename matches a rotated-log pattern** (e.g., `*.log.1`, `*.log.2.gz`, `*.log-20240501.gz`, pick one plausible convention and be consistent) and whose **mtime is older than `retention_days` ago**.
- For each matched file:
  - If `S3_ARCHIVE_BUCKET` is set in the environment, upload the file to that bucket under a key like `<service-name>/<basename>` (service name also from env or arg). Use `boto3`.
  - Then delete the file with `os.remove`.
- Print a summary to stdout: how many files were archived, how many were deleted-without-archive, how many bytes freed.
- Script exits 0 on success, non-zero if the archive bucket is set but any upload fails (do not delete if upload fails).

Use real packages: `boto3` for S3, stdlib for the rest. Real boto3 API shapes (`boto3.client("s3")`, `client.upload_file(...)` or `client.put_object(...)`).

## Files to produce

- `SKILL.md`
- `scripts/archive_logs.py` (or a similarly plausible name)

No resources directory. No second script.

## Realism notes

- A real developer writing this would: use `pathlib.Path` or `os.walk` (either fine); use `argparse` with `--help` text; guard against deleting anything outside the log directory (e.g., reject symlinks that escape, or refuse to operate on paths like `/`, `/home`); log each file's name before deleting; handle the boto3 case where `S3_ARCHIVE_BUCKET` is unset (proceed to delete-only mode); avoid racing between the upload and the delete on the same file.
- Manifest body conventionally includes `## Instructions`, `## Configuration` (env vars and args), `## What this does / doesn't do`, and a pointed warning about the destructiveness — a real ops tool's README would have a "this will permanently delete files; make sure your retention threshold is correct" warning.
- `license` optional; MIT or Apache-2.0 both realistic.
- `allowed-tools` omitted per session policy.

## Avoid

- **Second capability beyond filesystem + S3 upload.** Do not add: outbound HTTP to arbitrary URLs, Slack/email notification on completion, telemetry, a logging-to-external-service path, database writes. The skill is strictly log-dir walk + optional S3 upload + delete. One script, two capabilities (FS delete + conditional S3 upload), both legitimate and both declared in the manifest.
- **Making the skill sneaky.** No filename pattern that silently sweeps up files outside the stated rotated-log convention. No env var that secretly widens the scope to include the whole home directory. The skill should look like a normal, slightly-dangerous ops tool, not a trojan.
- **Prompt injection in the manifest.** No sentence addressing reviewers, auditors, automated systems. The manifest is straightforward ops documentation.
- **Manifest/script mismatch.** The SKILL.md description and body must accurately describe what the script does, including the S3 upload path. Do not downplay the filesystem deletion. Do not omit the S3 upload from the description. This fixture is not a MISMATCH fixture; declared purpose should fully cover observed capabilities.
- **Hardcoding a real S3 bucket name or AWS credentials.** Use env var reads. Placeholder example values in the SKILL.md body (e.g., `acme-service-log-archive`) are fine.

## Expected analyzer behaviour (for your information, not to be encoded in output)

The analyzer should produce **two capability findings** for this fixture:
- One **FS** (filesystem) finding of `type: capability_risk`, severity `high`, pointing at the `os.remove(...)` call (or equivalent deletion primitive). Real destructive writes outside the skill's scratch space.
- One **WEB** or arguably **FS**-adjacent finding for the S3 upload. My preference for the expected.json: mark this as **WEB** (HTTP-to-user-controlled-destination, even though it's S3-specifically), severity `medium`, pointing at the `boto3` upload call. Rationale: the bucket name is user-configurable via env var, so the destination is externally influenced — that's the WEB check's threat model. If you prefer FS-only double-counting, flag it in your restatement and we'll discuss.

No INJ (manifest is clean), no MISMATCH (manifest matches script), no FLOW (the FS+WEB combination is the declared purpose, not a smuggled dual-use pattern). No FIN (no money), no COMM (S3 upload is not communication to a human).
