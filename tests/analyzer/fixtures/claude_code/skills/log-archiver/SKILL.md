---
name: log-archiver
description: Maintenance helper that walks a service's log directory, finds rotated log files older than a retention threshold, optionally uploads them to a configured S3 archive bucket, and then deletes them from disk. Use periodically (cron, systemd timer) on a host where a long-running service's log directory is filling the disk and you've already decided what your retention policy should be. Permanently deletes files; if an S3 archive bucket is configured, files are uploaded there first and only deleted on successful upload.
license: Apache-2.0
---

# log-archiver

Walk a log directory, find old rotated log files, optionally archive
them to S3, then delete them.

## When to use this

You have a service writing to e.g. `/var/log/<svc>/`, logrotate is
producing `<svc>.log.1`, `<svc>.log.2.gz`, … or dated variants like
`<svc>.log-20250501.gz`, and you need a periodic job to clear out
the rotated files older than N days. If you also want a long-tail
archive in S3, set `S3_ARCHIVE_BUCKET` and the script will upload
each file before deleting the local copy.

If you don't need S3 archival at all, leave `S3_ARCHIVE_BUCKET` unset
and the script just deletes.

## Configuration

Args:

| Arg                 | Required | Notes                                              |
|---------------------|----------|----------------------------------------------------|
| `--log-dir PATH`    | yes      | The directory to scan. Must already exist.         |
| `--retention-days N`| yes      | Files older than N days (by mtime) are eligible.   |

Environment:

| Var                  | Required | Notes                                              |
|----------------------|----------|----------------------------------------------------|
| `S3_ARCHIVE_BUCKET`  | no       | If set, files uploaded here before delete.         |
| `S3_KEY_PREFIX`      | no       | Prefix under the bucket; defaults to `LOG_SERVICE_NAME` if set, else the log dir's basename. |
| `LOG_SERVICE_NAME`   | no       | Used as the default key prefix and shown in output.|
| AWS creds            | only if archiving | Standard `boto3` resolution: env, profile, IAM role. |

## Instructions

1. Decide your retention threshold. Once. Permanently. The script
   will delete files older than this, every time it runs.
2. Decide whether you want S3 archival. If yes, create the bucket
   and confirm the running user has `s3:PutObject` on it. If you
   want a non-default key layout, set `S3_KEY_PREFIX` explicitly.
3. Dry-run first by pointing it at a copy of the log dir, or — if
   you trust the pattern matcher — run with `--retention-days` set
   to a very large number so nothing is eligible, and confirm the
   "candidates" output looks right.
4. Wire it into cron or a systemd timer. Example crontab line:

   ```
   17 4 * * *  /usr/bin/python3 /opt/log-archiver/scripts/archive_logs.py \
       --log-dir /var/log/acme-api --retention-days 14
   ```

5. Check stdout / your cron mail occasionally. The script prints a
   one-line summary: how many files archived, how many deleted-only,
   how many bytes freed.

## What this does / doesn't do

Does:

- Walks `--log-dir` non-recursively (rotated logs live alongside the
  active log; nested dirs are intentionally ignored).
- Matches rotated-log filenames: `*.log.N`, `*.log.N.gz`,
  `*.log-YYYYMMDD`, `*.log-YYYYMMDD.gz`. Anything else is skipped.
- Optionally uploads each match to `s3://$S3_ARCHIVE_BUCKET/<prefix>/<basename>`.
- Deletes each match from local disk.
- Refuses to follow symlinks that escape `--log-dir`. Refuses to
  operate on `/`, `/home`, `/var`, `/etc`, `/usr`, `/opt`, or any of
  their immediate roots — has to be a deeper path than that.

Doesn't:

- Doesn't touch the active (currently-being-written) log file. The
  rotated-log pattern is precisely what excludes it.
- Doesn't recurse into subdirectories.
- Doesn't send notifications anywhere — no Slack, no email, no
  webhook. If you want alerts on failure, wrap it in your existing
  cron alerting.
- Doesn't write any state file. It's stateless; mtime + filename
  pattern is the entire decision surface.

## Warning

This permanently deletes files. If your `--retention-days` is wrong
or your `--log-dir` points somewhere it shouldn't, you will lose
data. Read the path-safety guards in the script if you don't trust
your own typing — they exist precisely because this is the failure
mode.
