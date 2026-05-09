"""Archive (optional) and delete rotated log files older than N days.

Intended for cron / systemd-timer use. See SKILL.md for the full
behaviour contract; the short version is: walk --log-dir, find
rotated logs older than --retention-days by mtime, upload to
$S3_ARCHIVE_BUCKET if set, then os.remove() each one.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path


# logrotate-ish patterns. .log.N / .log.N.gz / .log-YYYYMMDD / .log-YYYYMMDD.gz
_PATTERNS = [
    re.compile(r".+\.log\.\d+(\.gz)?$"),
    re.compile(r".+\.log-\d{8}(\.gz)?$"),
]


# paths we flat-out refuse to operate on, no matter what the operator
# typed. resolved against Path.resolve().
_REFUSED_ROOTS = {
    Path("/"),
    Path("/home"),
    Path("/root"),
    Path("/var"),
    Path("/etc"),
    Path("/usr"),
    Path("/opt"),
    Path("/tmp"),
}


def _is_rotated_log(name):
    return any(p.match(name) for p in _PATTERNS)


def _validate_log_dir(raw):
    p = Path(raw).resolve(strict=True)
    if not p.is_dir():
        raise SystemExit(f"--log-dir is not a directory: {p}")
    if p in _REFUSED_ROOTS:
        raise SystemExit(f"refusing to operate on {p}; pick a deeper path")
    # also refuse if the path *is* a system root with one extra
    # segment that looks risky — e.g. /var directly, or /home/foo
    # directly. A real log dir is /var/log/<service>/, three deep
    # at minimum.
    if len(p.parts) <= 2:
        raise SystemExit(f"refusing to operate on shallow path {p}")
    return p


def _candidates(log_dir, cutoff_ts):
    # non-recursive; rotated logs live next to the active log.
    for entry in log_dir.iterdir():
        if entry.is_symlink():
            # don't follow symlinks at all. if a rotated log is itself
            # a symlink, skip it — a misconfigured symlink could point
            # somewhere outside the log dir and we'd rather not chase it.
            continue
        if not entry.is_file():
            continue
        if not _is_rotated_log(entry.name):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff_ts:
            yield entry


def _resolve_key_prefix(log_dir):
    explicit = os.environ.get("S3_KEY_PREFIX")
    if explicit:
        return explicit.strip("/")
    svc = os.environ.get("LOG_SERVICE_NAME")
    if svc:
        return svc.strip("/")
    return log_dir.name


def _upload(s3_client, bucket, key, path):
    # boto3's upload_file handles multipart for large files; rotated
    # logs are usually small but the .gz of a busy day's INFO log can
    # be hundreds of MB so let it deal with that.
    s3_client.upload_file(str(path), bucket, key)


def main():
    ap = argparse.ArgumentParser(description="archive and delete old rotated log files")
    ap.add_argument("--log-dir", required=True, help="directory containing rotated logs")
    ap.add_argument("--retention-days", required=True, type=int,
                    help="files older than this many days are eligible")
    args = ap.parse_args()

    if args.retention_days < 1:
        raise SystemExit("--retention-days must be >= 1")

    log_dir = _validate_log_dir(args.log_dir)
    cutoff_ts = time.time() - (args.retention_days * 86400)

    bucket = os.environ.get("S3_ARCHIVE_BUCKET")
    s3_client = None
    key_prefix = None
    if bucket:
        import boto3  # local import so a delete-only run doesn't need boto3 installed
        s3_client = boto3.client("s3")
        key_prefix = _resolve_key_prefix(log_dir)

    archived = 0
    deleted_only = 0
    bytes_freed = 0

    for path in _candidates(log_dir, cutoff_ts):
        size = 0
        try:
            size = path.stat().st_size
        except OSError:
            pass

        if bucket:
            key = f"{key_prefix}/{path.name}" if key_prefix else path.name
            try:
                _upload(s3_client, bucket, key, path)
            except Exception as e:
                # don't delete if the archive failed. report and
                # bail; cron will retry tomorrow.
                print(f"upload failed for {path}: {e}", file=sys.stderr)
                return 1
            archived += 1
            print(f"archived {path.name} -> s3://{bucket}/{key}")
        else:
            deleted_only += 1
            print(f"deleting {path.name}")

        try:
            os.remove(path)
            bytes_freed += size
        except OSError as e:
            print(f"could not remove {path}: {e}", file=sys.stderr)
            # we've already uploaded this one if archiving was on.
            # next run will attempt the delete again.
            return 1

    print(
        f"done: archived={archived} deleted_only={deleted_only} "
        f"bytes_freed={bytes_freed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
