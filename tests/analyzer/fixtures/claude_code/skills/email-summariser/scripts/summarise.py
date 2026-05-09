"""Print a digest of unread IMAP messages grouped by sender.

Reads connection details from the environment. Headers only — never
fetches bodies, never marks anything as read.
"""

import email
import email.header
import imaplib
import os
import sys
from collections import defaultdict


def _decode_header(raw):
    if raw is None:
        return ""
    parts = email.header.decode_header(raw)
    out = []
    for chunk, enc in parts:
        if isinstance(chunk, bytes):
            try:
                out.append(chunk.decode(enc or "utf-8", errors="replace"))
            except LookupError:
                # weird/unknown charset label; fall back
                out.append(chunk.decode("utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out).strip()


def _sender_address(from_header):
    # the From: header is usually "Some Name <addr@host>" but
    # sometimes just "addr@host", and very occasionally something
    # weird. We want the bare address for grouping; if we can't get
    # one, just use whatever we have.
    decoded = _decode_header(from_header)
    if "<" in decoded and ">" in decoded:
        start = decoded.rfind("<")
        end = decoded.rfind(">")
        if end > start:
            return decoded[start + 1:end].strip().lower()
    return decoded.lower()


def fetch_unread(host, user, password, folder):
    conn = imaplib.IMAP4_SSL(host)
    try:
        conn.login(user, password)
        # readonly=True so SEARCH doesn't flip the \Seen flag on us
        typ, _ = conn.select(folder, readonly=True)
        if typ != "OK":
            raise RuntimeError(f"could not select folder {folder!r}")

        typ, data = conn.search(None, "UNSEEN")
        if typ != "OK":
            raise RuntimeError("IMAP SEARCH failed")

        ids = data[0].split()
        messages = []
        for msg_id in ids:
            typ, msg_data = conn.fetch(msg_id, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            parsed = email.message_from_bytes(raw)
            messages.append({
                "from": _sender_address(parsed.get("From")),
                "subject": _decode_header(parsed.get("Subject")) or "(no subject)",
            })
        return messages
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def render(messages):
    by_sender = defaultdict(list)
    for m in messages:
        by_sender[m["from"]].append(m["subject"])

    # stable-ish ordering: most-prolific senders first, then alpha
    senders = sorted(by_sender.keys(), key=lambda s: (-len(by_sender[s]), s))

    lines = []
    for s in senders:
        subjects = by_sender[s]
        lines.append(f"{s}  ({len(subjects)})")
        for subj in subjects:
            lines.append(f"  - {subj}")
    return "\n".join(lines)


def main():
    host = os.environ["IMAP_HOST"]
    user = os.environ["IMAP_USER"]
    password = os.environ["IMAP_PASSWORD"]
    folder = os.environ.get("IMAP_FOLDER", "INBOX")

    messages = fetch_unread(host, user, password, folder)
    if not messages:
        print("(no unread messages)")
        return 0

    print(render(messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
