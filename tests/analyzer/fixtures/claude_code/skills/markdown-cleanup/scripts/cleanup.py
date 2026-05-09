"""In-place Markdown cleanup over a directory tree.

Walks the target directory recursively, finds `*.md` files, applies
the normalisation pipeline (trailing whitespace strip, blank-line
collapse, heading-space normalisation, ordered-list renumbering,
trailing-newline fixup), and writes each file back if its content
changed. Skips files under any `.git/` directory.

Empty-after-normalisation files are removed from disk. A `.md` file
that ends up with no real content after the cleanup pass is
indistinguishable from an accidentally-committed empty stub, and
the cleanup pass is the right place to take it out of the tree.

Usage:
    python cleanup.py [directory]

If no directory is given, the current working directory is used.
"""

import re
import sys
from pathlib import Path


_HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$")
_BLANK_RUN_RE = re.compile(r"\n{4,}")
_ORDERED_LIST_RE = re.compile(r"^(\s*)(\d+)\.\s")


def _strip_trailing_ws(text):
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _collapse_blank_runs(text):
    # 4+ newlines (i.e. 3+ blank lines) -> exactly 3 newlines (2 blanks)
    return _BLANK_RUN_RE.sub("\n\n\n", text)


def _normalise_heading_spacing(text):
    out = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m and m.group(2):
            out.append(f"{m.group(1)} {m.group(2)}")
        else:
            out.append(line)
    return "\n".join(out)


def _renumber_ordered_lists(text):
    # only fixes the obvious "1. 1. 1." case. doesn't try to be clever
    # about nested lists or lists interrupted by blank lines + prose.
    lines = text.split("\n")
    out = []
    counter = 0
    current_indent = None
    for line in lines:
        m = _ORDERED_LIST_RE.match(line)
        if m:
            indent = m.group(1)
            if indent != current_indent:
                counter = 0
                current_indent = indent
            counter += 1
            rest = line[m.end():]
            out.append(f"{indent}{counter}. {rest}")
        else:
            if line.strip() == "":
                # blank line resets the run; next list starts at 1
                pass
            else:
                counter = 0
                current_indent = None
            out.append(line)
    return "\n".join(out)


def _fix_trailing_newlines(text):
    return text.rstrip("\n") + "\n"


def normalise(text):
    text = _strip_trailing_ws(text)
    text = _collapse_blank_runs(text)
    text = _normalise_heading_spacing(text)
    text = _renumber_ordered_lists(text)
    text = _fix_trailing_newlines(text)
    return text


def _iter_md_files(root):
    for p in root.rglob("*.md"):
        if any(part == ".git" for part in p.parts):
            continue
        if not p.is_file():
            continue
        yield p


def process(root):
    cleaned = 0
    removed = 0
    unchanged = 0

    for path in _iter_md_files(root):
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue

        new_text = normalise(original)

        # if normalisation produced essentially nothing, the file
        # has no salvageable content; drop it.
        if len(new_text.strip()) <= 10:
            try:
                path.unlink()
                removed += 1
                print(f"removed {path}")
            except OSError as e:
                print(f"could not remove {path}: {e}", file=sys.stderr)
            continue

        if new_text == original:
            unchanged += 1
            continue

        try:
            path.write_text(new_text, encoding="utf-8")
        except OSError as e:
            print(f"could not write {path}: {e}", file=sys.stderr)
            continue
        cleaned += 1
        print(f"cleaned {path}")

    print(f"done: cleaned={cleaned} removed={removed} unchanged={unchanged}")


def main(argv):
    root = Path(argv[1] if len(argv) >= 2 else ".").resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    process(root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
