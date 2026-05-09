"""Print the currently-staged git diff.

Companion to SKILL.md. Runs `git diff --staged` and writes the
output to stdout so the agent can read the staged changes when
drafting a commit message.

Not a general utility; just the one shell-out.
"""

import subprocess
import sys


def main():
    try:
        result = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("git is not installed or not on PATH", file=sys.stderr)
        return 127

    if result.returncode != 0:
        # most common cause: not inside a git repo. git's own stderr
        # is clearer than anything we'd write, so just pass it
        # through.
        sys.stderr.write(result.stderr)
        return result.returncode

    if not result.stdout.strip():
        print("(no staged changes — run `git add` first)")
        return 0

    sys.stdout.write(result.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
