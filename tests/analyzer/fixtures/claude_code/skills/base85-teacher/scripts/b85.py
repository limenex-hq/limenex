"""Tiny encode/decode wrapper around base64.b85encode/decode and
base64.a85encode/decode for the worked examples in SKILL.md.

Not a production utility. In real code, call the stdlib directly.

Usage:
    python b85.py encode <text>
    python b85.py decode <encoded>
    python b85.py encode --ascii85 <text>
    python b85.py decode --ascii85 <encoded>
    echo -n 'data' | python b85.py encode -

Defaults to RFC 1924 (the alphabet that's safe in JSON / XML / SQL
quoted contexts). Pass --ascii85 for the Adobe / PDF variant.
"""

import base64
import binascii
import sys


def _read_input(arg):
    if arg == "-":
        return sys.stdin.buffer.read()
    return arg.encode("utf-8")


def main(argv):
    args = list(argv[1:])
    if not args:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    use_ascii85 = False
    if "--ascii85" in args:
        use_ascii85 = True
        args.remove("--ascii85")

    if len(args) != 2:
        print("expected: <encode|decode> <text-or-->", file=sys.stderr)
        return 2

    op, payload = args
    if op not in ("encode", "decode"):
        print(f"unknown op: {op!r}", file=sys.stderr)
        return 2

    raw = _read_input(payload)

    try:
        if op == "encode":
            out = (base64.a85encode(raw) if use_ascii85
                   else base64.b85encode(raw))
            sys.stdout.buffer.write(out + b"\n")
        else:
            decoded = (base64.a85decode(raw) if use_ascii85
                       else base64.b85decode(raw))
            sys.stdout.buffer.write(decoded)
    except (binascii.Error, ValueError) as e:
        print(f"could not {op}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
