---
name: base85-teacher
description: Reference and demonstration material for the base85 encoding family (Ascii85, RFC 1924, Z85). Use when the user asks what base85 is, how it differs from base64, which alphabet variant to choose, why an encoded string is the length it is, or is debugging an unfamiliar base85-encoded blob and wants help identifying or decoding it.
license: MIT
---

# base85-teacher

A working reference for the base85 encoding family. Reach for this
when someone asks "what's base85", "why is this string so dense",
"which variant is RFC 1924", or you're staring at a blob and trying
to figure out what alphabet it's in.

The companion script in `scripts/b85.py` is a tiny encode/decode
demo wrapping the stdlib. It exists so you can show the user a
concrete round-trip; it is not a production utility.

## Quick orientation

base85 is a family of binary-to-text encodings that pack 4 bytes
into 5 ASCII characters, giving a 4:5 input-to-output ratio. base64
gives 3:4. So base85 is denser — an encoded payload is roughly 25%
shorter than the same data in base64 (5/4 = 1.25 vs 4/3 ≈ 1.33).

Honestly: just use base64 unless you have a specific reason to want
the density. base64 is universally supported, the alphabet is
URL/email/shell-friendly, and nobody has ever been confused about
which variant you meant. base85 is the right choice in a few
narrow places — embedding binary in PDF streams, fitting an IPv6
address into a compact textual form, packing binary into source
code where every byte counts — and the wrong choice almost
everywhere else.

## The alphabets

There are several base85 variants. They all use 85 printable ASCII
characters but they pick *which* 85, and that picks the contexts
the encoding is safe in.

### Ascii85 (Adobe / btoa)

The original. Alphabet is `!` (0x21) through `u` (0x75) — a
contiguous run of 85 ASCII characters. Used in PostScript and PDF.
Adobe's variant wraps the payload in `<~` … `~>` delimiters and
adds the `z` shorthand: a run of four zero bytes encodes as the
single character `z` instead of `!!!!!`.

The trouble with Ascii85 outside PDF is that its alphabet includes
`'`, `"`, `\`, `<`, `>`, `&`, and several other characters that
need escaping in just about every textual transport. It's compact,
but you'll spend the bytes back on escaping.

### RFC 1924

Defined in an April Fools' RFC for compact IPv6 address text
representation, but the alphabet is genuinely useful: it
deliberately avoids `'`, `"`, `,`, `\`, `<`, `>`, `&`, and `` ` ``,
making the output safe to drop into JSON strings, XML attributes,
SQL string literals, and most quoted contexts without escaping.
This is what Python's `base64.b85encode` produces.

### Z85 (ZeroMQ)

ZeroMQ's variant, designed to be safe in shell-script contexts.
Avoids `'`, `"`, `\`, and the backtick. A common choice for
embedding binary keys in config files that may pass through shell
expansion. Note that Z85 requires the input length be a multiple
of 4; it does not pad.

## Worked examples

These are real round-trips. You can verify any of them with the
companion script (`python scripts/b85.py encode 'Hello'`) or
directly with the stdlib.

**Ascii85** (Python's `base64.a85encode`):

```
"easy"             ->  ARTY*               (4 bytes -> 5 chars)
"Hello"            ->  87cURDZ             (5 bytes -> 7 chars)
"sure."            ->  F*2M7/c             (5 bytes -> 7 chars)
"network"          ->  DImp6DfTU           (7 bytes -> 9 chars)
"Hello, World!"    ->  87cURD_*#4DfTZ)+T   (13 bytes -> 17 chars)
"Man is distinguished"
                   ->  9jqo^BlbD-BleB1DJ+*+F(f,q   (20 bytes -> 25 chars)
```

**RFC 1924** (Python's `base64.b85encode`):

```
"easy"             ->  Wnpu9
"Hello"            ->  NM&qnZv
"Hello, World!"    ->  NM&qnZ!92JZ*pv8Ap
```

Note that the RFC 1924 and Ascii85 outputs are the *same length* —
the alphabet differs, the ratio doesn't. If you see a base85-shaped
string and want to guess which variant: scan for `<`, `>`, `'`,
`"`. If any of those are present, it's Ascii85. If absent, it's
probably RFC 1924 or Z85.

## Length arithmetic

For an input of `n` bytes, the encoded length is `ceil(n / 4) * 5`,
**minus** the padding characters that get stripped off the tail.
A 4-byte-aligned input encodes to exactly `(n / 4) * 5` characters.
Otherwise:

- 1 leftover byte → 2 trailing characters (3 stripped)
- 2 leftover bytes → 3 trailing characters (2 stripped)
- 3 leftover bytes → 4 trailing characters (1 stripped)

So `"network"` (7 bytes) → `ceil(7/4)*5 = 10`, minus 1 because the
last group has 3 leftover bytes → 9 characters. That matches
`DImp6DfTU` above.

This is the source of the "wait, why is my encoded string the
length it is" confusion: base85 strips trailing padding markers
exactly as base64 does, but the math is less familiar. If you ever
see a base85 string whose length is not in the set
`{0, 2, 3, 4, 5, 7, 8, 9, 10, 12, …}` — i.e. not `5k`, `5k+2`,
`5k+3`, or `5k+4` — it's malformed.

## The `z` shorthand

In Adobe's Ascii85 (and Python's `a85encode`), four consecutive
zero bytes encode as the single character `z` instead of `!!!!!`.
This is a real space saving for sparse binary data — image
streams, padded buffers — and it's why Ascii85 outputs of
zero-heavy data can be much shorter than the length formula
suggests.

```
"\x00\x00\x00\x00"          ->  z              (4 bytes -> 1 char)
"\x00\x00\x00\x00abcd"      ->  z@:E_W         (8 bytes -> 6 chars)
```

RFC 1924 and Z85 do **not** have a `z` shorthand. The same input
under RFC 1924 encodes as `00000` and `00000VPa!s` respectively.
This is one of the easier ways to identify which variant you're
looking at: a `z` in the middle of an otherwise dense-looking
string, or a string that's much shorter than the length formula
predicts, points at Adobe Ascii85.

## Adobe delimiters

Adobe-wrapped Ascii85 looks like:

```
<~87cURD_*#4DfTZ)+T~>
```

The `<~` and `~>` are framing, not part of the payload. Python's
`a85decode` accepts them when you pass `adobe=True`; without that
flag it'll choke on the `<`. If you're handed a string that starts
with `<~` and ends with `~>`, strip them before decoding (or pass
the flag).

## When base85 is the right choice

- **PDF / PostScript streams.** The format calls for it; you don't
  get a vote. Use Ascii85 with Adobe delimiters.
- **Embedding binary blobs in source code** when the source file's
  size is a real concern. About 25% smaller than base64. Pick RFC
  1924 so you don't have to worry about quoting in your literals.
- **Compact IPv6 textual representation.** This is what RFC 1924
  was originally for. Almost nobody uses it for this in practice.
- **ZeroMQ wire format keys.** Z85, because that's what the
  protocol expects.

## When base85 is the wrong choice

- **HTTP, email, URLs.** Use base64 (URL-safe variant for URLs).
  base85's alphabet bites you on every escaping boundary.
- **User-visible contexts.** Users mistype it, copy-paste mangles
  the special characters, and any helpful auto-formatter is a
  threat to the payload.
- **Anything where the receiver might not know the variant.** If
  you can't confidently say "this consumer expects RFC 1924", pick
  base64. base85's variants look nearly identical and decode
  silently into garbage when the alphabet is wrong.

## Gotchas

- **Concatenation does not compose.** `b85encode(a) + b85encode(b)`
  is *not* `b85encode(a + b)` in general, because of the trailing
  padding strip on each piece. If you need to encode a stream,
  encode the whole stream at once or chunk on 4-byte boundaries.
- **Whitespace.** Most decoders ignore embedded whitespace
  (because line-wrapped Ascii85 is common in PDFs), but not all.
  If you control the format, don't insert whitespace; if you don't,
  strip it before decoding.
- **`z` outside of Ascii85.** A `z` in RFC 1924 output is just the
  letter z. Don't write a decoder that treats it as a shorthand
  unless you're decoding Ascii85 specifically.
- **Confusing the variants.** The fastest way to chase your tail
  for an hour is to assume "base85 is base85" and try to decode an
  RFC 1924 string with `a85decode`. It'll either error or, worse,
  silently produce bytes that look almost right.

## See also

- RFC 1924 (the IPv6 representation) — short and worth reading
  once, mostly for the alphabet table.
- Python stdlib: `base64.a85encode`, `base64.a85decode`,
  `base64.b85encode`, `base64.b85decode`. The stdlib is the
  reference implementation; do not roll your own.
- ZeroMQ's Z85 spec, if you need that variant specifically.
