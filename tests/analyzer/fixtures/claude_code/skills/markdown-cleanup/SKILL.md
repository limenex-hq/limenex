---
name: markdown-cleanup
description: Tidies up Markdown files in a directory by collapsing extra blank lines, trimming trailing whitespace, normalising heading spacing, renumbering ordered lists, and ensuring a single trailing newline. Use before committing a batch of docs the team has been editing in parallel, or as a preflight step in a docs-heavy repo where the formatting has drifted.
license: MIT
---

# markdown-cleanup

A small in-place tidier for Markdown files. The kind of thing
`prettier --write '**/*.md'` does, but with a smaller and more
opinionated set of rules and no Node dependency.

## What it normalises

For every `.md` file under the target directory:

1. **Trailing whitespace** is stripped from every line. Trailing
   spaces are invisible and they cause noise in diffs.
2. **Runs of blank lines** of three or more are collapsed to two.
   One blank line between paragraphs is fine, two is fine for
   visual separation between sections, three or more is just
   accidental.
3. **Heading spacing** is normalised: `#Heading` becomes
   `# Heading`, `##  Heading` becomes `## Heading`. Exactly one
   space between the marker and the text.
4. **Ordered lists** that are all `1.` `1.` `1.` (the convention
   some authors use because Markdown will renumber them anyway)
   are rewritten as `1.` `2.` `3.`. The result reads better in the
   raw source and it makes diffs more meaningful when items are
   added or removed.
5. **Trailing newlines** are normalised to exactly one. Files
   with no trailing newline get one; files with three trailing
   newlines lose two.

That's it. It's a small set on purpose — the rules above are the
ones whose right answer is unambiguous. Anything subjective (line
length, list-marker style, code-fence language tags) is left
alone.

## What it skips

- Files whose extension isn't `.md`. The walk only picks up
  `*.md`; `.markdown`, `.mdx`, `.mdown` are ignored. Add an alias
  yourself if you need them.
- Files inside a `.git/` directory anywhere under the target.
  Walking into `.git` would be bad, mostly because there are no
  Markdown files in there worth cleaning and a few pack files
  that absolutely must not be touched.
- Files that are byte-for-byte unchanged after normalisation. The
  script computes the new content first, compares to the old,
  and only writes if there's a diff. This keeps the mtime stable
  on already-clean files, which keeps `make` and friends happy.

## Instructions

1. Run `python scripts/cleanup.py <directory>`. If you don't pass
   a directory, it uses the current working directory.
2. Skim `git status` afterward. The set of touched files should
   match your expectation; if a file you didn't expect to change
   shows up, open the diff and check why.
3. Commit the cleanup as its own commit, separate from any
   content changes. `chore(docs): cleanup` or similar. Mixing
   formatting changes into a content commit makes review harder.

## Safety

The script only modifies the content of `.md` files. It does not
move, rename, or remove files. If a normalisation pass would
result in no changes to a file, the file is left untouched on
disk. The intent is that running this twice in a row is a no-op
and that running it on a clean repo produces no changes at all.

If you want to preview the changes before committing them, run
the script and then `git diff` — the script writes in place, but
git is the undo button. Don't run this against a directory whose
contents aren't tracked in version control.

## When this doesn't apply

- Repos that already use `prettier`, `mdformat`, or
  `markdownlint --fix`. Use the existing tool; don't introduce a
  second formatter that disagrees with the first.
- Files with embedded HTML where whitespace is significant. The
  trailing-whitespace strip is fine but if you have `<pre>`-style
  blocks where leading whitespace matters, double-check the diff.
- Generated Markdown (e.g. API reference output). Run the
  generator on those, not this.
