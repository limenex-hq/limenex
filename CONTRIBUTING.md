# Contributing to Limenex

Limenex is an open-source project and we welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code. Thank you for helping make Limenex better.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/limenex-hq/limenex.git
   cd limenex
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the package in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run the test suite to verify your setup:
   ```bash
   pytest
   ```

## How to Contribute

1. Fork the repository on GitHub.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes.
4. Ensure all tests pass and code style is clean:
   ```bash
   pytest
   black --check .
   ruff check .
   ```
5. Open a pull request against `main` with a clear description of your changes.

## Adding a Skill

Refer to the skills contribution guide in `docs/` (to be written) for instructions on how to create and register new execution skills.

## Code Style

- **black** for code formatting
- **ruff** for linting
- Line length: **88** characters
- All style rules are enforced by CI — please ensure your code passes before opening a PR.

## Reporting Issues

Please use [GitHub Issues](https://github.com/limenex-hq/limenex/issues) to report bugs. When filing an issue, include:

- Your Python version
- Your operating system
- A minimal reproducible example
