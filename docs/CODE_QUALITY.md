# Code Quality

Automated code formatting and linting using ruff with pre-commit hooks.

## Setup

Install once:

```bash
pip install pre-commit ruff
pre-commit install
```

## How It Works

After setup, every `git commit` automatically:

1. Formats code (spacing, quotes, import order)
2. Checks for common bugs (unused variables, syntax errors)
3. If changes were made, commit pauses so you can review

To commit anyway (skip checks):

```bash
git commit --no-verify
```

## Manual Usage

Format all files:

```bash
ruff format src/
```

Check for issues:

```bash
ruff check src/
```

Fix auto-fixable issues:

```bash
ruff check --fix src/
```

## Configuration

Settings are in `pyproject.toml` under `[tool.ruff]`.

### What ruff checks

Ruff enforces rules from these categories:
- **E (pycodestyle)**: Basic style like spacing and indentation
- **F (pyflakes)**: Bugs like undefined variables
- **W (warnings)**: Minor style issues
- **B (bugbear)**: Likely bugs like mutable default arguments
- **UP (pyupgrade)**: Suggests modern Python syntax

### Current settings

- **Line length**: 150 characters max per line
- **Python version**: 3.11 syntax rules
- **Quote style**: Double quotes (`"string"` not `'string'`)

### Ignored rules

Some rules are disabled because they conflict with this codebase:

- **Long lines (E501)**: Some lines exceed 150 chars, formatter handles flexibly
- **Single-letter variables (E741)**: We use `n`, `s`, `G`, `q` intentionally (standard math/graph notation)
- **Undefined names (F821)**: `priority` and `start_n` are injected at runtime in evaluation scripts
- **Mutable default args (B006)**: Used intentionally in some places, not causing bugs
