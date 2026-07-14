# Contributing

Thanks for your interest in improving the T-Search harness.

## Development setup

```bash
git clone https://github.com/turbo-llm/t-search-harness
cd t-search-harness
poetry install
```

## Lint and tests

```bash
make lint    # ruff check + ruff format --check + mypy
make pretty  # auto-format (ruff check --fix + ruff format)
make test    # pytest
```

## Before opening a pull request

- Changes to prompts, tool schemas, gates, or thresholds alter model behavior —
  make them deliberately and explain the intent in the PR.
- Match the existing code style; keep comments short.
- Add or update tests for the change.
- Describe what changed and why in the PR, and link any related issue.

## Reporting bugs / requesting features

Open an issue using the templates under `.github/ISSUE_TEMPLATE/`.
