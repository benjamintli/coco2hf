# GitHub Actions Workflows

This repository uses GitHub Actions for continuous integration.

## Workflows

### Test & Lint (`test-lint.yml`)
Runs on every push and pull request to `main`.

**What it does:**
- Tests on Python 3.12 and 3.13
- Runs `ruff check` to lint the code
- Runs `ruff format --check` to ensure code is properly formatted

### CLI Smoke Test (`cli-smoke-test.yml`)
Runs on every push and pull request to `main`.

**What it does:**
- Installs the package
- Tests that the `coco2hf` CLI command works
- Verifies all main imports are functional
- Displays Python version info

## Running Locally

Before pushing, you can run these checks locally:

```bash
# Lint
ruff check .

# Format check
ruff format --check .

# Auto-format
ruff format .

# Test CLI
coco2hf --help
```
