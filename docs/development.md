---
icon: lucide/code
---

# Development

## Setup

Clone the repo and install all dependencies:

```bash
git clone https://github.com/sandeep-selvaraj/fuse.git
cd fuse
uv sync --all-extras
```

---

## Running checks

Fuse uses [nox](https://nox.thea.codes/) for CI task management:

```bash
uv run nox                    # All CI checks
uv run nox -s lint            # Ruff lint + format check
uv run nox -s typecheck       # ty type check
uv run nox -s tests           # Pytest across Python 3.11-3.13
```

### Individual tools

```bash
uv run ruff check src/ tests/        # Lint only
uv run ruff format src/ tests/       # Auto-format
uv run pytest tests/                 # Tests only
uv run fuse --help                   # CLI
```

---

## Key conventions

- **uv** for package management, **ruff** for linting, **ty** for type checking
- All inference backends implement the `InferenceBackend` protocol — the extraction layer never imports a concrete backend directly
- Training dependencies are optional (`fusellm[training]`)
- Heavy imports (torch, transformers, llama_cpp, outlines) are lazy-loaded to keep CLI startup fast
- Versioning is automatic via git tags (`hatch-vcs`)

---

## Versioning

Fuse uses semantic versioning derived from git tags via [hatch-vcs](https://github.com/ofek/hatch-vcs):

```bash
# Tag a release
git tag v0.1.0
git push --tags

# Dev versions are auto-generated from commits since last tag
# e.g., 0.1.1.dev3+g1a2b3c4
```

---

## Running tests

```bash
# All Python versions
uv run nox -s tests

# Specific Python version
uv run nox -s tests-3.13

# With pytest flags
uv run pytest tests/ -v -k test_schema
```

---

## CI/CD

GitHub Actions runs on every push and PR:

- **Lint**: ruff check + format
- **Type check**: ty
- **Tests**: pytest across Python 3.11, 3.12, 3.13
- **Docs**: zensical build + deploy to GitHub Pages (on push to master)
