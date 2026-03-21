import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff linter."""
    session.install("ruff>=0.4")
    session.run("ruff", "check", "src/", "tests/")
    session.run("ruff", "format", "--check", "src/", "tests/")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run ty type checker."""
    session.install(".", "ty")
    session.run("ty", "check", "src/")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run unit tests (excludes integration tests)."""
    session.install(".[dev]")
    session.run(
        "pytest", "tests/", "-v",
        "--ignore=tests/integration",
        *session.posargs,
    )


@nox.session
def integration(session: nox.Session) -> None:
    """Run integration tests (downloads models, slow).

    Usage:
        uv run nox -s integration                    # all integration tests
        uv run nox -s integration -- -k inference     # inference only
        uv run nox -s integration -- -k training      # training only
    """
    session.install(".[dev,training]")
    session.run(
        "pytest", "tests/integration/", "-v",
        "--tb=short",
        *session.posargs,
    )
