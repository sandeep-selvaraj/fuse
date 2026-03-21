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
    """Run pytest test suite."""
    session.install(".[dev]")
    session.run("pytest", "tests/", "-v", *session.posargs)
