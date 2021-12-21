"""Nox automation file."""

import nox
from nox.sessions import Session

python_versions = ["3.7", "3.8", "3.9", "3.10"]


@nox.session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    pass


@nox.session
def lint(session: Session) -> None:
    """Run linting."""
    pass


@nox.session
def coverage(session: Session) -> None:
    """Run coverage."""
    pass
