"""Nox automation file."""

import platform
import shutil
from pathlib import Path

from nox import Session, session

python_versions = ["3.10", "3.9", "3.8", "3.7"]


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    if platform.system() == "Windows":
        session.install(
            "jax[cpu]===0.3.14",
            "-f",
            "https://whls.blob.core.windows.net/unstable/index.html",
            "--use-deprecated",
            "legacy-resolver",
        )
    session.install(".")
    session.install(
        "dask",
        "pytest",
        "pytest-benchmark",
        "pytest-datadir",
        "pytest-cov",
        "xdoctest",
    )
    session.run("pytest", "--runslow")


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or [
        "-a",
        "-E",
        "--open-browser",
        "docs",
        "docs/_build",
        "--watch",
        "src",
    ]
    session.install("-e", ".")
    session.install("-r", "docs/requirements.txt")
    session.install("sphinx", "sphinx-autobuild")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
