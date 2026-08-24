"""The examples must keep working.

Example code is the fastest-rotting documentation there is: it is read more than
it is run, so a renamed function survives in it for releases. These tests do not
execute the examples — they need a real store and a real model — but they do
check every rebasis name each one references, which is where the rot happens.

The README tables are checked too. A documented flag that no longer exists sends
a reader to an error message instead of an answer.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(EXAMPLES.rglob("run.py"))
READMES = sorted(EXAMPLES.rglob("README.md"))


def rebasis_imports(source: str) -> list[tuple[str, str]]:
    """Every ``from rebasis... import name`` in a file, as (module, name)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rebasis"):
            found.extend((node.module or "", alias.name) for alias in node.names)
    return found


def test_there_are_examples() -> None:
    """A guard on the guard: an empty glob would make every test below vacuous."""
    assert SCRIPTS
    assert READMES


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_an_example_parses(script: Path) -> None:
    ast.parse(script.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_every_name_an_example_imports_exists(script: Path) -> None:
    """The check that actually catches a rename."""
    for module_name, name in rebasis_imports(script.read_text(encoding="utf-8")):
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"{module_name} has no {name!r} — {script.parent.name}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.parent.name)
def test_an_example_has_a_main(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    assert "main" in functions


@pytest.mark.parametrize("readme", READMES, ids=lambda p: p.parent.name or "index")
def test_the_cli_flags_a_readme_documents_are_real(readme: Path) -> None:
    """A documented flag that no longer exists is worse than no documentation.

    Only long options are checked. Matching bare words would flag every hyphen
    in prose, and matching short options would flag ordinary punctuation.
    """
    from typer.main import get_command

    from rebasis.cli import app

    command = get_command(app)
    known = {"--help"}
    for name, sub in getattr(command, "commands", {}).items():
        known.add(name)
        for parameter in sub.params:
            known.update(o for o in getattr(parameter, "opts", []) if o.startswith("--"))

    text = readme.read_text(encoding="utf-8")
    # Only inside fenced blocks that invoke rebasis: prose mentions options in
    # passing, and a table is not a command line.
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.DOTALL):
        if "rebasis " not in block:
            continue
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", block):
            # Environment-variable exports and docker arguments share the syntax.
            if flag in {"--rm", "--with", "--isolated"}:
                continue
            assert flag in known, f"{readme.parent.name}/README.md documents {flag}, which is gone"
