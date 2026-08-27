# rebasis — developer commands
# Install just:  brew install just

# Recipes that drive the maintainer's GPU host live in justfile.local, which is
# gitignored along with the scripts/ they call. The import is optional: in a
# clone the file is absent, `just --list` simply does not show them, and nothing
# here fails. Every recipe below runs from a fresh clone.
import? 'justfile.local'

default:
    @just --list

# ── local: fast loop, target 30 seconds ─────────────────────────────
sync:
    uv sync

# Everything CI enforces, runnable before every commit
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    uv run lint-imports
    uv run pytest -m "unit or property"

fmt:
    uv run ruff check --fix .
    uv run ruff format .

test:
    uv run pytest -m "not gpu and not slow and not network and not perf and not memory"

test-all:
    uv run pytest -m ""

# Everything CI gates on that the fast loop skips. The memory layer asserts peak
# allocation, which a shared runner measures exactly — so unlike `bench` it is on
# the merge path, and this is how to see it go red before pushing rather than
# after. It builds corpora up to 400,000 records; expect it to take a minute.
gate:
    uv run pytest -m memory -q

# Wall-clock benchmarks. These never gate a merge — a shared runner cannot
# measure timing — so their numbers only mean something on a machine whose
# specification is recorded next to them.
bench:
    uv run pytest -m perf -q

cov:
    uv run pytest --cov=rebasis --cov-report=term-missing --cov-report=html

# Verify the package works without torch installed (CI "no-torch" job)
test-no-torch:
    uv run --isolated --no-group spike pytest -m "unit or contract"

hooks:
    uv run pre-commit install --install-hooks --hook-type commit-msg

# ── documentation ────────────────────────────────────────────────────
# Generated reference pages — never hand-edited
docs-gen:
    uv run python -m rebasis.report.catalog --out docs/reference

# The site. Strict: a dead link is a build failure, not a warning.
docs:
    uv run --group docs mkdocs serve

docs-build: docs-gen
    uv run --group docs mkdocs build

# Preview the changelog without writing it
changelog:
    uv run towncrier build --draft --version 0.0.0
