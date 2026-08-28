"""Check the per-module coverage floors against a finished coverage run.

`coverage.py` enforces one number. The design asks for more than one: the
modules where a bug costs *data* rather than accuracy are held to a higher bar
than the average.

This is a script rather than a test on purpose. A test that reads
`coverage.json` runs *during* the session that writes it, so it can only ever
see the previous run's file — it would pass while measuring something that is no
longer true. Run after the suite:

    pytest --cov=rebasis --cov-report=json:reports/coverage.json
    python tools/check_coverage_floors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Individual modules, as line coverage.
FLOORS: dict[str, int] = {
    "src/rebasis/errors.py": 100,
    "src/rebasis/types.py": 100,
    "src/rebasis/observability/redaction.py": 100,
    # The chain is what makes the audit trail tamper-evident. A branch nobody
    # exercises is a branch nobody trusts.
    "src/rebasis/audit/chain.py": 100,
    "src/rebasis/compute/numpy_backend.py": 90,
    # These two write and restore the bytes a rollback depends on. ROADMAP.md
    # named 95 and 100 as their targets and neither was a floor here, so the two
    # modules were held only by the `storage/` package floor of 80 — a target
    # stated in prose and enforced nowhere is not a target. 90 is a ratchet, not
    # the goal: it is under the measured 93.5% and 93.3% so it cannot go red on
    # arrival, and it stops either one sliding back while the goal is unmet.
    "src/rebasis/storage/shadow.py": 90,
    "src/rebasis/storage/atomic.py": 90,
}

#: Package floors, summed over their files.
PACKAGE_FLOORS: dict[str, int] = {
    "src/rebasis/core/": 90,
    "src/rebasis/probe/": 80,
    "src/rebasis/audit/": 85,
    "src/rebasis/storage/": 80,
    # Both of these decide what a *query* returns. `serve` holds the two
    # arrangements a user can put in front of a live index; `migrate/spaces.py`
    # decides whether the index is in one embedding space or two, and the whole
    # protection against a mixed index is that somebody is told. A branch nobody
    # exercises there is a branch that answers a query wrongly and silently,
    # which is the same class of failure the audit chain's floor exists for.
    "src/rebasis/serve/": 85,
    "src/rebasis/migrate/": 80,
}

REPORT = Path("reports/coverage.json")


def main() -> int:
    if not REPORT.exists():
        print(f"{REPORT} is missing — run the suite with --cov first.", file=sys.stderr)
        return 2

    files = json.loads(REPORT.read_text(encoding="utf-8"))["files"]
    failures: list[str] = []

    for path, floor in FLOORS.items():
        summary = files.get(path, {}).get("summary")
        if summary is None:
            failures.append(f"{path}: not in the coverage report")
            continue
        percent = summary["percent_covered"]
        mark = "ok" if percent >= floor else "BELOW"
        print(f"  {path:<44}{percent:6.1f}%  floor {floor:>3}  {mark}")
        if percent < floor:
            failures.append(f"{path}: {percent:.1f}% against a floor of {floor}")

    for prefix, floor in PACKAGE_FLOORS.items():
        covered = sum(
            v["summary"]["covered_lines"] for p, v in files.items() if p.startswith(prefix)
        )
        total = sum(
            v["summary"]["num_statements"] for p, v in files.items() if p.startswith(prefix)
        )
        if not total:
            continue
        percent = covered * 100 / total
        mark = "ok" if percent >= floor else "BELOW"
        print(f"  {prefix:<44}{percent:6.1f}%  floor {floor:>3}  {mark}")
        if percent < floor:
            failures.append(f"{prefix}: {percent:.1f}% against a floor of {floor}")

    if failures:
        print("\nbelow their floor:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
