"""Probe an Obsidian vault indexed in Chroma, from Python.

The CLI does the same thing in one line. This exists for the case where the
decision feeds something else — a scheduled job that re-checks quarterly, or a
script that fits the adapter automatically when the answer is good enough.

    uv run --with 'rebasis[chroma,sentence-transformers]' examples/obsidian_vault/run.py \\
        --store chroma:///Users/me/vault/.chroma#notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEW_MODEL = "BAAI/bge-base-en-v1.5"

#: Above this, fitting an adapter is worth doing. Below it, the honest answer is
#: to reindex or to stay put — see docs/concepts/decision-rule.md.
GOOD_ENOUGH = {"bridge_sufficient", "bridge_and_migrate"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="chroma:///path/to/db#collection")
    parser.add_argument("--old", default=OLD_MODEL)
    parser.add_argument("--new", default=NEW_MODEL)
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--queries", type=Path, help="JSONL query log; strongly preferred")
    parser.add_argument("--report", type=Path, default=Path("vault-report.html"))
    parser.add_argument("--fit-to", type=Path, help="fit and save if the answer is good enough")
    args = parser.parse_args()

    from rebasis.cli._pipeline import load_query_log
    from rebasis.embed import open_embedder
    from rebasis.probe import probe_store
    from rebasis.report import render_html
    from rebasis.store import open_store

    query_log = load_query_log(args.queries) if args.queries else None
    store = open_store(args.store)

    result, _ = probe_store(
        store,
        open_embedder(args.new),
        # Only needed at T1: with document proxies, the old model's vectors are
        # already in the index and do not have to be recomputed.
        old_embedder=open_embedder(args.old) if query_log else None,
        query_log=query_log,
        size=args.sample,
    )

    args.report.write_text(render_html(result, store_uri=args.store), encoding="utf-8")

    decision = result.decision
    low, high = result.best.arr_ci
    print(f"{decision.decision}: {decision.rationale}")
    print(f"  ARR@{result.k} {result.best.arr:.3f}  (95% CI {low:.3f}-{high:.3f})")
    print(f"  adapter {result.best.name}")
    print(f"  report  {args.report}")

    if decision.borderline:
        print("  borderline: the interval spans a decision boundary. Raise --sample.")

    if args.fit_to and decision.decision in GOOD_ENOUGH:
        from rebasis.core import save_adapter
        from rebasis.embed import profile_for

        written = save_adapter(
            result.adapter,
            args.fit_to,
            direction="query_to_old",
            # The index is authoritative about the old model's dimension, so an
            # unregistered model id needs no --dim here.
            old_profile=profile_for(args.old, dim=store.dimension()),
            new_profile=open_embedder(args.new).profile,
            calibrator=result.calibrator,
            evaluation=result.to_dict(),
        )
        print(f"  adapter written to {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
