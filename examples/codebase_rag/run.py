"""Probe a code index in LanceDB with a real query log.

The point of this example over the vault one is the query log: with real
questions and the chunks that answered them, `probe` can tell you whether the
new model is actually better on your repository — which is the question that
matters, and the one document proxies cannot answer.

    uv run --with 'rebasis[lancedb,sentence-transformers]' examples/codebase_rag/run.py \\
        --store lancedb:///data/code-index#chunks --queries dev-queries.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD_MODEL = "sentence-transformers/all-mpnet-base-v2"
NEW_MODEL = "BAAI/bge-base-en-v1.5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="lancedb:///path/to/db#table")
    parser.add_argument("--old", default=OLD_MODEL)
    parser.add_argument("--new", default=NEW_MODEL)
    parser.add_argument("--queries", type=Path, required=True, help="JSONL query log")
    parser.add_argument("--sample", type=int, default=12000)
    parser.add_argument("--report", type=Path, default=Path("code-report.md"))
    args = parser.parse_args()

    from rebasis.cli._pipeline import load_query_log
    from rebasis.embed import open_embedder
    from rebasis.probe import probe_store
    from rebasis.report import render_markdown
    from rebasis.store import open_store

    query_log = load_query_log(args.queries)
    print(f"{len(query_log)} queries with judgements")

    result, _ = probe_store(
        open_store(args.store),
        open_embedder(args.new),
        old_embedder=open_embedder(args.old),
        query_log=query_log,
        size=args.sample,
    )

    args.report.write_text(render_markdown(result, store_uri=args.store), encoding="utf-8")

    decision = result.decision
    low, high = result.best.arr_ci
    print(f"\n{decision.decision}: {decision.rationale}")
    print(f"  ARR@{result.k} {result.best.arr:.3f}  (95% CI {low:.3f}-{high:.3f})")

    # Only available at T1, and the number that decides whether any of this is
    # worth doing: below ~1.02 the new model is not better on this corpus.
    if decision.upgrade_gain is not None:
        print(f"  upgrade gain {decision.upgrade_gain:.3f}")
        if decision.decision == "no_upgrade_needed":
            print("  The candidate model is not better on this repository.")
            print("  The cheapest upgrade is the one you skip.")
            return 0

    # A code corpus is heterogeneous: tests, generated code and business logic
    # live in different regions, and one global adapter can do well on average
    # while doing badly on one of them.
    for warning in decision.warnings:
        print(f"  ! {warning}")

    print("\nAlternatives:")
    for name, value in sorted(result.baselines.items()):
        print(f"  {name.replace('_', ' '):<16} {value:.3f}")

    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
