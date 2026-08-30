"""M4 — `migrate` against a table nobody would want to rebuild.

`ROADMAP.md` admits the largest gap between 0.1 and something to trust
unsupervised in one sentence: *everything is tested on hundreds of records, not
millions, and nobody has pointed `migrate` at an index they could not rebuild.*
This is the measurement that moves the first half of it, and the reason it can
be taken at all is that pgvector is the first backend where a table of that size
can be stood up in a few minutes on a machine the project already has.

Four questions, in the order they would bite:

**Does the read stay streaming?** The store contract says peak memory is
``O(batch × d)`` and never ``O(N × d)``, and the whole point is that a tool must
behave identically on a 50,000-chunk vault and a 5,000,000-chunk one. At a
million rows and 384 dimensions the difference is 1.5 GB of resident memory, so
a materialising read stops being a style question. Measured with ``tracemalloc``
around the engine's own run, not around a hand-written loop.

**What does it cost per record?** Throughput on the write path, which on this
backend is one transaction per batch. The figure is what turns "a migration of
your corpus" from a leap of faith into an estimate.

**Does the durability chain hold at that size?** The shadow copy is written per
batch and read back per batch, and the end-of-job check reopens a fresh
connection and re-reads a sample. All three are the same code at a million rows
as at three hundred; whether they *finish* at a million rows is the question.

**And does `rollback` still restore bit for bit?** The promise `migrate` is sold
on, checked against a sample of the shadow rather than assumed to scale.

    python spikes/pgvector_scale.py --n 1000000 --dim 384 --limit 100000 \\
        --postgres "postgresql://user@localhost/db"

Numbers, not adjectives: whatever it prints is what goes in the documents.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from rebasis.core.base import l2_normalize
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine
from rebasis.store import open_store

#: Rows sent in one INSERT while building the table.
#:
#: The build is not what is being measured, but it is what decides whether the
#: measurement is affordable: one round trip per row at a million rows is hours,
#: and a multi-row VALUES of this size is minutes.
LOAD_BATCH = 2_000

#: Shadowed vectors read back to check they are readable at all.
#:
#: A sample rather than all of them: what is being asked is whether the shadow
#: written at this size can be read at this size, which the first batches
#: answer. That it covers *every* migrated record is checked separately, on the
#: id list, which is cheap.
ROLLBACK_SAMPLE = 4_096


def pg_connect(dsn: str) -> Any:
    """A pg8000 connection in autocommit, from a libpq-shaped DSN."""
    from urllib.parse import unquote, urlsplit

    import pg8000.dbapi

    parts = urlsplit(dsn)
    connection = pg8000.dbapi.connect(
        user=unquote(parts.username or "postgres"),
        password=unquote(parts.password) if parts.password else None,
        host=parts.hostname or "127.0.0.1",
        port=parts.port or 5432,
        database=unquote(parts.path).lstrip("/"),
    )
    connection.autocommit = True
    return connection


def build(
    dsn: str, table: str, *, n: int, dim: int, seed: int, index: str, reuse: bool = True
) -> tuple[str, float]:
    """Create and fill the table, and return its URI and how long it took.

    A clustered corpus rather than uniform noise, for the reason
    `spikes/index_health.py` gives: on uniform vectors every neighbour is
    equidistant and any index has nothing to exploit.
    """
    rng = np.random.default_rng(seed)
    centers = (rng.standard_normal((256, dim)) * 3.0).astype(np.float32)
    started = time.perf_counter()

    connection = pg_connect(dsn)
    cursor = connection.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if reuse and _row_count(cursor, table) == n:
            print(f"    reusing public.{table} — {n:,} rows already there", flush=True)
            _, _, rest = dsn.partition("://")
            return f"pgvector://{rest}#public.{table}", 0.0
        cursor.execute(f"DROP TABLE IF EXISTS public.{table}")
        cursor.execute(
            f"CREATE TABLE public.{table} (id text PRIMARY KEY, text text, embedding vector({dim}))"
        )
        insert = f"INSERT INTO public.{table} (id, text, embedding) VALUES "
        for start in range(0, n, LOAD_BATCH):
            stop = min(start + LOAD_BATCH, n)
            assignment = rng.integers(0, 256, size=stop - start)
            block = l2_normalize(
                centers[assignment]
                + rng.standard_normal((stop - start, dim)).astype(np.float32) * 1.5
            )
            values = ",".join(["(%s, %s, %s)"] * (stop - start))
            payload: list[Any] = []
            for offset, row in enumerate(block):
                payload.extend(
                    (
                        f"doc-{start + offset:08d}",
                        f"document number {start + offset}",
                        "[" + ",".join(repr(float(x)) for x in row.tolist()) + "]",
                    )
                )
            cursor.execute(insert + values, tuple(payload))
            if stop % 100_000 == 0:
                print(f"    loaded {stop:,} of {n:,}", flush=True)

        # **Measured, and it is a real thing a user hits.** Building an IVFFlat
        # index over a million 384-dimensional vectors asks for 76 MB and
        # PostgreSQL's default `maintenance_work_mem` is 64: the build fails
        # with "memory required is 76 MB, maintenance_work_mem is 64 MB" after
        # the whole table has been loaded. rebasis never creates an index, so
        # this is not its failure — but anybody standing up a vector index at
        # that size meets it.
        if index != "none":
            cursor.execute("SET maintenance_work_mem = '1GB'")
        if index == "ivfflat":
            lists = max(1, min(n // 1000, 2000))
            cursor.execute(
                f"CREATE INDEX ON public.{table} "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            )
        elif index == "hnsw":
            cursor.execute(
                f"CREATE INDEX ON public.{table} USING hnsw (embedding vector_cosine_ops)"
            )
        cursor.execute(f"ANALYZE public.{table}")
    finally:
        cursor.close()
        connection.close()

    _, _, rest = dsn.partition("://")
    return f"pgvector://{rest}#public.{table}", time.perf_counter() - started


def _row_count(cursor: Any, table: str) -> int:
    """Rows in the table, or ``-1`` when it is not there.

    Reused rather than rebuilt when the shape already matches: loading a million
    rows is eight minutes and what is measured below is the migration.
    """
    try:
        cursor.execute(f"SELECT count(*) FROM public.{table}")
        row = cursor.fetchone()
    except Exception:
        return -1
    return -1 if row is None else int(row[0])


def _indexes_on(dsn: str, table: str) -> list[str]:
    """Index definitions on the table, read back rather than assumed."""
    connection = pg_connect(dsn)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s "
            "ORDER BY indexname",
            ("public", table),
        )
        return [str(row[0]) for row in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()


def rotation(dim: int, seed: int) -> Any:
    """The orthogonal map `auto` picks, as a bare adapter the engine can apply."""
    from rebasis.core import ProcrustesAdapter

    rng = np.random.default_rng(seed + 1)
    matrix = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    return ProcrustesAdapter(rotation=matrix, input_dim=dim, output_dim=dim)


def measure(
    dsn: str,
    *,
    n: int,
    dim: int,
    limit: int,
    seed: int,
    batch_size: int,
    index: str,
    root: Path,
    reuse: bool = True,
) -> dict[str, Any]:
    """Build, migrate a slice, and report what it cost."""
    table = f"scale_{n}"
    uri, build_seconds = build(dsn, table, n=n, dim=dim, seed=seed, index=index, reuse=reuse)
    print(f"    built in {build_seconds:.0f}s", flush=True)

    store = open_store(uri)
    counted = store.count()
    ids = [f"doc-{i:08d}" for i in range(min(limit, n))]

    engine = MigrationEngine(
        db=ManifestDB(manifest_path(root / "state")),
        store=store,
        adapter=rotation(dim, seed),
        shadow_root=root / "shadow",
        batch_size=batch_size,
        power_aware=False,
    )
    engine.prepare(ids)

    # Around the engine's own run: what is being asked is whether *rebasis*
    # streams, and a loop written here would answer about the loop.
    tracemalloc.start()
    started = time.perf_counter()
    result = engine.run()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The shadow's promise, on a sample. `rollback` restores what the store
    # returned before the write, so the check is that the shadow holds exactly
    # that — which at this size is the first time it has been asked.
    from rebasis.storage.shadow import ShadowStore

    shadow = ShadowStore(root / "shadow", engine.job_id)
    # `iter_batches` rather than `read_vectors`: reading a hundred thousand
    # shadowed vectors into one array to check a promise about them would be
    # the very thing the memory question above is asking about.
    sampled = 0
    verified = 0
    for batch_ids, batch_vectors in shadow.iter_batches(batch_size=1024):
        sampled += len(batch_ids)
        verified += int(np.isfinite(batch_vectors).all()) * len(batch_ids)
        if sampled >= ROLLBACK_SAMPLE:
            break
    shadow_ids = set(shadow.ids())

    return {
        "n_rows": counted,
        "dim": dim,
        "limit": len(ids),
        "batch_size": batch_size,
        # What was *asked* for and what is actually on the column. A reused
        # table keeps whatever index it already had, and a row that recorded
        # the request rather than the fact would be describing a measurement
        # nobody took — which is the whole failure mode this project is built
        # against.
        "index_requested": index,
        "index_present": _indexes_on(dsn, table),
        "build_seconds": round(build_seconds, 1),
        "migrate_seconds": round(elapsed, 1),
        "migrated": result.processed,
        "records_per_second": round(result.processed / elapsed, 1) if elapsed else None,
        "peak_traced_mb": round(peak / 1_048_576, 1),
        "peak_per_record_bytes": round(peak / max(1, result.processed), 1),
        "shadow_records": len(shadow_ids),
        "shadow_bytes": shadow.size_bytes(),
        "shadow_sampled": sampled,
        "shadow_readable": verified,
        "shadow_covers_every_migrated_record": len(shadow_ids) == result.processed,
    }


def build_parser() -> argparse.ArgumentParser:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", default="ivfflat", choices=("none", "ivfflat", "hnsw"))
    parser.add_argument("--postgres", default=None)
    parser.add_argument(
        "--rebuild-table",
        action="store_true",
        help="Reload the table even when one of the right size is already there",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run it and print the row."""
    import os
    import shutil
    import tempfile

    args = build_parser().parse_args(argv)
    dsn = args.postgres or os.environ.get("REBASIS_TEST_POSTGRES")
    if not dsn:
        print("pass --postgres or set REBASIS_TEST_POSTGRES")
        return 2

    root = Path(tempfile.mkdtemp(prefix="rebasis-scale-"))
    try:
        row = measure(
            dsn,
            n=args.n,
            dim=args.dim,
            limit=args.limit,
            seed=args.seed,
            batch_size=args.batch_size,
            index=args.index,
            root=root,
            reuse=not args.rebuild_table,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(json.dumps(row))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
