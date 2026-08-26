"""The whole tool, through the commands a user actually types.

Every other test reaches into a function. This one goes through argument
parsing, store opening, embedder construction, the pipeline, the report writer
and the audit trail — because that is where the seams are, and a seam that is
only ever crossed by a unit test is a seam that has never been crossed.

Vectors are precomputed and the store is an ``.npz`` export, so the flow runs
with no model download and no network.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.core import l2_normalize
from rebasis.errors import EXIT_OK, EXIT_USAGE
from rebasis.store import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.e2e

runner = CliRunner()

DIM = 40
N_DOCS = 800


@pytest.fixture
def corpus(tmp_path, rng):  # type: ignore[no-untyped-def]
    """A store on disk, plus a profiles file naming its two models.

    The new model's vectors are a rotation of the old ones — a different space
    that preserves the neighbour structure, which is the situation an adapter
    exists for.
    """
    centers = (rng.standard_normal((25, DIM)) * 3.0).astype(np.float32)
    assignment = rng.integers(0, 25, size=N_DOCS)
    old = l2_normalize(
        centers[assignment] + rng.standard_normal((N_DOCS, DIM)).astype(np.float32) * 1.5
    )
    rotation = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0].astype(np.float32)
    new = l2_normalize(old @ rotation.T)

    ids = [f"doc-{i:04d}" for i in range(N_DOCS)]
    texts = [f"document number {i}" for i in range(N_DOCS)]
    store_path = MemoryStore(ids, old, texts).save(tmp_path / "corpus.npz")

    vectors_path = tmp_path / "new_vectors.npz"
    np.savez(vectors_path, texts=np.array(texts), vectors=new)

    return {
        "uri": f"memory://{store_path}",
        "state": tmp_path / "state",
        "texts": texts,
        "old": old,
        "new": new,
        "ids": ids,
        "tmp": tmp_path,
    }


@pytest.fixture(autouse=True)
def _precomputed_embedders(corpus, monkeypatch):  # type: ignore[no-untyped-def]
    """Serve the fixture's vectors wherever the CLI asks for an embedder.

    Patched at the registry rather than the CLI so the command path under test
    is the real one, right down to `open_embedder`.
    """
    from rebasis.embed import PrecomputedEmbedder
    from rebasis.types import EncodingProfile

    tables = {
        "old-model": dict(zip(corpus["texts"], corpus["old"], strict=True)),
        "new-model": dict(zip(corpus["texts"], corpus["new"], strict=True)),
    }

    def fake_open(model_id: str, **kwargs: object) -> PrecomputedEmbedder:
        del kwargs
        return PrecomputedEmbedder(EncodingProfile(model_id=model_id, dim=DIM), tables[model_id])

    monkeypatch.setattr("rebasis.embed.registry.open_embedder", fake_open)
    monkeypatch.setattr("rebasis.embed.open_embedder", fake_open, raising=False)


def forward_adapter(corpus, out) -> Path:  # type: ignore[no-untyped-def]
    """A migratable adapter, through the command that produces one.

    `migrate` rewrites the **indexed document vectors**, so it needs a map out of
    the index's space and into the new model's — ``old_to_new``. `Bridge` needs
    the reverse. Each is useless in the other's place and both guards now say so.

    This goes through `rebasis fit --direction old_to_new` rather than
    constructing an adapter directly, because the point of these tests is the
    path a user takes. It replaces a helper that built one by hand, which existed
    only for the window in which nothing could produce a forward map at all.
    """
    result = runner.invoke(app, fit_args(corpus, out, "--direction", "old_to_new"))
    assert result.exit_code == EXIT_OK, result.output
    return out


def probe_args(corpus, *extra: str) -> list[str]:  # type: ignore[no-untyped-def]
    """The flags a user with unregistered models has to pass.

    `old-model` and `new-model` are deliberately absent from the profile table:
    that is the situation `--old-dim` / `--new-dim` exist for, and running the
    whole flow through them is the only way to know they work.
    """
    return [
        "probe",
        "--store",
        corpus["uri"],
        "--old",
        "old-model",
        "--new",
        "new-model",
        "--old-dim",
        str(DIM),
        "--new-dim",
        str(DIM),
        "--sample",
        "600",
        "--heldout",
        "150",
        "--state-dir",
        str(corpus["state"]),
        *extra,
    ]


def fit_args(corpus, out, *extra: str) -> list[str]:  # type: ignore[no-untyped-def]
    """The `fit` invocation, in one place.

    It appears in four tests, and four copies of an argument list is four
    places to forget when a flag changes — which is exactly what happened when
    `--new-dim` became necessary.
    """
    return [
        "fit",
        "--store",
        corpus["uri"],
        "--old",
        "old-model",
        "--new",
        "new-model",
        "--new-dim",
        str(DIM),
        "--pairs",
        "500",
        "--heldout",
        "150",
        "--out",
        str(out),
        *extra,
    ]


class TestProbe:
    def test_it_reaches_a_decision(self, corpus) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, probe_args(corpus))

        assert result.exit_code == EXIT_OK, result.output
        assert "ARR@10" in result.output

    def test_it_writes_the_report_format_the_suffix_asks_for(self, corpus) -> None:  # type: ignore[no-untyped-def]
        report = corpus["tmp"] / "report.html"

        result = runner.invoke(app, probe_args(corpus, "--report", str(report)))

        assert result.exit_code == EXIT_OK, result.output
        assert report.read_text(encoding="utf-8").lstrip().startswith("<")

    def test_it_leaves_the_index_untouched(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """`probe` is read-only, and the file's bytes are the proof."""
        before = (corpus["tmp"] / "corpus.npz").read_bytes()

        runner.invoke(app, probe_args(corpus))

        assert (corpus["tmp"] / "corpus.npz").read_bytes() == before

    def test_the_decision_lands_in_the_audit_trail(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The record is what makes a decision reproducible months later."""
        runner.invoke(app, probe_args(corpus))

        # A wide terminal: rich truncates the action column at the default 80,
        # and what is under test is the record, not the ellipsis.
        listed = runner.invoke(
            app,
            ["audit", "list", "--state-dir", str(corpus["state"])],
            env={"COLUMNS": "220"},
        )

        assert listed.exit_code == EXIT_OK, listed.output
        assert "probe.decision.made" in listed.output

    def test_the_decision_replays_to_the_same_answer(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The record must carry everything the decision depended on.

        The whole point of the audit trail is that a recommendation from six
        months ago can be re-derived. If the record is missing an input, the
        replay produces a different answer and this test says so.
        """
        runner.invoke(app, probe_args(corpus))

        replayed = runner.invoke(
            app,
            [
                "audit",
                "replay",
                "1",
                "--state-dir",
                str(corpus["state"]),
            ],
        )

        assert replayed.exit_code == EXIT_OK, replayed.output
        assert "matched" in replayed.output

    def test_replay_without_a_store_says_so(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """A record with no store cannot be replayed; that is a usage error,
        not a silent success."""
        from rebasis.audit import AuditWriter
        from rebasis.manifest import ManifestDB, manifest_path
        from rebasis.observability import Events

        db = ManifestDB(manifest_path(corpus["state"]))
        AuditWriter(db, run_id="test").write(
            Events.PROBE_DECISION_MADE,
            inputs={"seed": 0, "count": 10, "new_model": "new-model"},
            outputs={"decision": "bridge_sufficient", "arr_r10": 0.99},
        )
        db.close()

        replayed = runner.invoke(app, ["audit", "replay", "1", "--state-dir", str(corpus["state"])])

        assert replayed.exit_code == EXIT_USAGE, replayed.output

    def test_an_access_log_weights_the_queries_and_the_run_says_so(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """ARR under an access log estimates a different quantity — retention on
        the questions people send rather than on a uniform draw over the corpus
        — so a run that used one has to carry that fact into its output. Two
        numbers under one name is the failure this project keeps designing
        against.
        """
        log = corpus["tmp"] / "access.jsonl"
        # A small hot set, the shape an access log has. The ids come from the
        # fixture, so this weights records that are really in the index rather
        # than names the sampler will never see.
        log.write_text(
            "".join(
                json.dumps({"id": record_id, "count": 500}) + "\n"
                for record_id in corpus["ids"][:100]
            ),
            encoding="utf-8",
        )

        result = runner.invoke(app, probe_args(corpus, "--access-log", str(log), "--json"))

        assert result.exit_code == EXIT_OK, result.output
        assert json.loads(result.stdout)["access_weighted"] is True

    def test_without_one_the_run_says_that_too(self, corpus) -> None:  # type: ignore[no-untyped-def]
        result = runner.invoke(app, probe_args(corpus, "--json"))

        assert json.loads(result.stdout)["access_weighted"] is False

    def test_an_access_log_naming_nothing_in_the_index_is_not_a_weighted_run(  # type: ignore[no-untyped-def]
        self, corpus
    ) -> None:
        """Reporting the flag the user passed rather than the draw that happened
        would claim a measurement that was not taken."""
        log = corpus["tmp"] / "elsewhere.jsonl"
        log.write_text('{"id": "not-in-this-index", "count": 900}\n', encoding="utf-8")

        result = runner.invoke(app, probe_args(corpus, "--access-log", str(log), "--json"))

        assert result.exit_code == EXIT_OK, result.output
        assert json.loads(result.stdout)["access_weighted"] is False

    def test_a_malformed_query_log_exits_with_the_usage_code(self, corpus) -> None:  # type: ignore[no-untyped-def]
        broken = corpus["tmp"] / "queries.jsonl"
        broken.write_text("this is not json\n", encoding="utf-8")

        result = runner.invoke(app, probe_args(corpus, "--queries", str(broken)))

        assert result.exit_code == EXIT_USAGE


class TestFitAndMigrate:
    def test_fit_writes_a_loadable_adapter(self, corpus) -> None:  # type: ignore[no-untyped-def]
        out = corpus["tmp"] / "adapter.rbs"

        result = runner.invoke(
            app,
            fit_args(corpus, out),
        )

        assert result.exit_code == EXIT_OK, result.output
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["old_model_id"] == "old-model"
        assert manifest["new_model_id"] == "new-model"

    def test_eval_reads_back_what_fit_wrote(self, corpus) -> None:  # type: ignore[no-untyped-def]
        out = corpus["tmp"] / "adapter.rbs"
        runner.invoke(
            app,
            fit_args(corpus, out),
        )

        result = runner.invoke(app, ["eval", str(out), "--verify"])

        assert result.exit_code == EXIT_OK, result.output
        assert "verified" in result.output

    def test_migrate_refuses_the_adapter_fit_produces(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The one path a user is most likely to take, and it has to be refused.

        `README` and the migration guide both show `fit` writing an adapter and
        `migrate` taking it. That adapter maps a new-model *query* into the
        index; `migrate` rewrites the indexed *documents* and needs the reverse.
        Applying the query map to document vectors succeeds at every level that
        checks anything — the write lands, the count holds, the text survives,
        the read-back compares what was written against what came back, and the
        index-health check measures the store's search against exact kNN over
        the vectors it now holds. Measured on synthetic data where both spaces
        are known and the bridge itself scores 1.000, the index left behind
        answers recall@1 **0.000** to a raw new-model query, a bridged query and
        an old-model query alike.

        Nothing else in the suite catches it, which is why this is here.
        """
        out = corpus["tmp"] / "adapter.rbs"
        fitted = runner.invoke(app, fit_args(corpus, out))
        assert fitted.exit_code == EXIT_OK, fitted.output
        before = _vectors(corpus)

        refused = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--yes",
            ],
        )

        assert refused.exit_code == EXIT_USAGE, refused.output
        assert "query_to_old" in refused.output
        # Refused before anything was written, not part-way through.
        np.testing.assert_array_equal(_vectors(corpus), before)

    def test_migrate_then_rollback_restores_the_index(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The shadow copy returns the vectors bit for bit."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        before = _vectors(corpus)

        migrated = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--batch",
                "128",
                "--yes",
            ],
        )
        assert migrated.exit_code == EXIT_OK, migrated.output
        after = _vectors(corpus)
        assert not np.allclose(before, after)

        job_id = _latest_job(corpus)
        rolled = runner.invoke(
            app,
            ["rollback", job_id, "--state-dir", str(corpus["state"]), "--yes"],
        )

        assert rolled.exit_code == EXIT_OK, rolled.output
        np.testing.assert_array_equal(_vectors(corpus), before)

    def test_migrate_measures_what_the_index_can_still_find(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The check the read-back cannot do.

        Verifying a write proves the store took the vector. It does not prove
        the vector can still be retrieved, and on a graph index those are
        separate questions — the edges were chosen from the geometry the old
        vectors had. The store here searches exactly, so the honest answer is
        "no change"; what is being tested is that the question gets asked.
        """
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)

        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--yes",
            ],
        )

        assert result.exit_code == EXIT_OK, result.output
        assert "recall@10 against exact kNN" in result.output

    def test_the_health_check_can_be_turned_off(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """It costs two scans of the collection, which on a large index is a
        real amount of time to spend on a diagnostic."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)

        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--no-health-check",
                "--yes",
            ],
        )

        assert result.exit_code == EXIT_OK, result.output
        assert "exact kNN" not in result.output

    def test_rebuild_index_says_so_when_the_backend_cannot(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Asking for a rebuild on a backend that has no index is not an error.

        The in-memory store searches by matrix multiply, so there is no
        structure to rebuild and nothing was damaged. What must not happen is
        the flag appearing to work: the migration succeeds and the command says
        plainly that the rebuild was not available.
        """
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)

        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--rebuild-index",
                "--yes",
            ],
        )

        assert result.exit_code == EXIT_OK, result.output
        assert "index rebuild not available" in result.output

    def test_a_limited_run_says_the_index_is_now_mixed(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The whole CLI path, on a store that really got half rewritten.

        `--limit` is what the migration guide recommends for trying a
        migration, and it is the flag that most reliably produces an index
        holding both models' vectors. The unit tests cover the manifest query;
        this covers the thing a user would actually run into.
        """
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)

        migrated = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "20",
                "--yes",
            ],
        )

        assert migrated.exit_code == EXIT_OK, migrated.output
        # Once before the confirmation, once on the way out.
        assert "--limit stops this run short" in migrated.output
        assert "two embedding spaces" in migrated.output

        status = runner.invoke(app, ["status", "--state-dir", str(corpus["state"])])
        assert "two embedding spaces" in status.output

    def test_finishing_the_job_clears_the_warning(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Resuming to completion puts the index back into one space, and the
        warning has to stop — a notice that never clears is one people learn to
        ignore before the day it matters."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "20",
                "--yes",
            ],
        )
        runner.invoke(
            app,
            [
                "migrate",
                "--resume",
                _latest_job(corpus),
                "--state-dir",
                str(corpus["state"]),
                "--yes",
            ],
        )

        status = runner.invoke(app, ["status", "--state-dir", str(corpus["state"])])

        assert "two embedding spaces" not in status.output

    def test_resume_needs_only_the_job_id(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """A migration is resumed after an interruption, which is exactly when
        the adapter path and store URI are least likely to still be to hand.
        Both are on the job; `--resume` reads them back."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)

        started = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "2",
                "--yes",
            ],
        )
        assert started.exit_code == EXIT_OK, started.output

        job_id = _latest_job(corpus)
        resumed = runner.invoke(
            app,
            ["migrate", "--resume", job_id, "--state-dir", str(corpus["state"]), "--yes"],
        )

        assert resumed.exit_code == EXIT_OK, resumed.output
        assert "completed" in resumed.output

    def test_the_resume_command_finishes_what_migrate_started(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """`rebasis resume <job-id>` is the verb that pairs with `rebasis pause`.

        It forwards to `migrate --resume`, so what this proves is the
        forwarding: the job id reaches the right parameter, the adapter and
        store come off the job row exactly as they do for the flag, and the
        queue picks up where it stopped.
        """
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        started = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "2",
                "--yes",
            ],
        )
        assert started.exit_code == EXIT_OK, started.output

        resumed = runner.invoke(
            app,
            ["resume", _latest_job(corpus), "--state-dir", str(corpus["state"]), "--yes"],
        )

        assert resumed.exit_code == EXIT_OK, resumed.output
        assert "completed" in resumed.output

    def test_resume_forwards_the_flags_that_describe_the_run(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """`--limit` is the one that is visible in the outcome: a resume that
        stops short again leaves the index mixed, which `status` reports."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "2",
                "--yes",
            ],
        )

        resumed = runner.invoke(
            app,
            [
                "resume",
                _latest_job(corpus),
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "2",
                "--yes",
            ],
        )

        assert resumed.exit_code == EXIT_OK, resumed.output
        status = runner.invoke(app, ["status", "--state-dir", str(corpus["state"])])
        assert "two embedding spaces" in status.output

    def test_resume_says_so_when_the_job_never_recorded_an_adapter(  # type: ignore[no-untyped-def]
        self, corpus
    ) -> None:
        """Jobs written before the adapter path was recorded have an empty one.
        The remedy is to pass --adapter, and saying that beats a stack trace."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--limit",
                "2",
                "--yes",
            ],
        )
        job_id = _latest_job(corpus)

        db = sqlite3.connect(corpus["state"] / "manifest.db")
        db.execute("UPDATE jobs SET adapter_path = '' WHERE job_id = ?", (job_id,))
        db.commit()
        db.close()

        resumed = runner.invoke(
            app,
            ["migrate", "--resume", job_id, "--state-dir", str(corpus["state"]), "--yes"],
        )

        assert resumed.exit_code != EXIT_OK
        assert "--adapter" in resumed.output

    def test_status_reports_the_job(self, corpus) -> None:  # type: ignore[no-untyped-def]
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--yes",
            ],
        )

        result = runner.invoke(app, ["status", "--state-dir", str(corpus["state"])])

        assert result.exit_code == EXIT_OK, result.output
        assert "100%" in result.output


def _vectors(corpus) -> np.ndarray:  # type: ignore[no-untyped-def]
    from rebasis.store import open_store

    store = open_store(corpus["uri"])
    by_id = {r.id: r.vector for r in store.iter_records(with_text=False)}
    return np.vstack([by_id[i] for i in corpus["ids"]])


def _latest_job(corpus) -> str:  # type: ignore[no-untyped-def]
    from rebasis.manifest import ManifestDB, manifest_path

    db = ManifestDB(manifest_path(corpus["state"]))
    rows = db.query("SELECT job_id FROM jobs ORDER BY created_utc DESC")
    db.close()
    return str(rows[0]["job_id"])


class TestNonInteractiveUse:
    """The CLI in a script, a CI job or a pipe — where nobody can answer."""

    def test_a_missing_yes_is_a_usage_error_not_a_crash(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """`typer.confirm` raises `Abort`, which is not `typer.Exit`, so it
        reached the unexpected-error boundary and told the user their perfectly
        normal invocation was a bug in rebasis, with an issue link."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)

        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
            ],
            input="",
        )

        assert result.exit_code == EXIT_USAGE, result.output
        assert "--yes" in result.output
        assert "bug in rebasis" not in result.output

    def test_dry_run_writes_nothing(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The plan, and then nothing. `-n` is the flag people reach for."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        before = _vectors(corpus)

        result = runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--dry-run",
            ],
        )

        assert result.exit_code == EXIT_OK, result.output
        np.testing.assert_array_equal(_vectors(corpus), before)

    def test_apply_and_dry_run_together_are_refused(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Opposite instructions; guessing which one was meant is worse."""
        result = runner.invoke(
            app, ["gc", "--apply", "--dry-run", "--state-dir", str(corpus["state"])]
        )

        assert result.exit_code == EXIT_USAGE, result.output


class TestMachineReadableOutput:
    """`--json` has to be parseable — that is the whole contract."""

    def test_probe_json_is_only_json(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """Progress and 'Report written to' go to stderr precisely so this
        parses. A single stray line on stdout breaks every caller."""
        report = corpus["tmp"] / "r.md"
        result = runner.invoke(app, [*probe_args(corpus, "--json", "--report", str(report))])

        assert result.exit_code == EXIT_OK, result.output
        payload = json.loads(result.stdout)
        assert "decision" in payload
        assert report.exists()

    def test_status_json_carries_the_untruncated_job_id(self, corpus) -> None:  # type: ignore[no-untyped-def]
        """The table renders `job-daf2ac…`, which cannot be passed to rollback."""
        out = corpus["tmp"] / "adapter.rbs"
        forward_adapter(corpus, out)
        runner.invoke(
            app,
            [
                "migrate",
                "--adapter",
                str(out),
                "--store",
                corpus["uri"],
                "--state-dir",
                str(corpus["state"]),
                "--yes",
            ],
        )

        result = runner.invoke(app, ["status", "--json", "--state-dir", str(corpus["state"])])

        assert result.exit_code == EXIT_OK, result.output
        jobs = json.loads(result.stdout)
        assert jobs
        assert jobs[0]["job_id"] == _latest_job(corpus)
        assert "…" not in jobs[0]["job_id"]

    def test_doctor_json_parses(self) -> None:
        result = runner.invoke(app, ["doctor", "--json"])

        assert result.exit_code == EXIT_OK, result.output
        payload = json.loads(result.stdout)
        assert payload["rebasis"]
        assert isinstance(payload["store_backends"], list)
