"""`doctor --store`: what it reports about a live index, and what it refuses to.

`doctor` is the command a person runs when they are already confused, so two
properties matter more here than anywhere else in the CLI.

**Nothing it is pointed at may change.** Not the index, not the state directory.
The manifest is the sharp edge: `ManifestDB` migrates its schema on connect and
takes a backup on the way, which is right for `status` and wrong for a
diagnostic. That is asserted rather than assumed.

**A failed check may not take the report with it.** A store that will not open is
the most likely reason someone typed this command, so every path through it —
a URI that does not parse, a file that is not there, a backend that cannot
return text — has to leave the rest of the report on screen and `--json`
parseable. There is a test per failure mode for exactly that reason.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from typer.testing import CliRunner

from rebasis.cli import app
from rebasis.core import ProcrustesAdapter, l2_normalize, save_adapter
from rebasis.errors import EXIT_OK
from rebasis.manifest import ManifestDB, manifest_path
from rebasis.migrate import MigrationEngine
from rebasis.store import MemoryStore, open_store
from rebasis.store.uri import StoreURI
from rebasis.types import EncodingProfile

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

DIM = 16
N = 64


@pytest.fixture
def vectors(rng: np.random.Generator) -> np.ndarray:
    return l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))


@pytest.fixture
def ids() -> list[str]:
    return [f"doc-{i:04d}" for i in range(N)]


@pytest.fixture
def store(tmp_path: Path, ids: list[str], vectors: np.ndarray) -> str:
    """An index on disk, with text, reachable through the reference backend."""
    path = tmp_path / "vectors.npz"
    MemoryStore(ids, vectors, [f"chunk {i}" for i in range(N)]).save(path)
    return f"memory://{path}"


@pytest.fixture
def textless(tmp_path: Path, ids: list[str], vectors: np.ndarray) -> str:
    """The same index with no text column, which is a declared limitation."""
    path = tmp_path / "vectors-only.npz"
    MemoryStore(ids, vectors).save(path)
    return f"memory://{path}"


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """A state directory that does not exist until a test makes one."""
    return tmp_path / "state"


def doctor(store_uri: str | None, state_dir: Path, *, as_json: bool = False) -> Any:
    """Run `doctor`, always naming the state directory.

    Never left to the default: it resolves to `.rebasis` beside the working
    directory, and the working directory during a test run is this repository —
    which has one.
    """
    args = ["doctor", "--state-dir", str(state_dir)]
    if store_uri is not None:
        args += ["--store", store_uri]
    if as_json:
        args.append("--json")
    return runner.invoke(app, args, env=WIDE)


def payload_for(store_uri: str | None, state_dir: Path) -> dict[str, Any]:
    """The `store` section of `doctor --json`, parsed from stdout alone.

    stdout, not `output`: `output` interleaves stderr, and every rebasis log
    line goes there — including `runtime.detected`, emitted on every run. A
    script doing `rebasis doctor --json | jq` reads stdout.
    """
    result = doctor(store_uri, state_dir, as_json=True)
    assert result.exit_code == EXIT_OK, result.output
    document: dict[str, Any] = json.loads(result.stdout)
    return document


def checks_of(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["name"]: check for check in document["store"]["checks"]}


class TestTheHappyPath:
    def test_it_reports_what_the_index_holds(self, store: str, state: Path) -> None:
        result = doctor(store, state)

        assert result.exit_code == EXIT_OK, result.output
        assert "memory" in result.output
        assert str(N) in result.output
        assert str(DIM) in result.output

    def test_the_environment_report_is_still_there(self, store: str, state: Path) -> None:
        """`--store` adds a section; it does not replace the one people rely on."""
        result = doctor(store, state)

        assert "store backends" in result.output
        assert "python" in result.output

    def test_json_carries_the_facts(self, store: str, state: Path) -> None:
        document = payload_for(store, state)

        assert document["store"]["opened"] is True
        assert document["store"]["records"] == N
        assert document["store"]["dimension"] == DIM
        assert document["store"]["backend"] == "memory"

    def test_the_keys_that_were_there_before_are_still_there(self, store: str, state: Path) -> None:
        """`doctor --json` is what the README tells people to attach to a bug
        report, so a script reading it must not have to know which flags the
        reporter passed."""
        document = payload_for(store, state)

        assert document["rebasis"]
        assert "blas" in document
        assert "store_backends" in document

    def test_without_a_store_the_section_is_null(self, state: Path) -> None:
        document = payload_for(None, state)

        assert document["store"] is None

    def test_capabilities_are_not_enumerated_by_hand(self, store: str, state: Path) -> None:
        """Every field of `StoreCapabilities`, including ones added after this
        was written.

        The rendering reads the dataclass rather than naming its fields, because
        a report that silently omits the newest capability is worse than one
        that omits all of them: it looks complete.
        """
        from dataclasses import fields

        from rebasis.types import StoreCapabilities

        document = payload_for(store, state)

        declared = document["store"]["capabilities"]
        assert set(declared) == {entry.name for entry in fields(StoreCapabilities)}

    def test_text_is_reported_as_readable(self, store: str, state: Path) -> None:
        assert checks_of(payload_for(store, state))["text"]["ok"] is True


class TestAUriThatDoesNotParse:
    def test_it_says_so_without_failing(self, state: Path) -> None:
        """Exit zero: `doctor` reporting a broken URI has done its job."""
        result = doctor("not-a-uri", state)

        assert result.exit_code == EXIT_OK, result.output
        assert "RB-E1001" in result.output

    def test_the_rest_of_the_report_still_prints(self, state: Path) -> None:
        result = doctor("not-a-uri", state)

        assert "store backends" in result.output

    def test_json_is_still_parseable(self, state: Path) -> None:
        document = payload_for("not-a-uri", state)

        assert document["store"]["opened"] is False
        assert checks_of(document)["uri"]["ok"] is False

    def test_nothing_downstream_of_it_claims_a_verdict(self, state: Path) -> None:
        """A check that could not run is not a check that passed."""
        checks = checks_of(payload_for("not-a-uri", state))

        assert checks["open"]["ok"] is None
        assert checks["text"]["ok"] is None
        assert checks["sqlite"]["ok"] is None


class TestWhatReachesABugReport:
    """`doctor --json` is what the README tells people to attach to an issue.

    Which makes the credential portion of a store URI the one thing that must
    never reach it. `StoreURI.redacted` is the route everywhere the URI is
    parseable; where it is not, the string is still not passed through
    verbatim.
    """

    def test_a_credential_in_a_working_uri_is_redacted(self, state: Path) -> None:
        document = payload_for("nosuchbackend://user:hunter2@host/db", state)

        assert "hunter2" not in json.dumps(document)
        assert "<credentials>" in document["store"]["uri"]

    def test_a_credential_in_an_unparseable_uri_is_redacted_too(self, state: Path) -> None:
        """A password does not stop being a password because the parser
        rejected the string it was in."""
        document = payload_for("://user:hunter2@host/db", state)

        assert "hunter2" not in json.dumps(document)


class TestAStoreThatWillNotOpen:
    def test_the_backend_error_is_what_gets_reported(self, tmp_path: Path, state: Path) -> None:
        result = doctor(f"memory://{tmp_path / 'absent.npz'}", state)

        assert result.exit_code == EXIT_OK, result.output
        assert "RB-E3003" in result.output

    def test_json_is_still_parseable(self, tmp_path: Path, state: Path) -> None:
        document = payload_for(f"memory://{tmp_path / 'absent.npz'}", state)

        assert document["store"]["opened"] is False
        assert document["store"]["records"] is None
        assert checks_of(document)["open"]["ok"] is False


class TestTextThatCannotBeRead:
    def test_a_declared_limitation_is_reported_as_one(self, textless: str, state: Path) -> None:
        checks = checks_of(payload_for(textless, state))

        assert checks["text"]["ok"] is False
        assert "cannot return document text" in checks["text"]["detail"]

    def test_everything_else_about_the_index_still_reports(
        self, textless: str, state: Path
    ) -> None:
        """The collection is fine; one capability is missing."""
        document = payload_for(textless, state)

        assert document["store"]["records"] == N
        assert document["store"]["dimension"] == DIM

    def test_the_human_report_says_what_it_costs(self, textless: str, state: Path) -> None:
        result = doctor(textless, state)

        assert result.exit_code == EXIT_OK, result.output
        assert "rebasis probe" in result.output


# ── the manifest half ─────────────────────────────────────────────────


def build_job(
    state_dir: Path,
    shadow_root: Path,
    *,
    store_uri: str,
    adapter_path: str = "",
    rng: np.random.Generator,
) -> MigrationEngine:
    """A migration job registered against ``store_uri``, ready to be run short.

    The engine opens the store from the URI rather than being handed one: a job
    that runs to completion re-opens it to re-check that its writes survived,
    and a store built separately would not be holding them.
    """
    src = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((N, DIM)).astype(np.float32))
    ids = [f"doc-{i:04d}" for i in range(N)]
    engine = MigrationEngine(
        db=ManifestDB(manifest_path(state_dir)),
        store=open_store(store_uri),
        adapter=ProcrustesAdapter.fit(src, dst),
        shadow_root=shadow_root,
        batch_size=8,
        power_aware=False,
        store_uri=store_uri,
        adapter_path=adapter_path,
    )
    engine.prepare(ids)
    return engine


def write_adapter(path: Path, *, input_dim: int, output_dim: int, rng: Any) -> Path:
    """A real `.rbs` on disk, because the check reads the real manifest."""
    src = l2_normalize(rng.standard_normal((N, input_dim)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((N, output_dim)).astype(np.float32))
    return save_adapter(
        ProcrustesAdapter.fit(src, dst),
        path,
        direction="query_to_old",
        old_profile=EncodingProfile(model_id="old-model", dim=output_dim),
        new_profile=EncodingProfile(model_id="new-model", dim=input_dim),
    )


class TestMixedSpaces:
    def test_it_fires_on_a_half_migrated_index(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The check `--store` exists for: nothing raises, no count changes, and
        a third of the queries come back wrong."""
        engine = build_job(state, tmp_path / "shadow", store_uri=store, rng=rng)
        engine.run(limit=16)
        engine.db.close()

        result = doctor(store, state)

        assert result.exit_code == EXIT_OK, result.output
        assert "two embedding spaces" in result.output
        assert f"migrate --resume {engine.job_id}" in result.output

    def test_json_carries_the_field_a_script_branches_on(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        engine = build_job(state, tmp_path / "shadow", store_uri=store, rng=rng)
        engine.run(limit=16)
        engine.db.close()

        document = payload_for(store, state)

        assert checks_of(document)["spaces"]["ok"] is False
        assert document["store"]["mixed_spaces"][0]["migrated"] == 16
        assert document["store"]["mixed_spaces"][0]["unmigrated"] == N - 16

    def test_it_does_not_fire_on_a_finished_job(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        engine = build_job(state, tmp_path / "shadow", store_uri=store, rng=rng)
        engine.run()
        engine.db.close()

        result = doctor(store, state)

        assert "two embedding spaces" not in result.output
        assert checks_of(payload_for(store, state))["spaces"]["ok"] is True

    def test_it_does_not_fire_for_a_different_index(
        self, store: str, textless: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """Matched on the URI as recorded, the same way `status` matches."""
        engine = build_job(state, tmp_path / "shadow", store_uri=store, rng=rng)
        engine.run(limit=16)
        engine.db.close()

        assert checks_of(payload_for(textless, state))["spaces"]["ok"] is True


class TestTheRecordedProfile:
    def test_nothing_recorded_says_nothing_recorded(self, store: str, state: Path) -> None:
        """Not silence, and not a pass. The three-valued verdict is the point."""
        checks = checks_of(payload_for(store, state))

        assert checks["profile"]["ok"] is None
        assert "no recorded profile for this collection" in checks["profile"]["detail"]

    def test_an_adapter_that_matches_the_index_passes(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        adapter = write_adapter(tmp_path / "a.rbs", input_dim=32, output_dim=DIM, rng=rng)
        build_job(
            state, tmp_path / "shadow", store_uri=store, adapter_path=str(adapter), rng=rng
        ).db.close()

        checks = checks_of(payload_for(store, state))

        assert checks["profile"]["ok"] is True
        assert "old-model" in checks["profile"]["detail"]

    def test_an_adapter_fitted_against_something_else_is_caught(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """The one comparison a recorded profile supports against a live index:
        a `query_to_old` adapter maps into the index's own space, so an output
        dimension the index does not have was fitted against another one."""
        adapter = write_adapter(tmp_path / "b.rbs", input_dim=32, output_dim=8, rng=rng)
        build_job(
            state, tmp_path / "shadow", store_uri=store, adapter_path=str(adapter), rng=rng
        ).db.close()

        checks = checks_of(payload_for(store, state))

        assert checks["profile"]["ok"] is False
        assert "8 dimensions" in checks["profile"]["detail"]
        assert checks["profile"]["hint"]

    def test_an_adapter_that_has_been_deleted_says_nothing(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """A job recording a path that no longer holds an adapter tells you
        nothing about the index, and must not be reported as if it did."""
        build_job(
            state,
            tmp_path / "shadow",
            store_uri=store,
            adapter_path=str(tmp_path / "gone.rbs"),
            rng=rng,
        ).db.close()

        assert checks_of(payload_for(store, state))["profile"]["ok"] is None


class TestTheManifestItself:
    def test_a_healthy_manifest_is_reported_intact(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        build_job(state, tmp_path / "shadow", store_uri=store, rng=rng).db.close()

        checks = checks_of(payload_for(store, state))

        assert checks["manifest"]["ok"] is True

    def test_no_state_directory_is_not_a_verdict(self, store: str, state: Path) -> None:
        checks = checks_of(payload_for(store, state))

        assert checks["manifest"]["ok"] is None
        assert checks["spaces"]["ok"] is None


class TestItWritesNothing:
    def test_a_missing_state_directory_is_not_created(self, store: str, state: Path) -> None:
        doctor(store, state)

        assert not state.exists()

    def test_the_index_is_left_byte_for_byte(self, tmp_path: Path, store: str, state: Path) -> None:
        path = tmp_path / "vectors.npz"
        before = path.read_bytes()

        doctor(store, state)

        assert path.read_bytes() == before

    def test_a_manifest_this_release_would_migrate_is_not_migrated(
        self, store: str, state: Path, tmp_path: Path, rng: np.random.Generator
    ) -> None:
        """`ManifestDB` upgrades its schema on connect and takes a `VACUUM INTO`
        backup on the way. That is right for `status` and wrong for a command
        whose whole promise is that it changes nothing, so `doctor` reads the
        schema out of the file header and declines to open an older one.
        """
        build_job(state, tmp_path / "shadow", store_uri=store, rng=rng).db.close()
        path = manifest_path(state)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA user_version = 1")
        connection.close()

        checks = checks_of(payload_for(store, state))

        assert checks["manifest"]["ok"] is None
        assert "does not write" in checks["manifest"]["detail"]
        connection = sqlite3.connect(path)
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 1
        connection.close()
        assert not list(path.parent.glob("*.bak.*"))


class TestTheSqliteCheck:
    """`PRAGMA integrity_check` against the file under an index.

    Exercised directly rather than through a backend: which stores it reaches is
    a question about file layouts, and standing up a real sqlite-vec table would
    test the extension loader instead.
    """

    def test_a_real_database_passes(self, tmp_path: Path) -> None:
        from rebasis.cli.doctor import _check_sqlite

        path = tmp_path / "index.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE documents (id TEXT, embedding BLOB)")
        connection.commit()
        connection.close()

        uri = StoreURI(backend="sqlite-vec", path=str(path), collection="documents")
        check = _check_sqlite(uri)

        assert check.ok is True

    def test_a_chroma_directory_is_reached_through_its_own_file(self, tmp_path: Path) -> None:
        from rebasis.cli.doctor import _check_sqlite

        directory = tmp_path / "chroma"
        directory.mkdir()
        connection = sqlite3.connect(directory / "chroma.sqlite3")
        connection.execute("CREATE TABLE collections (id TEXT)")
        connection.commit()
        connection.close()

        check = _check_sqlite(StoreURI(backend="chroma", path=str(directory), collection="docs"))

        assert check.ok is True

    def test_a_backend_that_keeps_no_sqlite_file_is_not_a_failure(self, tmp_path: Path) -> None:
        """A backend with nothing to check has not failed a check."""
        from rebasis.cli.doctor import _check_sqlite

        path = tmp_path / "vectors.lance"
        path.write_bytes(b"not sqlite at all")

        check = _check_sqlite(StoreURI(backend="lancedb", path=str(path), collection="documents"))

        assert check.ok is None
        assert "no SQLite database" in check.detail
