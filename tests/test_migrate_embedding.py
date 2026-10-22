"""Tests for the embedding collection migration runbook (atomic staging swap)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from chromadb.errors import NotFoundError

from app.config import Settings
from scripts.migrate_embedding import (
    EXPECTED_DOCUMENT_COUNT,
    STAGING_SUFFIX,
    _snapshot_persist_dir,
    main,
    migrate_collection,
)


class FakeCollection:
    def __init__(
        self,
        count: int = EXPECTED_DOCUMENT_COUNT,
        name: str = "tkpi_gi",
        client: FakeClient | None = None,
    ) -> None:
        self._count = count
        self._name = name
        self._client = client

    @property
    def name(self) -> str:
        return self._name

    def count(self) -> int:
        return self._count

    def modify(self, *, name: str) -> None:
        old_name = self._name
        self._name = name
        if self._client is not None:
            del self._client.collections[old_name]
            self._client.collections[name] = self


class FakeClient:
    """Minimal in-memory Chroma client covering modify/delete/get/list."""

    def __init__(self, collection_exists: bool = True) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.deleted: list[str] = []
        if collection_exists:
            self.collections["tkpi_gi"] = FakeCollection(client=self)

    def list_collections(self) -> list[FakeCollection]:
        return list(self.collections.values())

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise NotFoundError(f"Collection {name} not found")
        self.deleted.append(name)
        del self.collections[name]

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections[name]


def _settings(persist_dir: Path) -> Settings:
    return Settings(
        chroma_persist_dir=str(persist_dir),
        collection_name="tkpi_gi",
        embedding_model_name="snowflake-arctic-embed2:latest",
        embedding_dim=1024,
        database_url="postgresql+asyncpg://u:p@h:5432/d",
        telegram_bot_token="test-token",
        llm_api_key="test-key",
    )


def test_main_refuses_to_run_without_confirmation(tmp_path: Path, monkeypatch: Any) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/d")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    result = main(["--persist-dir", str(persist_dir)])

    assert result != 0
    assert not (tmp_path / "reports").exists()


def test_migrate_collection_builds_staging_and_swaps_atomically(
    tmp_path: Path,
) -> None:
    """Staging is built and verified before the old collection is replaced."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    log_path = tmp_path / "indexing-raw.log"
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    client = FakeClient(collection_exists=True)
    events: list[str] = []

    def indexer(**kwargs: object) -> int:
        events.append("index")
        staging_name = str(kwargs["settings"].collection_name)
        assert staging_name == f"tkpi_gi{STAGING_SUFFIX}"
        assert Path(str(kwargs["dataset_path"])) == dataset_path
        client.collections[staging_name] = FakeCollection(name=staging_name, client=client)
        return EXPECTED_DOCUMENT_COUNT

    result = migrate_collection(
        settings=_settings(persist_dir),
        client=client,
        embedder=object(),
        dataset_path=dataset_path,
        log_path=log_path,
        timestamp="2026-10-22T12:34:56+00:00",
        indexer=indexer,
        after_staging_build=lambda _: events.append("staging"),
        after_drop=lambda _: events.append("drop"),
    )

    assert result == EXPECTED_DOCUMENT_COUNT
    assert events == ["index", "staging", "drop"]
    # Swap is a rename: the production name now points at the verified collection.
    assert set(client.collections) == {"tkpi_gi"}
    assert log_path.exists()
    log_entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert log_entry == {
        "timestamp": "2026-10-22T12:34:56+00:00",
        "collection": "tkpi_gi",
        "embedding_model": "snowflake-arctic-embed2:latest",
        "embedding_dim": 1024,
        "document_count": EXPECTED_DOCUMENT_COUNT,
    }


def test_migrate_collection_leaves_old_collection_on_failed_indexer(
    tmp_path: Path,
) -> None:
    """A failing indexer raises before the swap; the production collection survives."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    client = FakeClient(collection_exists=True)

    def failing_indexer(**_: object) -> int:
        name = f"tkpi_gi{STAGING_SUFFIX}"
        client.collections[name] = FakeCollection(count=3, name=name, client=client)
        return 3

    try:
        migrate_collection(
            settings=_settings(persist_dir),
            client=client,
            embedder=object(),
            dataset_path=tmp_path / "dataset.jsonl",
            log_path=tmp_path / "indexing-raw.log",
            timestamp="2026-10-22T12:34:56+00:00",
            indexer=failing_indexer,
        )
    except RuntimeError as error:
        assert "staging" in str(error)
    else:
        raise AssertionError("expected RuntimeError for underfilled staging collection")

    assert "tkpi_gi" in client.collections


def test_migrate_collection_skips_absent_old_collection(tmp_path: Path) -> None:
    """An absent production collection is fine: staging swap still completes."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    client = FakeClient(collection_exists=False)

    def indexer(**kwargs: object) -> int:
        name = str(kwargs["settings"].collection_name)
        client.collections[name] = FakeCollection(name=name, client=client)
        return EXPECTED_DOCUMENT_COUNT

    result = migrate_collection(
        settings=_settings(persist_dir),
        client=client,
        embedder=object(),
        dataset_path=tmp_path / "dataset.jsonl",
        log_path=tmp_path / "indexing-raw.log",
        timestamp="2026-10-22T12:34:56+00:00",
        indexer=indexer,
    )

    assert result == EXPECTED_DOCUMENT_COUNT
    assert client.deleted == []
    assert "tkpi_gi" in client.collections


def test_main_refuses_when_no_snapshot_and_no_acknowledgement(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """--no-snapshot without --i-know-app-is-stopped is rejected outright."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/d")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    result = main(
        [
            "--persist-dir",
            str(persist_dir),
            "--no-snapshot",
        ]
    )

    assert result != 0


def test_migrate_collection_restores_previous_when_staging_rename_fails(
    tmp_path: Path,
) -> None:
    """A failed swap puts the previous collection back; serving name survives."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    client = FakeClient(collection_exists=True)

    def indexer(**kwargs: object) -> int:
        name = str(kwargs["settings"].collection_name)
        staging = client.collections[name] = FakeCollection(name=name, client=client)
        # The staging rename fails after the previous collection was renamed
        # to the backup name (the exact rollback branch the fix targets).
        staging.modify = lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))
        return EXPECTED_DOCUMENT_COUNT

    try:
        migrate_collection(
            settings=_settings(persist_dir),
            client=client,
            embedder=object(),
            dataset_path=tmp_path / "dataset.jsonl",
            log_path=tmp_path / "indexing-raw.log",
            timestamp="2026-10-22T12:34:56+00:00",
            indexer=indexer,
        )
    except RuntimeError as error:
        assert "boom" in str(error)
    else:
        raise AssertionError("expected RuntimeError from failing staging rename")

    assert "tkpi_gi" in client.collections
    assert "tkpi_gi_pre_migration" not in client.collections


def test_migrate_collection_aborts_when_backup_exists(tmp_path: Path) -> None:
    """A stale backup from a failed run blocks the swap before any deletion."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    client = FakeClient(collection_exists=True)
    client.collections["tkpi_gi_pre_migration"] = FakeCollection(
        name="tkpi_gi_pre_migration", client=client
    )

    def indexer(**kwargs: object) -> int:
        name = str(kwargs["settings"].collection_name)
        client.collections[name] = FakeCollection(name=name, client=client)
        return EXPECTED_DOCUMENT_COUNT

    try:
        migrate_collection(
            settings=_settings(persist_dir),
            client=client,
            embedder=object(),
            dataset_path=tmp_path / "dataset.jsonl",
            log_path=tmp_path / "indexing-raw.log",
            timestamp="2026-10-22T12:34:56+00:00",
            indexer=indexer,
        )
    except RuntimeError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("expected RuntimeError for existing backup collection")

    # Nothing was destroyed: both the serving collection and the stale backup.
    assert "tkpi_gi" in client.collections
    assert "tkpi_gi_pre_migration" in client.collections
    assert client.deleted == []


def test_migrate_collection_dry_run_never_touches_serving_collection(
    tmp_path: Path,
) -> None:
    """With swap=False the staging is built and verified; nothing is renamed."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    client = FakeClient(collection_exists=True)

    def indexer(**kwargs: object) -> int:
        name = str(kwargs["settings"].collection_name)
        client.collections[name] = FakeCollection(name=name, client=client)
        return EXPECTED_DOCUMENT_COUNT

    result = migrate_collection(
        settings=_settings(persist_dir),
        client=client,
        embedder=object(),
        dataset_path=tmp_path / "dataset.jsonl",
        log_path=tmp_path / "indexing-raw.log",
        timestamp="2026-10-22T12:34:56+00:00",
        indexer=indexer,
        swap=False,
    )

    assert result == EXPECTED_DOCUMENT_COUNT
    assert f"tkpi_gi{STAGING_SUFFIX}" in client.collections
    assert "tkpi_gi" in client.collections
    assert client.deleted == []


def test_snapshot_rejects_empty_archive(tmp_path: Path) -> None:
    """An archive without the sqlite file is rejected instead of trusted."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    backup_dir = tmp_path / "backups"

    # An empty persist dir produces an archive without the sqlite file.
    try:
        _snapshot_persist_dir(persist_dir, backup_dir, "2026-10-22T00:00:00+00:00")
    except RuntimeError as error:
        assert "does not contain" in str(error)
    else:
        raise AssertionError("expected RuntimeError for archive without sqlite file")


def test_main_refuses_when_app_health_answers(tmp_path: Path, monkeypatch: Any) -> None:
    """The preflight refuses to run while the app answers on /health."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/d")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())

    result = main(["--persist-dir", str(persist_dir), "--confirm-drop-collection"])

    assert result == 3


def test_main_proceeds_when_app_is_stopped_flag_given(tmp_path: Path, monkeypatch: Any) -> None:
    """--i-know-app-is-stopped bypasses the health check and runs the flow."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/d")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "Nama Bahan Pangan": "Beras",
                "Kategori": "serealia",
                "BDD": 100.0,
                "gi_value": None,
                "gl_value": None,
                "serving_size_g": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())

    with pytest.raises(SystemExit):
        # The flow gets past the preflight and exits on the shortened dataset.
        main(
            [
                "--persist-dir",
                str(persist_dir),
                "--confirm-drop-collection",
                "--i-know-app-is-stopped",
                "--dataset",
                str(dataset_path),
                "--dry-run",
            ]
        )
