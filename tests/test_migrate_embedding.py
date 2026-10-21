"""Tests for the embedding collection migration runbook."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from chromadb.errors import NotFoundError

from app.config import Settings
from scripts.migrate_embedding import EXPECTED_DOCUMENT_COUNT, main, migrate_collection


class FakeCollection:
    def __init__(self, count: int = EXPECTED_DOCUMENT_COUNT) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class FakeClient:
    def __init__(self, collection_exists: bool = True) -> None:
        self.collections = {"tkpi_gi": FakeCollection()} if collection_exists else {}
        self.deleted: list[str] = []

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
        embedding_model_name="snowflake-arctic-embed2",
        embedding_dim=1024,
    )


def test_main_refuses_to_run_without_confirmation(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    result = main(["--persist-dir", str(persist_dir)])

    assert result != 0
    assert not (tmp_path / "reports").exists()


def test_migrate_collection_snapshots_before_drop_and_logs_verified_count(
    tmp_path: Path,
) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    (persist_dir / "chroma.sqlite3").write_bytes(b"database")
    backup_dir = tmp_path / "backups"
    log_path = tmp_path / "indexing-raw.log"
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    client = FakeClient()
    events: list[str] = []

    def indexer(**kwargs: object) -> int:
        events.append("index")
        assert Path(str(kwargs["dataset_path"])) == dataset_path
        client.collections["tkpi_gi"] = FakeCollection()
        return EXPECTED_DOCUMENT_COUNT

    result = migrate_collection(
        settings=_settings(persist_dir),
        client=client,
        embedder=object(),
        dataset_path=dataset_path,
        backup_dir=backup_dir,
        log_path=log_path,
        timestamp="2026-10-22T12:34:56+00:00",
        indexer=indexer,
        after_snapshot=lambda _: events.append("snapshot"),
        after_drop=lambda _: events.append("drop"),
    )

    assert result == EXPECTED_DOCUMENT_COUNT
    assert events == ["snapshot", "drop", "index"]
    assert client.deleted == ["tkpi_gi"]
    backups = list(backup_dir.glob("chroma_data-*.tar.gz"))
    assert len(backups) == 1
    with tarfile.open(backups[0], "r:gz") as archive:
        assert archive.getnames() == ["chroma_data", "chroma_data/chroma.sqlite3"]
    log_entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert log_entry == {
        "timestamp": "2026-10-22T12:34:56+00:00",
        "collection": "tkpi_gi",
        "embedding_model": "snowflake-arctic-embed2",
        "embedding_dim": 1024,
        "document_count": EXPECTED_DOCUMENT_COUNT,
    }


def test_migrate_collection_skips_dropping_absent_collection(tmp_path: Path) -> None:
    """An absent collection is not dropped; the snapshot still precedes reindex."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    (persist_dir / "chroma.sqlite3").write_bytes(b"database")
    client = FakeClient(collection_exists=False)

    def indexer(**_: object) -> int:
        client.collections["tkpi_gi"] = FakeCollection()
        return EXPECTED_DOCUMENT_COUNT

    result = migrate_collection(
        settings=_settings(persist_dir),
        client=client,
        embedder=object(),
        dataset_path=tmp_path / "dataset.jsonl",
        backup_dir=tmp_path / "backups",
        log_path=tmp_path / "indexing-raw.log",
        timestamp="2026-10-22T12:34:56+00:00",
        indexer=indexer,
    )

    assert result == EXPECTED_DOCUMENT_COUNT
    assert client.deleted == []
    assert (tmp_path / "backups" / "chroma_data-2026-10-22T12:34:56+00:00.tar.gz").exists()
