"""Migration runbook: snapshot -> drop -> reindex the ``tkpi_gi`` collection.

Executes the in-place cutover from the old OpenAI embeddings to the
self-hosted ``snowflake-arctic-embed2`` index, per the arctic migration spec.
It must be run explicitly with ``--confirm-drop-collection`` because dropping
the production collection is destructive.

Flow
----
1. Snapshot the configured ChromaDB persist dir to
   ``reports/backups/chroma_data-<timestamp>.tar.gz`` (skip with
   ``--no-snapshot``) and verify the archive lists non-empty.
2. Drop the ``tkpi_gi`` collection (skipped when it is absent).
3. Re-index 1146 documents via ``scripts.index_data.index_dataset`` and verify
   ``collection.count() == 1146``.
4. Append a raw migration log entry to ``reports/logs/indexing-raw.log``.

Rollback
--------
Restore the snapshot archive over the persist dir, then restart the app:

.. code-block:: bash

    # Docker deployment: the collection lives in the app's volume (or the
    # ``chroma_data`` volume when mounted separately).
    docker compose stop app
    docker run --rm -v <project>_chroma_data:/data -v "$PWD:/backup" \
        alpine sh -c 'rm -rf /data/* && tar xzf /backup/chroma_data-<ts>.tar.gz -C /data'
    # Local development: replace the persist dir contents.
    tar xzf reports/backups/chroma_data-<ts>.tar.gz -C chroma_db/ \
        --strip-components=1

Usage
-----
    uv run python -m scripts.migrate_embedding --confirm-drop-collection
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from app.config import Settings
from app.data.parser import load_records
from app.services.ollama_embedder import OllamaEmbedder
from scripts.index_data import index_dataset

EXPECTED_DOCUMENT_COUNT = 1146


def _snapshot_persist_dir(persist_dir: Path, backup_dir: Path, timestamp: str) -> Path:
    """Tar the persist dir and verify the archive lists before the drop."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_dir / f"chroma_data-{timestamp}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(persist_dir, arcname="chroma_data")
    with tarfile.open(archive_path, "r:gz") as archive:
        if not archive.getnames():
            raise RuntimeError(f"snapshot archive is empty: {archive_path}")
    return archive_path


def migrate_collection(
    *,
    settings: Settings,
    client: Any,
    embedder: Any,
    dataset_path: Path,
    backup_dir: Path,
    log_path: Path,
    timestamp: str,
    indexer: Callable[..., int] = index_dataset,
    after_snapshot: Callable[[Path], None] | None = None,
    after_drop: Callable[[list[str]], None] | None = None,
    snapshot: bool = True,
) -> int:
    """Drop the configured collection and rebuild it, returning its count."""
    persist_dir = Path(settings.chroma_persist_dir)
    if snapshot:
        archive_path = _snapshot_persist_dir(persist_dir, backup_dir, timestamp)
        if after_snapshot is not None:
            after_snapshot(archive_path)

    dropped: list[str] = []
    try:
        client.delete_collection(settings.collection_name)
        dropped.append(settings.collection_name)
    except NotFoundError:
        pass
    if after_drop is not None:
        after_drop(dropped)

    indexed_count = indexer(
        dataset_path=dataset_path,
        settings=settings,
        client=client,
        embedder=embedder,
    )
    count = client.get_collection(settings.collection_name).count()
    if indexed_count != EXPECTED_DOCUMENT_COUNT or count != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_DOCUMENT_COUNT} documents after migration, "
            f"indexed {indexed_count}, collection holds {count}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as raw_log:
        payload = {
            "timestamp": timestamp,
            "collection": settings.collection_name,
            "embedding_model": settings.embedding_model_name,
            "embedding_dim": settings.embedding_dim,
            "document_count": count,
        }
        raw_log.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-drop-collection",
        action="store_true",
        help="Required safety flag: proceed with dropping the production collection",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip the pre-drop snapshot of the persist dir",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Override the configured ChromaDB persist dir",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/tkpi-gi-final.jsonl"),
        help="Path to the TKPI+GI JSONL dataset",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("reports/backups"),
        help="Where to write the pre-drop snapshot archive",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("reports/logs/indexing-raw.log"),
        help="Where to append the raw migration log",
    )
    args = parser.parse_args(argv)

    if not args.confirm_drop_collection:
        print(
            "Refusing to migrate: pass --confirm-drop-collection to drop and reindex "
            f"{Settings().collection_name!r}",
            file=sys.stderr,
        )
        return 2

    settings = Settings()
    if args.persist_dir is not None:
        settings = settings.model_copy(update={"chroma_persist_dir": str(args.persist_dir)})
    if not Path(settings.chroma_persist_dir).exists():
        raise SystemExit(f"persist dir not found: {settings.chroma_persist_dir}")

    records = load_records(args.dataset)
    if len(records) != EXPECTED_DOCUMENT_COUNT:
        raise SystemExit(f"expected {EXPECTED_DOCUMENT_COUNT} dataset records, got {len(records)}")

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedder = OllamaEmbedder(base_url=settings.ollama_base_url)

    if args.no_snapshot:
        print("WARNING: --no-snapshot given, skipping the pre-drop backup", file=sys.stderr)

    timestamp = datetime.now(UTC).isoformat()
    count = migrate_collection(
        settings=settings,
        client=client,
        embedder=embedder,
        dataset_path=args.dataset,
        backup_dir=args.backup_dir,
        log_path=args.log,
        timestamp=timestamp,
        snapshot=not args.no_snapshot,
    )
    print(
        f"Migrated {count} documents into {settings.collection_name!r} "
        f"@ {settings.embedding_dim} dim"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
