"""Migration runbook: snapshot -> build staging -> atomic swap the collection.

Executes the cutover to a new embedding model (currently
``snowflake-arctic-embed2``) without ever leaving the production collection
missing or half-populated: the new index is built in a *staging* collection
and swapped in only after its document count is verified, so a failed reindex
never destroys the serving index.

The script refuses to run when the application is reachable (ChromaDB does
not support concurrent multi-process access to one persist directory) and
must be run explicitly with ``--confirm-drop-collection`` because the swap
still deletes the previous collection.

Flow
----
1. Snapshot the configured ChromaDB persist dir to
   ``reports/backups/chroma_data-<timestamp>.tar.gz`` (skip only with
   ``--no-snapshot=yes-i-accept-data-loss``) and verify the archive contains
   the expected SQLite database file. The snapshot runs *before* any
   ``chromadb.PersistentClient`` is created, so the archived database is
   closed and consistent.
2. Preflight: refuse when the app answers on ``app_host:app_port``/health
   (a running app must be stopped first; ``--i-know-app-is-stopped``
   overrides for deployments without a reachable health endpoint).
3. Build the staging collection ``<collection>_staging`` with
   ``scripts.index_data.index_dataset`` and verify it holds exactly
   ``EXPECTED_DOCUMENT_COUNT`` documents.
4. Atomic swap: delete the production collection (skipped when absent) and
   copy the verified staging documents into the production name via
   ``collection.modify``-free rename (drop staging after success).
5. Append a raw migration log entry to ``reports/logs/indexing-raw.log``.

Rollback
--------
Restore the snapshot archive over the persist dir, then restart the app:

.. code-block:: bash

    # Docker deployment: the collection lives in the app's volume (or the
    # ``chroma_data`` volume when mounted separately).
    docker compose stop app
    docker run --rm -v <project>_chroma_data:/data -v "$PWD:/backup" \\
        alpine sh -c 'rm -rf /data/* && tar xzf /backup/chroma_data-<ts>.tar.gz -C /data'
    # Local development: replace the persist dir contents.
    tar xzf reports/backups/chroma_data-<ts>.tar.gz -C chroma_db/ \\
        --strip-components=1

Usage
-----
    uv run python -m scripts.migrate_embedding --confirm-drop-collection
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from app.config import Settings
from app.data.parser import load_records
from app.services.ollama_embedder import OllamaEmbedder
from scripts.index_data import EXPECTED_DOCUMENT_COUNT, index_dataset

STAGING_SUFFIX = "_staging"


def _snapshot_persist_dir(persist_dir: Path, backup_dir: Path, timestamp: str) -> Path:
    """Tar the persist dir and verify the archive is restorable before use."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_dir / f"chroma_data-{timestamp}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(persist_dir, arcname="chroma_data")
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        if not names:
            raise RuntimeError(f"snapshot archive is empty: {archive_path}")
        if not any(name.endswith("chroma.sqlite3") for name in names):
            raise RuntimeError(
                f"snapshot archive {archive_path} does not contain chroma.sqlite3"
            )
    return archive_path


def _app_is_reachable(settings: Settings) -> bool:
    """Return whether the configured application health endpoint answers."""
    host = "127.0.0.1" if settings.app_host == "0.0.0.0" else settings.app_host
    request = urllib.request.Request(
        f"http://{host}:{settings.app_port}/health",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _drop_staging(client: Any, settings: Settings) -> None:
    """Remove any leftover staging collection from an earlier failed run."""
    staging_name = f"{settings.collection_name}{STAGING_SUFFIX}"
    with contextlib.suppress(NotFoundError):
        client.delete_collection(staging_name)


def migrate_collection(
    *,
    settings: Settings,
    client: Any,
    embedder: Any,
    dataset_path: Path,
    log_path: Path,
    timestamp: str,
    indexer: Callable[..., int] = index_dataset,
    after_drop: Callable[[list[str]], None] | None = None,
    after_staging_build: Callable[[int], None] | None = None,
    swap: bool = True,
) -> int:
    """Build the new index in staging, verify it, then swap it in.

    The production collection is only deleted *after* the staging collection
    has been verified to hold ``EXPECTED_DOCUMENT_COUNT`` documents, so a
    failed reindex leaves the serving index untouched.
    """
    staging_name = f"{settings.collection_name}{STAGING_SUFFIX}"
    _drop_staging(client, settings)

    staging_settings = settings.model_copy(update={"collection_name": staging_name})
    indexed_count = indexer(
        dataset_path=dataset_path,
        settings=staging_settings,
        client=client,
        embedder=embedder,
    )
    staging = client.get_collection(staging_name)
    staging_count = staging.count()
    if indexed_count != EXPECTED_DOCUMENT_COUNT or staging_count != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_DOCUMENT_COUNT} documents in staging collection "
            f"{staging_name!r}, indexed {indexed_count}, collection holds {staging_count}"
        )
    if after_staging_build is not None:
        after_staging_build(staging_count)

    if swap:
        dropped: list[str] = []
        backup_name = f"{settings.collection_name}_pre_migration"
        with contextlib.suppress(NotFoundError):
            client.delete_collection(backup_name)

        previous = None
        with contextlib.suppress(NotFoundError, KeyError):
            previous = client.get_collection(settings.collection_name)

        try:
            if previous is not None:
                previous.modify(name=backup_name)
                dropped.append(settings.collection_name)
            staging.modify(name=settings.collection_name)
        except Exception:
            # Rename the previous collection back if the staging rename fails;
            # the serving name is therefore never left absent after an error.
            if previous is not None:
                try:
                    renamed = client.get_collection(backup_name)
                    renamed.modify(name=settings.collection_name)
                except Exception:
                    pass
            raise

        if after_drop is not None:
            after_drop(dropped)
        if previous is not None:
            with contextlib.suppress(NotFoundError):
                client.delete_collection(backup_name)

    count = client.get_collection(settings.collection_name).count()
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
        help="Required safety flag: proceed with the swap that drops the current collection",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip the pre-swap snapshot of the persist dir",
    )
    parser.add_argument(
        "--i-know-app-is-stopped",
        action="store_true",
        help="Skip the health-endpoint preflight check that the app is stopped",
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
        help="Where to write the pre-swap snapshot archive",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("reports/logs/indexing-raw.log"),
        help="Where to append the raw migration log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight, snapshot, and staging build only; never swap",
    )
    args = parser.parse_args(argv)

    if not args.confirm_drop_collection:
        print(
            "Refusing to migrate: pass --confirm-drop-collection to swap the "
            f"{Settings().collection_name!r} collection",
            file=sys.stderr,
        )
        return 2

    settings = Settings()
    if args.persist_dir is not None:
        settings = settings.model_copy(update={"chroma_persist_dir": str(args.persist_dir)})
    if not Path(settings.chroma_persist_dir).exists():
        raise SystemExit(f"persist dir not found: {settings.chroma_persist_dir}")

    if _app_is_reachable(settings) and not args.i_know_app_is_stopped:
        print(
            "Refusing to migrate: the app answers on "
            f"{settings.app_host}:{settings.app_port}/health. Stop the app first "
            "(ChromaDB does not support concurrent multi-process access), or pass "
            "--i-know-app-is-stopped when the app runs elsewhere.",
            file=sys.stderr,
        )
        return 3

    records = load_records(args.dataset)
    if len(records) != EXPECTED_DOCUMENT_COUNT:
        raise SystemExit(f"expected {EXPECTED_DOCUMENT_COUNT} dataset records, got {len(records)}")

    timestamp = datetime.now(UTC).isoformat()
    persist_dir = Path(settings.chroma_persist_dir)
    if args.no_snapshot:
        if not args.i_know_app_is_stopped:
            print(
                "Refusing: --no-snapshot requires --i-know-app-is-stopped so a "
                "data-losing run is never accidental.",
                file=sys.stderr,
            )
            return 2
        print("WARNING: --no-snapshot given, skipping the pre-swap backup", file=sys.stderr)
    else:
        # Snapshot BEFORE creating the PersistentClient: the client can write
        # to the SQLite database on creation, and the snapshot must capture a
        # closed, consistent database.
        archive_path = _snapshot_persist_dir(
            persist_dir, args.backup_dir, timestamp
        )
        print(f"Snapshot written: {archive_path}")

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model_name,
        read_timeout_seconds=settings.ollama_batch_read_timeout_seconds,
    )

    count = migrate_collection(
        settings=settings,
        client=client,
        embedder=embedder,
        dataset_path=args.dataset,
        log_path=args.log,
        timestamp=timestamp,
        swap=not args.dry_run,
    )
    if args.dry_run:
        print(
            f"Dry run: staging collection "
            f"{settings.collection_name + STAGING_SUFFIX!r} verified; nothing swapped."
        )
    else:
        print(
            f"Migrated {count} documents into {settings.collection_name!r} "
            f"@ {settings.embedding_dim} dim"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
