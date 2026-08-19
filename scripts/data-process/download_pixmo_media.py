#!/usr/bin/env python3
"""Materialize PixMo image URLs and create local-only ms-swift JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


DATASETS = ("pixmo-cap", "pixmo-points")
REWRITE_SPLITS = ("train", "val", "global_train")
SUCCESS_STATUSES = ("downloaded", "verified_existing")
IMAGE_SUFFIXES = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
USER_AGENT = "ms-swift-pixmo-materializer/1.0"
_THREAD_STATE = threading.local()


@dataclass(frozen=True)
class DownloadTask:
    canonical_url: str
    source_url: str
    identity: str
    expected_hashes: tuple[str, ...]


@dataclass(frozen=True)
class DownloadResult:
    canonical_url: str
    status: str
    attempts: int
    local_path: str | None = None
    http_status: int | None = None
    size_bytes: int | None = None
    actual_sha256: str | None = None
    content_type: str | None = None
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="Root containing source PixMo parquet directories.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root containing converted ms-swift datasets.")
    parser.add_argument(
        "--state-root",
        type=Path,
        help="Local-filesystem directory for SQLite state; required when output-root is an object-store mount.",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--phase", choices=("all", "inventory", "download", "rewrite", "status"), default="all")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--max-media", type=int, help="Limit downloads for a smoke test; inventory is still complete.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    parser.add_argument("--allow-incomplete-rewrite", action="store_true")
    parser.add_argument(
        "--rewrite-splits",
        nargs="+",
        choices=REWRITE_SPLITS,
        default=["train", "val"],
        help=(
            "Converted JSONL stems to localize (default: train val). Include global_train "
            "to retain the existing cross-dataset deduplication for training."
        ),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help=(
            "Write a partial-media training manifest after processing. Its train entries use "
            "local_global_train.jsonl and its eval entries use local_val.jsonl."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {args.source_root}")
    if not args.output_root.is_dir():
        raise SystemExit(f"Output root does not exist: {args.output_root}")
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than zero")
    if args.connect_timeout <= 0 or args.read_timeout <= 0:
        raise SystemExit("HTTP timeouts must be greater than zero")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.max_bytes <= 0:
        raise SystemExit("--max-bytes must be greater than zero")
    if args.max_media is not None and args.max_media <= 0:
        raise SystemExit("--max-media must be greater than zero")
    if args.report_every <= 0:
        raise SystemExit("--report-every must be greater than zero")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be greater than zero")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid image URL: {value!r}")
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    default_port = ((parsed.scheme.lower() == "http" and port == 80)
                    or (parsed.scheme.lower() == "https" and port == 443))
    netloc = hostname if not port or default_port else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS media (
            canonical_url TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            identity TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            local_path TEXT,
            http_status INTEGER,
            size_bytes INTEGER,
            actual_sha256 TEXT,
            content_type TEXT,
            image_format TEXT,
            width INTEGER,
            height INTEGER,
            error TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS expected_hashes (
            canonical_url TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            PRIMARY KEY (canonical_url, sha256),
            FOREIGN KEY (canonical_url) REFERENCES media(canonical_url)
        );
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS media_status_idx ON media(status);
        """
    )
    return connection


def source_parquets(source_root: Path, dataset: str) -> list[Path]:
    paths = sorted((source_root / dataset / "data").glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No source parquet files found for {dataset} below {source_root}")
    return paths


def build_inventory(connection: sqlite3.Connection, source_root: Path, dataset: str) -> dict[str, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to inventory PixMo parquet files") from exc

    row_count = 0
    invalid_urls = 0
    invalid_hashes = 0
    for source in source_parquets(source_root, dataset):
        parquet = pq.ParquetFile(source)
        columns = ["image_url"]
        if dataset == "pixmo-points":
            columns.append("image_sha256")
        for batch in parquet.iter_batches(columns=columns, batch_size=65536):
            rows = batch.to_pylist()
            media_rows: dict[str, tuple[str, str, str]] = {}
            expected_rows: set[tuple[str, str]] = set()
            for row in rows:
                row_count += 1
                source_url = str(row.get("image_url") or "").strip()
                try:
                    canonical = canonical_url(source_url)
                except (TypeError, ValueError):
                    invalid_urls += 1
                    continue
                identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                media_rows.setdefault(canonical, (canonical, source_url, identity))
                if dataset == "pixmo-points":
                    expected = str(row.get("image_sha256") or "").strip().lower()
                    if len(expected) == 64 and all(character in "0123456789abcdef" for character in expected):
                        expected_rows.add((canonical, expected))
                    else:
                        invalid_hashes += 1
            now = utc_now()
            connection.executemany(
                """
                INSERT OR IGNORE INTO media(canonical_url, source_url, identity, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ((*row, now) for row in media_rows.values()),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO expected_hashes(canonical_url, sha256) VALUES (?, ?)", expected_rows)
            connection.commit()
        print(f"[{dataset}] inventoried {source.name}: {row_count:,} source rows", flush=True)

    inventory = {
        "source_rows": row_count,
        "unique_media": connection.execute("SELECT COUNT(*) FROM media").fetchone()[0],
        "invalid_urls": invalid_urls,
        "invalid_hashes": invalid_hashes,
    }
    values = {
        "dataset": dataset,
        "inventory_completed_at": utc_now(),
        "inventory": json.dumps(inventory, sort_keys=True),
    }
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", values.items())
    connection.commit()
    return inventory


def iter_download_tasks(connection: sqlite3.Connection, retry_failed: bool,
                        max_media: int | None, batch_size: int) -> Iterator[DownloadTask]:
    last_rowid = 0
    yielded = 0
    if retry_failed:
        status_clause = "status NOT IN ('downloaded', 'verified_existing')"
    else:
        status_clause = "status = 'pending'"
    while max_media is None or yielded < max_media:
        limit = batch_size if max_media is None else min(batch_size, max_media - yielded)
        rows = connection.execute(
            f"""
            SELECT m.rowid, m.canonical_url, m.source_url, m.identity,
                   COALESCE(GROUP_CONCAT(h.sha256), '')
            FROM media AS m
            LEFT JOIN expected_hashes AS h ON h.canonical_url = m.canonical_url
            WHERE m.rowid > ? AND {status_clause}
            GROUP BY m.rowid, m.canonical_url, m.source_url, m.identity
            ORDER BY m.rowid
            LIMIT ?
            """, (last_rowid, limit)).fetchall()
        if not rows:
            return
        for rowid, canonical, source_url, identity, expected in rows:
            last_rowid = rowid
            yielded += 1
            hashes = tuple(value for value in expected.split(",") if value)
            yield DownloadTask(canonical, source_url, identity, hashes)


def get_http_session():
    session = getattr(_THREAD_STATE, "session", None)
    if session is not None:
        return session
    import requests

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"})
    _THREAD_STATE.session = session
    return session


def validate_image(path: Path) -> tuple[str, int, int, str]:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    suffix = IMAGE_SUFFIXES.get(image_format)
    if suffix is None:
        raise ValueError(f"unsupported image format: {image_format or 'unknown'}")
    return image_format, width, height, suffix


def retry_delay(attempt: int) -> float:
    return min(2**(attempt - 1), 8)


def download_one(task: DownloadTask, image_root: Path, partial_root: Path, connect_timeout: float,
                 read_timeout: float, retries: int, max_bytes: int) -> DownloadResult:
    session = get_http_session()
    partial_dir = partial_root / task.identity[:2]
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"{task.identity}.part"
    last_status: str | None = None
    last_http_status: int | None = None
    last_error: str | None = None
    attempts = 0
    for attempt in range(1, retries + 2):
        attempts = attempt
        partial_path.unlink(missing_ok=True)
        try:
            with session.get(
                    task.source_url,
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                    allow_redirects=True) as response:
                last_http_status = response.status_code
                if response.status_code >= 400:
                    last_status = "http_error"
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code not in RETRYABLE_HTTP_STATUSES or attempt > retries:
                        break
                    time.sleep(retry_delay(attempt))
                    continue
                raw_length = response.headers.get("Content-Length")
                if raw_length and int(raw_length) > max_bytes:
                    return DownloadResult(
                        task.canonical_url,
                        "too_large",
                        attempt,
                        http_status=response.status_code,
                        error=f"Content-Length {raw_length} exceeds {max_bytes}",
                    )
                digest = hashlib.sha256()
                size = 0
                with partial_path.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"download exceeded {max_bytes} bytes")
                        digest.update(chunk)
                        stream.write(chunk)
                if size == 0:
                    raise ValueError("empty response body")
                actual_sha256 = digest.hexdigest()
                if task.expected_hashes and actual_sha256 not in task.expected_hashes:
                    last_status = "hash_mismatch"
                    last_error = f"SHA-256 {actual_sha256} not in declared hashes {','.join(task.expected_hashes)}"
                    if attempt <= retries:
                        time.sleep(retry_delay(attempt))
                        continue
                    break
                image_format, width, height, suffix = validate_image(partial_path)
                target_dir = image_root / task.identity[:2]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{task.identity}{suffix}"
                os.replace(partial_path, target)
                return DownloadResult(
                    task.canonical_url,
                    "downloaded",
                    attempt,
                    local_path=str(target.resolve()),
                    http_status=response.status_code,
                    size_bytes=size,
                    actual_sha256=actual_sha256,
                    content_type=response.headers.get("Content-Type"),
                    image_format=image_format,
                    width=width,
                    height=height,
                )
        except ValueError as exc:
            last_status = "too_large" if "exceeded" in str(exc) else "invalid_image"
            last_error = str(exc)
            if attempt <= retries and last_status == "invalid_image":
                time.sleep(retry_delay(attempt))
                continue
            break
        except Exception as exc:
            last_status = "network_error"
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                time.sleep(retry_delay(attempt))
                continue
            break
        finally:
            if last_status is not None:
                partial_path.unlink(missing_ok=True)
    return DownloadResult(
        task.canonical_url,
        last_status or "network_error",
        attempts,
        http_status=last_http_status,
        error=(last_error or "unknown download error")[:2000],
    )


def record_result(connection: sqlite3.Connection, result: DownloadResult) -> None:
    connection.execute(
        """
        UPDATE media
        SET status = ?, attempts = attempts + ?, local_path = ?, http_status = ?, size_bytes = ?,
            actual_sha256 = ?, content_type = ?, image_format = ?, width = ?, height = ?,
            error = ?, updated_at = ?
        WHERE canonical_url = ?
        """,
        (
            result.status,
            result.attempts,
            result.local_path,
            result.http_status,
            result.size_bytes,
            result.actual_sha256,
            result.content_type,
            result.image_format,
            result.width,
            result.height,
            result.error,
            utc_now(),
            result.canonical_url,
        ),
    )


def database_status(connection: sqlite3.Connection) -> dict[str, Any]:
    statuses = dict(connection.execute("SELECT status, COUNT(*) FROM media GROUP BY status"))
    total_bytes = connection.execute(
        "SELECT COALESCE(SUM(size_bytes), 0) FROM media WHERE status IN ('downloaded', 'verified_existing')"
    ).fetchone()[0]
    total = sum(statuses.values())
    successful = sum(statuses.get(status, 0) for status in SUCCESS_STATUSES)
    pending = statuses.get("pending", 0)
    return {
        "total_media": total,
        "successful_media": successful,
        "failed_media": total - successful - pending,
        "pending_media": pending,
        "downloaded_bytes": total_bytes,
        "statuses": dict(sorted(statuses.items())),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def restore_database(state_path: Path, checkpoint_path: Path) -> None:
    if state_path.exists() or not checkpoint_path.is_file():
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.restore")
    try:
        shutil.copyfile(checkpoint_path, temporary)
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_database(connection: sqlite3.Connection, state_path: Path, checkpoint_path: Path) -> None:
    connection.commit()
    local_snapshot = state_path.with_name(f".{state_path.name}.{os.getpid()}.snapshot")
    remote_temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    local_snapshot.unlink(missing_ok=True)
    remote_temporary.unlink(missing_ok=True)
    try:
        snapshot_connection = sqlite3.connect(local_snapshot)
        try:
            connection.backup(snapshot_connection)
        finally:
            snapshot_connection.close()
        shutil.copyfile(local_snapshot, remote_temporary)
        os.replace(remote_temporary, checkpoint_path)
    finally:
        local_snapshot.unlink(missing_ok=True)
        remote_temporary.unlink(missing_ok=True)


def write_status_report(connection: sqlite3.Connection, dataset_dir: Path, dataset: str,
                        started_at: str | None = None) -> dict[str, Any]:
    report = {"dataset": dataset, "updated_at": utc_now(), **database_status(connection)}
    if started_at is not None:
        report["download_started_at"] = started_at
    write_json_atomic(dataset_dir / "media_download_report.json", report)
    return report


def download_inventory(connection: sqlite3.Connection, state_path: Path, checkpoint_path: Path,
                       dataset_dir: Path, dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    image_root = dataset_dir / "images"
    partial_root = dataset_dir / ".image-download-parts"
    image_root.mkdir(parents=True, exist_ok=True)
    partial_root.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    tasks = iter_download_tasks(connection, args.retry_failed, args.max_media, args.workers * 8)
    futures: dict[Future[DownloadResult], DownloadTask] = {}
    completed = 0
    runtime_counts: Counter[str] = Counter()
    exhausted = False
    last_report_time = time.monotonic()
    last_checkpoint_time = time.monotonic()
    last_checkpoint_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while futures or not exhausted:
            while not exhausted and len(futures) < args.workers * 4:
                try:
                    task = next(tasks)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(
                    download_one,
                    task,
                    image_root,
                    partial_root,
                    args.connect_timeout,
                    args.read_timeout,
                    args.retries,
                    args.max_bytes,
                )
                futures[future] = task
            if not futures:
                continue
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = DownloadResult(
                        task.canonical_url,
                        "worker_error",
                        1,
                        error=f"{type(exc).__name__}: {exc}"[:2000],
                    )
                record_result(connection, result)
                runtime_counts[result.status] += 1
                completed += 1
            if completed % 100 == 0:
                connection.commit()
            now = time.monotonic()
            if completed % args.report_every == 0 or now - last_report_time >= 60:
                connection.commit()
                report = write_status_report(connection, dataset_dir, dataset, started_at)
                print(
                    f"[{dataset}] processed={completed:,} success={report['successful_media']:,}/"
                    f"{report['total_media']:,} pending={report['pending_media']:,} "
                    f"bytes={report['downloaded_bytes']:,}",
                    flush=True,
                )
                last_report_time = now
            if (completed - last_checkpoint_count >= args.checkpoint_every
                    or now - last_checkpoint_time >= 600):
                connection.commit()
                checkpoint_database(connection, state_path, checkpoint_path)
                last_checkpoint_count = completed
                last_checkpoint_time = now
    connection.commit()
    try:
        partial_root.rmdir()
    except OSError:
        pass
    report = write_status_report(connection, dataset_dir, dataset, started_at)
    report["this_run"] = {"processed": completed, "statuses": dict(sorted(runtime_counts.items()))}
    write_json_atomic(dataset_dir / "media_download_report.json", report)
    checkpoint_database(connection, state_path, checkpoint_path)
    return report


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, value


def rewrite_split(connection: sqlite3.Connection, dataset_dir: Path, split: str) -> dict[str, int]:
    source = dataset_dir / f"{split}.jsonl"
    destination = dataset_dir / f"local_{split}.jsonl"
    rejected = dataset_dir / f"local_{split}_rejected.jsonl"
    temporary_destination = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary_rejected = rejected.with_name(f".{rejected.name}.{os.getpid()}.tmp")
    counters: Counter[str] = Counter()
    cache: dict[str, tuple[str, str | None, str | None, bool]] = {}
    cache_limit = 200000
    try:
        with (temporary_destination.open("w", encoding="utf-8", newline="\n") as output_stream,
              temporary_rejected.open("w", encoding="utf-8", newline="\n") as rejected_stream):
            for line_number, record in iter_jsonl(source):
                counters["source_records"] += 1
                images = record.get("images")
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                    raise RuntimeError(f"Expected exactly one image at {source}:{line_number}")
                url = images[0]
                canonical = canonical_url(url)
                state = cache.get(canonical)
                if state is None:
                    row = connection.execute(
                        "SELECT status, local_path, error FROM media WHERE canonical_url = ?", (canonical, )).fetchone()
                    if row is None:
                        state = ("missing_inventory", None, "URL is absent from inventory", False)
                    else:
                        status, local_path, error = row
                        exists = bool(local_path and Path(local_path).is_file())
                        state = (status, local_path, error, exists)
                    if len(cache) >= cache_limit:
                        cache.clear()
                    cache[canonical] = state
                status, local_path, error, exists = state
                if status not in SUCCESS_STATUSES or not local_path or not exists:
                    counters[f"rejected_{status}"] += 1
                    rejected_stream.write(
                        json.dumps(
                            {
                                "split": split,
                                "line": line_number,
                                "url": url,
                                "status": status,
                                "error": error,
                            }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    continue
                record["images"] = [local_path]
                output_stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counters["written_records"] += 1
        os.replace(temporary_destination, destination)
        os.replace(temporary_rejected, rejected)
    finally:
        temporary_destination.unlink(missing_ok=True)
        temporary_rejected.unlink(missing_ok=True)
    return dict(sorted(counters.items()))


def write_download_manifest(connection: sqlite3.Connection, dataset_dir: Path) -> None:
    destination = dataset_dir / "media_download_manifest.jsonl"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    query = """
        SELECT canonical_url, source_url, status, local_path, attempts, http_status, size_bytes,
               actual_sha256, content_type, image_format, width, height, error
        FROM media ORDER BY canonical_url
    """
    fields = (
        "canonical_url", "source_url", "status", "local_path", "attempts", "http_status", "size_bytes",
        "actual_sha256", "content_type", "image_format", "width", "height", "error")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in connection.execute(query):
                stream.write(json.dumps(dict(zip(fields, row)), ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def rewrite_local_jsonl(
    connection: sqlite3.Connection,
    dataset_dir: Path,
    dataset: str,
    allow_incomplete: bool,
    splits: Sequence[str] = ("train", "val"),
) -> dict[str, Any]:
    status = database_status(connection)
    if status["pending_media"] and not allow_incomplete:
        raise RuntimeError(
            f"{dataset} still has {status['pending_media']:,} pending media; refusing an incomplete rewrite")
    split_reports = {split: rewrite_split(connection, dataset_dir, split) for split in splits}
    write_download_manifest(connection, dataset_dir)
    report = {
        "dataset": dataset,
        "completed_at": utc_now(),
        "media": status,
        "splits": split_reports,
    }
    write_json_atomic(dataset_dir / "localization_report.json", report)
    return report


def write_training_manifest(output_root: Path, datasets: Sequence[str], destination: Path) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for dataset in datasets:
        dataset_dir = output_root / dataset
        report_path = dataset_dir / "localization_report.json"
        with report_path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        split_reports = report.get("splits")
        if not isinstance(split_reports, dict) or "global_train" not in split_reports or "val" not in split_reports:
            raise RuntimeError(
                f"{dataset} localization report lacks global_train/val; rewrite those splits first"
            )
        train_path = (dataset_dir / "local_global_train.jsonl").resolve()
        eval_path = (dataset_dir / "local_val.jsonl").resolve()
        for label, path in (("train", train_path), ("eval", eval_path)):
            if not path.is_file():
                raise RuntimeError(f"{dataset} {label} entry does not exist: {path}")
        entries[dataset] = {
            "train": str(train_path),
            "eval": str(eval_path),
            "train_records": int(split_reports["global_train"]["written_records"]),
            "eval_records": int(split_reports["val"]["written_records"]),
            "successful_media": int(report["media"]["successful_media"]),
            "total_media": int(report["media"]["total_media"]),
            "localization_report": str(report_path.resolve()),
        }
    manifest = {
        "schema_version": 1,
        "status": "ready_partial_media",
        "generated_at": utc_now(),
        "datasets": entries,
    }
    write_json_atomic(destination, manifest)
    return manifest


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    dataset_dir = args.output_root / dataset
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Converted dataset directory does not exist: {dataset_dir}")
    state_root = args.state_root or args.output_root
    database_path = state_root / dataset / "media_download.sqlite3"
    checkpoint_path = dataset_dir / ".media_download.sqlite3.checkpoint"
    restore_database(database_path, checkpoint_path)
    connection = connect_database(database_path)
    try:
        if args.phase in {"all", "inventory"}:
            inventory = build_inventory(connection, args.source_root, dataset)
            print(f"[{dataset}] inventory: {json.dumps(inventory, sort_keys=True)}", flush=True)
            checkpoint_database(connection, database_path, checkpoint_path)
        if args.phase in {"download", "rewrite", "status"}:
            count = connection.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            if not count:
                raise RuntimeError(f"Inventory is empty for {dataset}; run --phase inventory first")
        if args.phase in {"all", "download"}:
            report = download_inventory(connection, database_path, checkpoint_path, dataset_dir, dataset, args)
            print(f"[{dataset}] download: {json.dumps(report, sort_keys=True)}", flush=True)
        if args.phase == "all" and args.max_media is not None:
            print(f"[{dataset}] smoke-test limit used; skipping local JSONL rewrite", flush=True)
        elif args.phase in {"all", "rewrite"}:
            report = rewrite_local_jsonl(
                connection,
                dataset_dir,
                dataset,
                args.allow_incomplete_rewrite,
                args.rewrite_splits,
            )
            print(f"[{dataset}] localization: {json.dumps(report, sort_keys=True)}", flush=True)
            checkpoint_database(connection, database_path, checkpoint_path)
        elif args.phase == "status":
            report = write_status_report(connection, dataset_dir, dataset)
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    for dataset in args.datasets:
        process_dataset(args, dataset)
    if args.training_manifest is not None:
        manifest = write_training_manifest(
            args.output_root,
            args.datasets,
            args.training_manifest.expanduser().resolve(),
        )
        print(f"[training-manifest] {json.dumps(manifest, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
