#!/usr/bin/env python3
"""Download and audit the Hy-Embodied benchmark bundle.

ModelScope repositories are downloaded through the public HTTP API because the
ModelScope CLI is not usable in the target runtime. Hugging Face repositories
use ``hf download`` so LFS files retain the client's retry and resume support.
Known restricted or unavailable media are reported explicitly and are never
silently promoted to a complete benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_MANIFEST = Path(__file__).with_name('hy_benchmark_sources.json')
DEFAULT_REPORT_NAME = '_hy_benchmark_download_report.json'
MODELSCOPE_API = 'https://www.modelscope.cn/api/v1/datasets'
MARKER_NAME = '.hy-benchmark-download.json'
USER_AGENT = 'ms-swift-hy-benchmark-downloader/1.0'

PROVIDERS = {'modelscope', 'huggingface', 'existing', 'reuse', 'git', 'manual', 'unavailable'}
CATEGORIES = {
    'action_relevant_state_understanding',
    'action_transition_reasoning',
    'sequential_adaptive_reasoning',
}
BENCHMARK_AVAILABILITY = {'complete', 'metadata_only', 'restricted', 'manual_composite', 'unavailable'}
SOURCE_AVAILABILITY = {
    'public',
    'existing_complete',
    'metadata_only',
    'manual_required',
    'restricted',
    'unavailable',
}
SUCCESS_SOURCE_STATUSES = {'complete', 'existing', 'reused'}


class ManifestError(ValueError):
    pass


class DownloadError(RuntimeError):
    pass


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--target-root', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--benchmark', action='append', default=[], help='Exact benchmark name; repeat as needed.')
    parser.add_argument('--category', action='append', choices=sorted(CATEGORIES), default=[])
    parser.add_argument('--provider', action='append', choices=sorted(PROVIDERS), default=[])
    parser.add_argument('--jobs', type=int, default=4, help='Concurrent ModelScope files / HF workers.')
    parser.add_argument('--timeout', type=float, default=90.0, help='HTTP timeout in seconds.')
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--hf-command', help='Path or command name for hf/huggingface-cli.')
    parser.add_argument('--refresh', action='store_true', help='Ignore matching completion markers.')
    parser.add_argument('--no-link-reuse', action='store_true', help='Verify reused data without creating a symlink.')
    parser.add_argument('--require-complete', action='store_true',
                        help='Return non-zero for known partial/restricted/unavailable benchmarks too.')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print actions without network or writes.')
    parser.add_argument('--list', action='store_true', help='List manifest benchmark mappings and exit.')
    return parser


def _safe_relative_path(value: Any, field: str) -> PurePosixPath:
    text = str(value or '').replace('\\', '/')
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
        raise ManifestError(f'{field} must be a safe non-empty relative path, got {value!r}.')
    if path.parts and ':' in path.parts[0]:
        raise ManifestError(f'{field} must not contain a drive prefix, got {value!r}.')
    return path


def _target_path(root: Path, relative: Any, field: str = 'target_dir') -> Path:
    path = _safe_relative_path(relative, field)
    return root.joinpath(*path.parts)


def validate_manifest(data: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ['Manifest root must be an object.']
    if data.get('schema_version') != 1:
        errors.append('schema_version must be 1.')
    sources = data.get('sources')
    benchmarks = data.get('benchmarks')
    if not isinstance(sources, list):
        errors.append('sources must be a list.')
        sources = []
    if not isinstance(benchmarks, list):
        errors.append('benchmarks must be a list.')
        benchmarks = []

    source_keys: Set[str] = set()
    target_dirs: Dict[str, str] = {}
    for index, source in enumerate(sources):
        prefix = f'sources[{index}]'
        if not isinstance(source, dict):
            errors.append(f'{prefix} must be an object.')
            continue
        key = source.get('key')
        if not isinstance(key, str) or not key:
            errors.append(f'{prefix}.key must be a non-empty string.')
        elif key in source_keys:
            errors.append(f'Duplicate source key: {key!r}.')
        else:
            source_keys.add(key)
        provider = source.get('provider')
        if provider not in PROVIDERS:
            errors.append(f'{prefix}.provider must be one of {sorted(PROVIDERS)}, got {provider!r}.')
        availability = source.get('availability')
        if availability not in SOURCE_AVAILABILITY:
            errors.append(
                f'{prefix}.availability must be one of {sorted(SOURCE_AVAILABILITY)}, got {availability!r}.')
        if provider in {'modelscope', 'huggingface'}:
            for field in ('repo_id', 'revision', 'target_dir'):
                if not isinstance(source.get(field), str) or not source[field]:
                    errors.append(f'{prefix}.{field} must be a non-empty string for {provider}.')
        elif provider == 'git':
            for field in ('repo_url', 'revision', 'target_dir'):
                if not isinstance(source.get(field), str) or not source[field]:
                    errors.append(f'{prefix}.{field} must be a non-empty string for git.')
        elif provider == 'existing':
            if not isinstance(source.get('target_dir'), str) or not source['target_dir']:
                errors.append(f'{prefix}.target_dir must be set for existing data.')
        elif provider == 'reuse':
            for field in ('target_dir', 'reuse_path'):
                if not isinstance(source.get(field), str) or not source[field]:
                    errors.append(f'{prefix}.{field} must be set for reused data.')

        target_dir = source.get('target_dir')
        if target_dir:
            try:
                normalized = _safe_relative_path(target_dir, f'{prefix}.target_dir').as_posix()
                previous = target_dirs.get(normalized)
                if previous is not None:
                    errors.append(
                        f'Sources {previous!r} and {key!r} share target_dir {normalized!r}; '
                        'shared benchmarks must reference one source key.')
                elif isinstance(key, str):
                    target_dirs[normalized] = key
            except ManifestError as error:
                errors.append(str(error))
        for file_index, required in enumerate(source.get('required_files') or []):
            required_prefix = f'{prefix}.required_files[{file_index}]'
            if not isinstance(required, dict):
                errors.append(f'{required_prefix} must be an object.')
                continue
            try:
                _safe_relative_path(required.get('path'), f'{required_prefix}.path')
            except ManifestError as error:
                errors.append(str(error))
            size = required.get('size')
            if size is not None and (not isinstance(size, int) or size < 0):
                errors.append(f'{required_prefix}.size must be a non-negative integer.')
            sha256 = required.get('sha256')
            if sha256 is not None and (
                    not isinstance(sha256, str) or len(sha256) != 64
                    or any(char not in '0123456789abcdefABCDEF' for char in sha256)):
                errors.append(f'{required_prefix}.sha256 must be a 64-character hexadecimal digest.')
        for path_index, required_path in enumerate(source.get('required_paths') or []):
            try:
                _safe_relative_path(required_path, f'{prefix}.required_paths[{path_index}]')
            except ManifestError as error:
                errors.append(str(error))

    benchmark_names: Set[str] = set()
    referenced_sources: Set[str] = set()
    for index, benchmark in enumerate(benchmarks):
        prefix = f'benchmarks[{index}]'
        if not isinstance(benchmark, dict):
            errors.append(f'{prefix} must be an object.')
            continue
        name = benchmark.get('name')
        if not isinstance(name, str) or not name:
            errors.append(f'{prefix}.name must be a non-empty string.')
        elif name.casefold() in benchmark_names:
            errors.append(f'Duplicate benchmark name (case-insensitive): {name!r}.')
        else:
            benchmark_names.add(name.casefold())
        if benchmark.get('category') not in CATEGORIES:
            errors.append(f'{prefix}.category is invalid: {benchmark.get("category")!r}.')
        if benchmark.get('availability') not in BENCHMARK_AVAILABILITY:
            errors.append(f'{prefix}.availability is invalid: {benchmark.get("availability")!r}.')
        source_refs = benchmark.get('source_keys')
        if not isinstance(source_refs, list) or not source_refs:
            errors.append(f'{prefix}.source_keys must be a non-empty list.')
            continue
        for source_key in source_refs:
            if source_key not in source_keys:
                errors.append(f'{prefix} references unknown source {source_key!r}.')
            else:
                referenced_sources.add(source_key)
    for source_key in sorted(source_keys - referenced_sources):
        errors.append(f'Source {source_key!r} is not referenced by any benchmark.')
    return errors


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f'Cannot read manifest {path}: {error}') from error
    errors = validate_manifest(data)
    if errors:
        raise ManifestError('Invalid manifest:\n- ' + '\n- '.join(errors))
    return data


def select_benchmarks(
        manifest: Mapping[str, Any], names: Sequence[str], categories: Sequence[str]) -> List[Dict[str, Any]]:
    benchmarks = list(manifest['benchmarks'])
    by_name = {benchmark['name'].casefold(): benchmark for benchmark in benchmarks}
    unknown = [name for name in names if name.casefold() not in by_name]
    if unknown:
        raise ManifestError(f'Unknown benchmark name(s): {", ".join(unknown)}')
    selected_names = {name.casefold() for name in names}
    selected_categories = set(categories)
    selected = []
    for benchmark in benchmarks:
        if selected_names and benchmark['name'].casefold() not in selected_names:
            continue
        if selected_categories and benchmark['category'] not in selected_categories:
            continue
        selected.append(benchmark)
    if not selected:
        raise ManifestError('No benchmarks matched the requested filters.')
    return selected


def selected_sources(
        manifest: Mapping[str, Any], benchmarks: Sequence[Mapping[str, Any]], providers: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = {key for benchmark in benchmarks for key in benchmark['source_keys']}
    provider_filter = set(providers)
    sources = []
    for source in manifest['sources']:
        if source['key'] not in wanted:
            continue
        if provider_filter and source['provider'] not in provider_filter:
            continue
        sources.append(source)
    return sources


def _request_json(url: str, timeout: float, retries: int) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 8))
    raise DownloadError(f'HTTP JSON request failed after {retries + 1} attempt(s): {url}: {last_error}')


def _modelscope_tree_url(repo_id: str, revision: str, root: str, page: int, page_size: int) -> str:
    query = urlencode({
        'Revision': revision,
        'Root': root,
        'PageNumber': page,
        'PageSize': page_size,
    })
    return f'{MODELSCOPE_API}/{quote(repo_id, safe="/")}/repo/tree?{query}'


def list_modelscope_files(source: Mapping[str, Any], timeout: float, retries: int) -> List[Dict[str, Any]]:
    repo_id = source['repo_id']
    revision = source['revision']
    directories = ['']
    seen_directories: Set[str] = set()
    files: Dict[str, Dict[str, Any]] = {}
    page_size = 100
    while directories:
        root = directories.pop()
        if root in seen_directories:
            continue
        seen_directories.add(root)
        page = 1
        while True:
            payload = _request_json(
                _modelscope_tree_url(repo_id, revision, root, page, page_size), timeout, retries)
            if payload.get('Code') != 200 or not isinstance(payload.get('Data'), dict):
                raise DownloadError(
                    f'ModelScope tree request failed for {repo_id!r} root={root!r}: '
                    f'Code={payload.get("Code")!r}, Message={payload.get("Message")!r}')
            data = payload['Data']
            entries = data.get('Files')
            if not isinstance(entries, list):
                raise DownloadError(f'ModelScope returned no Files list for {repo_id!r} root={root!r}.')
            for entry in entries:
                if not isinstance(entry, dict):
                    raise DownloadError(f'ModelScope returned an invalid tree entry for {repo_id!r}.')
                try:
                    path = _safe_relative_path(entry.get('Path'), 'ModelScope file path').as_posix()
                except ManifestError as error:
                    raise DownloadError(str(error)) from error
                entry_type = entry.get('Type')
                if entry_type == 'tree':
                    directories.append(path)
                elif entry_type == 'blob':
                    size = entry.get('Size')
                    if not isinstance(size, int) or size < 0:
                        raise DownloadError(f'ModelScope file {path!r} has invalid size {size!r}.')
                    candidate = {
                        'path': path,
                        'size': size,
                        'sha256': str(entry.get('Sha256') or '').lower(),
                    }
                    previous = files.get(path)
                    if previous is not None and previous != candidate:
                        raise DownloadError(f'ModelScope returned conflicting metadata for {path!r}.')
                    files[path] = candidate
                else:
                    raise DownloadError(
                        f'ModelScope returned unsupported entry type {entry_type!r} for {path!r}.')
            total = data.get('TotalCount', payload.get('TotalCount', len(entries)))
            if not isinstance(total, int) or total < 0:
                total = len(entries)
            if page * page_size >= total:
                break
            page += 1
    if not files:
        raise DownloadError(f'ModelScope repository {repo_id!r} has no downloadable files.')
    return [files[path] for path in sorted(files)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_digest(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value['path']):
        line = f'{item["path"]}\0{item.get("size", "")}\0{item.get("sha256", "")}\n'
        digest.update(line.encode('utf-8'))
    return digest.hexdigest()


def _modelscope_download_url(source: Mapping[str, Any], remote_path: str) -> str:
    query = urlencode({'Revision': source['revision'], 'FilePath': remote_path})
    return f'{MODELSCOPE_API}/{quote(source["repo_id"], safe="/")}/repo?{query}'


def _verify_download(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DownloadError(f'{path} has {actual_size} bytes, expected {expected_size}.')
    if expected_sha256:
        actual_sha256 = _sha256_file(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise DownloadError(
                f'{path} has SHA-256 {actual_sha256}, expected {expected_sha256.lower()}.')


def _download_modelscope_file(
        source: Mapping[str, Any], item: Mapping[str, Any], target_root: Path,
        timeout: float, retries: int) -> Dict[str, Any]:
    remote_path = item['path']
    expected_size = item['size']
    expected_sha256 = item.get('sha256') or ''
    target = _target_path(target_root, remote_path, 'ModelScope file path')
    if target.is_file() and target.stat().st_size == expected_size:
        try:
            _verify_download(target, expected_size, expected_sha256)
            return {'path': remote_path, 'status': 'verified_existing', 'size': expected_size}
        except DownloadError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    if partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    if partial.is_file() and partial.stat().st_size == expected_size:
        try:
            _verify_download(partial, expected_size, expected_sha256)
            os.replace(str(partial), str(target))
            return {'path': remote_path, 'status': 'resumed_existing', 'size': expected_size}
        except DownloadError:
            partial.unlink()

    url = _modelscope_download_url(source, remote_path)
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        start = partial.stat().st_size if partial.is_file() else 0
        headers = {'User-Agent': USER_AGENT, 'Accept': 'application/octet-stream'}
        if start:
            headers['Range'] = f'bytes={start}-'
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                status_code = getattr(response, 'status', None) or response.getcode()
                append = start > 0 and status_code == 206
                mode = 'ab' if append else 'wb'
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
            _verify_download(partial, expected_size, expected_sha256)
            os.replace(str(partial), str(target))
            return {
                'path': remote_path,
                'status': 'resumed' if start and append else 'downloaded',
                'size': expected_size,
            }
        except (HTTPError, URLError, TimeoutError, OSError, DownloadError) as error:
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 8))
    raise DownloadError(f'Failed to download ModelScope file {remote_path!r}: {last_error}')


def _tree_stats(root: Path) -> Tuple[int, int]:
    file_count = 0
    total_bytes = 0
    if not root.is_dir():
        return file_count, total_bytes
    for index, (directory, directory_names, file_names) in enumerate(os.walk(root)):
        if index == 0:
            directory_names[:] = [
                name for name in directory_names if name not in {'.git', '.cache'}
            ]
        for file_name in file_names:
            if file_name == MARKER_NAME:
                continue
            path = Path(directory, file_name)
            try:
                file_stat = path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            file_count += 1
            total_bytes += file_stat.st_size
    return file_count, total_bytes


def _read_marker(target: Path, source: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    marker_path = target / MARKER_NAME
    try:
        with marker_path.open('r', encoding='utf-8') as stream:
            marker = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        'schema_version': 1,
        'source_key': source['key'],
        'provider': source['provider'],
        'revision': source.get('revision'),
        'download_status': 'complete',
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        return None
    file_count, total_bytes = _tree_stats(target)
    if marker.get('file_count') != file_count or marker.get('total_bytes') != total_bytes:
        return None
    return marker


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(str(temporary), str(path))


def _write_marker(
        target: Path, source: Mapping[str, Any], file_count: int, total_bytes: int,
        inventory_sha256: Optional[str] = None) -> Dict[str, Any]:
    marker = {
        'schema_version': 1,
        'source_key': source['key'],
        'provider': source['provider'],
        'repo_id': source.get('repo_id'),
        'repo_url': source.get('repo_url'),
        'revision': source.get('revision'),
        'availability': source['availability'],
        'download_status': 'complete',
        'file_count': file_count,
        'total_bytes': total_bytes,
        'completed_at': _utc_now(),
    }
    if inventory_sha256:
        marker['inventory_sha256'] = inventory_sha256
    _write_json_atomic(target / MARKER_NAME, marker)
    return marker


def _materialized_status(source: Mapping[str, Any], existing: bool = False) -> str:
    availability = source['availability']
    if availability == 'metadata_only':
        return 'metadata_only'
    if existing:
        return 'existing'
    return 'complete'


def download_modelscope_source(
        source: Mapping[str, Any], target_root: Path, jobs: int, timeout: float,
        retries: int, refresh: bool) -> Dict[str, Any]:
    target = _target_path(target_root, source['target_dir'])
    if not refresh:
        marker = _read_marker(target, source)
        if marker is not None:
            return {
                'status': _materialized_status(source),
                'action': 'skipped_matching_marker',
                'target': str(target),
                'file_count': marker['file_count'],
                'total_bytes': marker['total_bytes'],
            }
    files = list_modelscope_files(source, timeout, retries)
    print(
        f'[modelscope] {source["repo_id"]}@{source["revision"]}: '
        f'{len(files)} files, {sum(item["size"] for item in files)} bytes',
        flush=True)
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _download_modelscope_file, source, item, target, timeout, retries): item
            for item in files
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # keep downloading independent files for a useful retry state
                errors.append(f'{item["path"]}: {error}')
            completed += 1
            if completed == len(files) or completed % 100 == 0:
                print(f'[modelscope] {source["key"]}: {completed}/{len(files)} files checked', flush=True)
    if errors:
        preview = '\n'.join(errors[:10])
        raise DownloadError(
            f'{len(errors)} ModelScope file(s) failed for {source["repo_id"]}:\n{preview}')
    file_count, total_bytes = _tree_stats(target)
    marker = _write_marker(
        target, source, file_count, total_bytes, inventory_sha256=_inventory_digest(files))
    return {
        'status': _materialized_status(source),
        'action': 'downloaded_and_verified',
        'target': str(target),
        'file_count': file_count,
        'total_bytes': total_bytes,
        'downloaded_files': sum(item['status'] == 'downloaded' for item in results),
        'resumed_files': sum(item['status'] in {'resumed', 'resumed_existing'} for item in results),
        'verified_existing_files': sum(item['status'] == 'verified_existing' for item in results),
        'inventory_sha256': marker['inventory_sha256'],
    }


def _find_hf_command(explicit: Optional[str]) -> Tuple[str, str]:
    if explicit:
        path = shutil.which(explicit) or explicit
        flavor = 'huggingface-cli' if Path(explicit).name.startswith('huggingface-cli') else 'hf'
        return path, flavor
    for candidate in ('hf', 'huggingface-cli'):
        path = shutil.which(candidate)
        if path:
            return path, candidate
    raise DownloadError('Neither hf nor huggingface-cli is available on PATH.')


def _run_command(command: Sequence[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command), check=False, text=True, capture_output=capture_output)


def download_huggingface_source(
        source: Mapping[str, Any], target_root: Path, jobs: int,
        hf_command: Optional[str], refresh: bool) -> Dict[str, Any]:
    target = _target_path(target_root, source['target_dir'])
    if not refresh:
        marker = _read_marker(target, source)
        if marker is not None:
            return {
                'status': _materialized_status(source),
                'action': 'skipped_matching_marker',
                'target': str(target),
                'file_count': marker['file_count'],
                'total_bytes': marker['total_bytes'],
            }
    executable, flavor = _find_hf_command(hf_command)
    target.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        'download',
        source['repo_id'],
        '--repo-type',
        'dataset',
        '--revision',
        source['revision'],
        '--local-dir',
        str(target),
    ]
    if flavor == 'hf':
        command.extend(['--max-workers', str(jobs)])
    for pattern in source.get('include') or []:
        command.extend(['--include', pattern])
    print(f'[huggingface] {source["repo_id"]}@{source["revision"]} -> {target}', flush=True)
    result = _run_command(command)
    if result.returncode != 0:
        raise DownloadError(
            f'Hugging Face command failed with exit code {result.returncode}: '
            + ' '.join(command))
    file_count, total_bytes = _tree_stats(target)
    if file_count == 0:
        raise DownloadError(f'Hugging Face command succeeded but {target} contains no files.')
    _write_marker(target, source, file_count, total_bytes)
    return {
        'status': _materialized_status(source),
        'action': 'downloaded_and_client_verified',
        'target': str(target),
        'file_count': file_count,
        'total_bytes': total_bytes,
    }


def download_git_source(
        source: Mapping[str, Any], target_root: Path, refresh: bool) -> Dict[str, Any]:
    target = _target_path(target_root, source['target_dir'])
    if not refresh:
        marker = _read_marker(target, source)
        if marker is not None:
            return {
                'status': _materialized_status(source),
                'action': 'skipped_matching_marker',
                'target': str(target),
                'file_count': marker['file_count'],
                'total_bytes': marker['total_bytes'],
            }
    target.parent.mkdir(parents=True, exist_ok=True)
    newly_cloned = False
    if not target.exists():
        clone = _run_command([
            'git', 'clone', '--filter=blob:none', '--no-checkout', source['repo_url'], str(target)
        ])
        if clone.returncode != 0:
            raise DownloadError(f'git clone failed with exit code {clone.returncode}: {source["repo_url"]}')
        newly_cloned = True
    if not (target / '.git').is_dir():
        raise DownloadError(f'Git target exists but is not a repository: {target}')
    if not newly_cloned:
        dirty = _run_command(['git', '-C', str(target), 'status', '--porcelain'], capture_output=True)
        if dirty.returncode != 0:
            raise DownloadError(f'Cannot inspect git repository: {target}')
        dirty_lines = [
            line for line in dirty.stdout.splitlines()
            if line[3:].strip('"') != MARKER_NAME
        ]
        if dirty_lines:
            raise DownloadError(f'Git repository has local changes and will not be overwritten: {target}')
    fetch = _run_command([
        'git', '-C', str(target), 'fetch', '--depth', '1', 'origin', source['revision']
    ])
    if fetch.returncode != 0:
        raise DownloadError(f'git fetch failed for {source["repo_url"]}@{source["revision"]}.')
    checkout = _run_command(['git', '-C', str(target), 'checkout', '--detach', source['revision']])
    if checkout.returncode != 0:
        raise DownloadError(f'git checkout failed for {source["repo_url"]}@{source["revision"]}.')
    head = _run_command(['git', '-C', str(target), 'rev-parse', 'HEAD'], capture_output=True)
    if head.returncode != 0 or head.stdout.strip().lower() != source['revision'].lower():
        raise DownloadError(f'Git HEAD verification failed for {target}.')
    file_count, total_bytes = _tree_stats(target)
    _write_marker(target, source, file_count, total_bytes)
    return {
        'status': _materialized_status(source),
        'action': 'cloned_and_pinned',
        'target': str(target),
        'file_count': file_count,
        'total_bytes': total_bytes,
    }


def _verify_required_data(root: Path, source: Mapping[str, Any]) -> Dict[str, Any]:
    problems: List[str] = []
    for required in source.get('required_files') or []:
        path = _target_path(root, required['path'], 'required file path')
        if not path.is_file():
            problems.append(f'missing file: {path}')
            continue
        expected_size = required.get('size')
        if expected_size is not None and path.stat().st_size != expected_size:
            problems.append(
                f'wrong size: {path} ({path.stat().st_size}, expected {expected_size})')
            continue
        expected_sha256 = required.get('sha256')
        if expected_sha256 and _sha256_file(path).lower() != expected_sha256.lower():
            problems.append(f'wrong SHA-256: {path}')
    for relative in source.get('required_paths') or []:
        path = _target_path(root, relative, 'required path')
        if not path.exists():
            problems.append(f'missing path: {path}')
    file_count, total_bytes = _tree_stats(root)
    minimum_files = source.get('minimum_files')
    if isinstance(minimum_files, int) and file_count < minimum_files:
        problems.append(f'{root} has {file_count} files, expected at least {minimum_files}.')
    if problems:
        raise DownloadError('; '.join(problems))
    return {'file_count': file_count, 'total_bytes': total_bytes}


def verify_existing_source(source: Mapping[str, Any], target_root: Path) -> Dict[str, Any]:
    target = _target_path(target_root, source['target_dir'])
    stats = _verify_required_data(target, source)
    return {
        'status': 'existing',
        'action': 'verified_existing_no_download',
        'target': str(target),
        **stats,
    }


def reuse_source(source: Mapping[str, Any], target_root: Path, link_reuse: bool) -> Dict[str, Any]:
    reuse_path = Path(source['reuse_path'])
    stats = _verify_required_data(reuse_path, source)
    target = _target_path(target_root, source['target_dir'])
    action = 'verified_external_reuse'
    if link_reuse:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != reuse_path.resolve():
                raise DownloadError(f'Existing symlink {target} does not point to {reuse_path}.')
            action = 'verified_existing_reuse_symlink'
        elif target.exists():
            if target.resolve() != reuse_path.resolve():
                raise DownloadError(f'Reuse target already exists and will not be replaced: {target}')
            action = 'verified_existing_reuse_target'
        else:
            target.symlink_to(reuse_path, target_is_directory=True)
            action = 'created_reuse_symlink'
    return {
        'status': _materialized_status(source),
        'action': action,
        'target': str(target) if link_reuse else str(reuse_path),
        'reused_from': str(reuse_path),
        **stats,
    }


def _planned_source(source: Mapping[str, Any], target_root: Path, link_reuse: bool) -> Dict[str, Any]:
    provider = source['provider']
    if provider == 'manual':
        status = source['availability']
        action = 'manual_action_required'
        target = None
    elif provider == 'unavailable':
        status = 'unavailable'
        action = 'no_public_source'
        target = None
    elif provider == 'existing':
        status = 'planned'
        action = 'would_verify_existing'
        target = str(_target_path(target_root, source['target_dir']))
    elif provider == 'reuse':
        status = 'planned'
        action = 'would_verify_and_link_reuse' if link_reuse else 'would_verify_reuse'
        target = str(_target_path(target_root, source['target_dir']))
    else:
        status = 'planned'
        action = 'would_download'
        target = str(_target_path(target_root, source['target_dir']))
    return {'status': status, 'action': action, 'target': target}


def process_source(
        source: Mapping[str, Any], target_root: Path, jobs: int, timeout: float,
        retries: int, hf_command: Optional[str], refresh: bool,
        link_reuse: bool, dry_run: bool) -> Dict[str, Any]:
    base = {
        'key': source['key'],
        'provider': source['provider'],
        'repo_id': source.get('repo_id'),
        'revision': source.get('revision'),
        'availability': source['availability'],
        'note': source.get('note'),
    }
    if dry_run:
        return {**base, **_planned_source(source, target_root, link_reuse)}
    provider = source['provider']
    if provider == 'modelscope':
        result = download_modelscope_source(source, target_root, jobs, timeout, retries, refresh)
    elif provider == 'huggingface':
        result = download_huggingface_source(source, target_root, jobs, hf_command, refresh)
    elif provider == 'git':
        result = download_git_source(source, target_root, refresh)
    elif provider == 'existing':
        result = verify_existing_source(source, target_root)
    elif provider == 'reuse':
        result = reuse_source(source, target_root, link_reuse)
    elif provider == 'manual':
        result = {'status': source['availability'], 'action': 'manual_action_required', 'target': None}
    elif provider == 'unavailable':
        result = {'status': 'unavailable', 'action': 'no_public_source', 'target': None}
    else:  # guarded by manifest validation
        raise DownloadError(f'Unsupported provider: {provider}')
    return {**base, **result}


def benchmark_result(
        benchmark: Mapping[str, Any], source_results: Mapping[str, Mapping[str, Any]],
        dry_run: bool) -> Dict[str, Any]:
    statuses = {
        key: source_results[key]['status']
        for key in benchmark['source_keys']
        if key in source_results
    }
    missing = [key for key in benchmark['source_keys'] if key not in source_results]
    expected = benchmark['availability']
    if missing:
        status = 'not_selected'
    elif any(value == 'failed' for value in statuses.values()):
        status = 'failed'
    elif expected == 'unavailable':
        status = 'unavailable'
    elif expected == 'restricted':
        status = 'restricted'
    elif expected == 'manual_composite':
        status = 'manual_required'
    elif expected == 'metadata_only':
        status = 'metadata_only'
    elif dry_run:
        status = 'planned'
    elif all(value in SUCCESS_SOURCE_STATUSES for value in statuses.values()):
        status = 'complete'
    else:
        status = 'partial'
    return {
        'name': benchmark['name'],
        'category': benchmark['category'],
        'expected_availability': expected,
        'status': status,
        'subset': benchmark.get('subset'),
        'source_statuses': statuses,
        'missing_filtered_sources': missing,
    }


def _print_manifest(manifest: Mapping[str, Any]) -> None:
    sources = {source['key']: source for source in manifest['sources']}
    for benchmark in manifest['benchmarks']:
        mappings = ', '.join(
            f'{sources[key]["provider"]}:{sources[key].get("repo_id") or sources[key].get("target_dir") or key}'
            for key in benchmark['source_keys'])
        print(
            f'{benchmark["name"]}\t{benchmark["category"]}\t'
            f'{benchmark["availability"]}\t{mappings}')


def run(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    if args.jobs <= 0:
        raise ManifestError('--jobs must be positive.')
    if args.retries < 0:
        raise ManifestError('--retries must be non-negative.')
    if args.timeout <= 0:
        raise ManifestError('--timeout must be positive.')
    manifest = load_manifest(args.manifest)
    if args.list:
        _print_manifest(manifest)
        return {'status': 'listed'}, 0
    target_root = args.target_root or Path(manifest['default_target_root'])
    target_root = target_root.expanduser().absolute()
    benchmarks = select_benchmarks(manifest, args.benchmark, args.category)
    sources = selected_sources(manifest, benchmarks, args.provider)
    started_at = _utc_now()
    source_results: List[Dict[str, Any]] = []
    for source in sources:
        print(f'[source] {source["key"]} ({source["provider"]})', flush=True)
        try:
            result = process_source(
                source=source,
                target_root=target_root,
                jobs=args.jobs,
                timeout=args.timeout,
                retries=args.retries,
                hf_command=args.hf_command,
                refresh=args.refresh,
                link_reuse=not args.no_link_reuse,
                dry_run=args.dry_run,
            )
        except Exception as error:
            result = {
                'key': source['key'],
                'provider': source['provider'],
                'repo_id': source.get('repo_id'),
                'revision': source.get('revision'),
                'availability': source['availability'],
                'status': 'failed',
                'action': 'failed',
                'error': f'{type(error).__name__}: {error}',
            }
            print(f'[failed] {source["key"]}: {error}', file=sys.stderr, flush=True)
        source_results.append(result)
    source_by_key = {source['key']: source for source in source_results}
    benchmark_results = [benchmark_result(item, source_by_key, args.dry_run) for item in benchmarks]
    summary: Dict[str, int] = {}
    for item in benchmark_results:
        summary[item['status']] = summary.get(item['status'], 0) + 1
    report = {
        'schema_version': 1,
        'status': 'failed' if any(item['status'] == 'failed' for item in source_results) else 'complete',
        'dry_run': bool(args.dry_run),
        'manifest': str(args.manifest.absolute()),
        'target_root': str(target_root),
        'started_at': started_at,
        'finished_at': _utc_now(),
        'selected_benchmark_count': len(benchmarks),
        'selected_source_count': len(sources),
        'source_results': source_results,
        'benchmark_results': benchmark_results,
        'benchmark_status_counts': summary,
    }
    if not args.dry_run:
        report_path = args.report or target_root / DEFAULT_REPORT_NAME
        _write_json_atomic(report_path, report)
        print(f'[report] {report_path}', flush=True)
    failures = any(item['status'] == 'failed' for item in source_results)
    incomplete = any(item['status'] != 'complete' for item in benchmark_results)
    return report, 1 if failures or (args.require_complete and incomplete) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, exit_code = run(args)
    except (ManifestError, DownloadError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    if not args.list:
        print(json.dumps({
            'status': report['status'],
            'dry_run': report['dry_run'],
            'benchmarks': report['selected_benchmark_count'],
            'sources': report['selected_source_count'],
            'benchmark_status_counts': report['benchmark_status_counts'],
        }, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
