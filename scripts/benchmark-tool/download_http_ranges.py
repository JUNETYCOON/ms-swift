#!/usr/bin/env python3
"""Resume one HTTP object through independently retryable byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import requests


CONTENT_RANGE_RE = re.compile(r'^bytes (\d+)-(\d+)/(\d+)$')
USER_AGENT = 'ms-swift-range-downloader/1.0'


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    path: Path

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--sha256')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--retries', type=int, default=1000)
    parser.add_argument('--connect-timeout', type=float, default=30.0)
    parser.add_argument('--read-timeout', type=float, default=120.0)
    return parser


def partition_ranges(start: int, end: int, jobs: int, directory: Path) -> List[Segment]:
    if start < 0 or end < start or jobs <= 0:
        raise ValueError('Invalid byte range or worker count.')
    byte_count = end - start + 1
    chunk_size = math.ceil(byte_count / min(jobs, byte_count))
    segments = []
    offset = start
    while offset <= end:
        segment_end = min(offset + chunk_size - 1, end)
        segments.append(Segment(offset, segment_end, directory / f'{offset}-{segment_end}.part'))
        offset = segment_end + 1
    return segments


def parse_content_range(value: Optional[str]) -> Tuple[int, int, int]:
    match = CONTENT_RANGE_RE.fullmatch(value or '')
    if match is None:
        raise ValueError(f'Invalid Content-Range header: {value!r}')
    return tuple(int(item) for item in match.groups())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_segment(
        url: str, segment: Segment, total_size: int, retries: int,
        connect_timeout: float, read_timeout: float) -> None:
    segment.path.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    while True:
        current_size = segment.path.stat().st_size if segment.path.is_file() else 0
        if current_size == segment.size:
            return
        if current_size > segment.size:
            raise RuntimeError(f'{segment.path} exceeds its expected size {segment.size}.')

        request_start = segment.start + current_size
        try:
            with requests.get(
                    url,
                    headers={
                        'Range': f'bytes={request_start}-{segment.end}',
                        'User-Agent': USER_AGENT,
                    },
                    stream=True,
                    allow_redirects=True,
                    timeout=(connect_timeout, read_timeout),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f'Expected HTTP 206 for {request_start}-{segment.end}, got {response.status_code}.')
                received_start, received_end, received_total = parse_content_range(
                    response.headers.get('Content-Range'))
                if received_start != request_start or received_end > segment.end or received_total != total_size:
                    raise RuntimeError(
                        f'Unexpected Content-Range {response.headers.get("Content-Range")!r} for '
                        f'{request_start}-{segment.end}/{total_size}.')
                with segment.path.open('ab') as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            attempts = 0
        except (OSError, requests.RequestException, RuntimeError, ValueError) as error:
            attempts += 1
            if attempts > retries:
                raise RuntimeError(f'Range {segment.start}-{segment.end} failed: {error}') from error
            print(
                f'[retry] range={segment.start}-{segment.end} attempt={attempts} '
                f'bytes={current_size}/{segment.size}: {error}',
                flush=True,
            )
            time.sleep(min(2**min(attempts - 1, 3), 8))


def assemble_and_verify(
        prefix: Path, segments: Sequence[Segment], expected_size: int,
        expected_sha256: Optional[str]) -> Tuple[int, str]:
    assembled = prefix.with_name(prefix.name + '.assembled')
    with assembled.open('wb') as destination:
        if prefix.is_file():
            with prefix.open('rb') as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        for segment in segments:
            if segment.path.stat().st_size != segment.size:
                raise RuntimeError(f'Incomplete segment: {segment.path}')
            with segment.path.open('rb') as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    actual_size = assembled.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f'Assembled size {actual_size}, expected {expected_size}.')
    actual_sha256 = _sha256_file(assembled)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(f'Assembled SHA-256 {actual_sha256}, expected {expected_sha256.lower()}.')
    os.replace(str(assembled), str(prefix))
    return actual_size, actual_sha256


def run(args: argparse.Namespace) -> dict:
    if args.size <= 0 or args.jobs <= 0 or args.retries < 0:
        raise ValueError('Size and jobs must be positive; retries must be non-negative.')
    if args.sha256 and not re.fullmatch(r'[0-9a-fA-F]{64}', args.sha256):
        raise ValueError('--sha256 must be a 64-character hexadecimal digest.')
    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_size = output.stat().st_size if output.is_file() else 0
    if prefix_size > args.size:
        raise RuntimeError(f'{output} is larger than the expected object size.')

    segments: List[Segment] = []
    if prefix_size < args.size:
        segment_dir = output.with_name(output.name + f'.ranges-{prefix_size}')
        segments = partition_ranges(prefix_size, args.size - 1, args.jobs, segment_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(segments)) as executor:
            futures = [
                executor.submit(
                    download_segment, args.url, segment, args.size, args.retries,
                    args.connect_timeout, args.read_timeout)
                for segment in segments
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    actual_size, actual_sha256 = assemble_and_verify(output, segments, args.size, args.sha256)
    result = {
        'status': 'complete',
        'output': str(output),
        'prefix_bytes': prefix_size,
        'downloaded_range_count': len(segments),
        'size': actual_size,
        'sha256': actual_sha256,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        run(_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f'error: {error}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
