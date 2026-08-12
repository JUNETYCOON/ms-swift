"""Streaming JSON/JSONL helpers with atomic output replacement."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, TextIO, Tuple


def _reject_nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number {!r} is not allowed".format(value))


def parse_json_strict(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_nonfinite)


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = parse_json_strict(line)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError("Invalid JSON at {}:{}: {}".format(path, line_number, error)) from error
            if not isinstance(value, dict):
                raise ValueError("JSONL record at {}:{} must be an object".format(path, line_number))
            yield line_number, value


def json_line(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


@contextlib.contextmanager
def atomic_text_writer(path: Path, overwrite: bool = False) -> Iterator[TextIO]:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError("Output already exists: {}. Use --overwrite to replace it.".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent), text=True
    )
    temporary = Path(temporary_name)
    stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
    try:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        os.replace(str(temporary), str(path))
    except BaseException:
        if not stream.closed:
            stream.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], overwrite: bool = False) -> int:
    count = 0
    with atomic_text_writer(path, overwrite=overwrite) as stream:
        for record in records:
            stream.write(json_line(record))
            count += 1
    return count
