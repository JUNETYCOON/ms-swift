#!/usr/bin/env python3
"""Convert existing PixMo-Points ms-swift jsonl files from norm1 to norm1000.

The original converter wrote PixMo-Points assistant text as norm1000-like
coordinates while objects.bbox stayed in norm1. This cleaner makes the
grounding payload internally consistent:

  objects.bbox      [0, 1]  -> [0, 1000]
  objects.bbox_type norm1   -> norm1000

It is intentionally dataset-scoped and does not touch other datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"bbox value is not numeric: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"bbox value is not finite: {value!r}")
    return number


def norm1_to_norm1000_box(box: Any) -> list[int]:
    if not isinstance(box, list) or len(box) not in {2, 4}:
        raise ValueError(f"bbox must be a 2-point or 4-box list: {box!r}")
    converted: list[int] = []
    for value in box:
        number = finite_number(value)
        if number < 0 or number > 1:
            raise ValueError(f"norm1 bbox coordinate out of range [0,1]: {box!r}")
        converted.append(int(round(number * 1000)))
    if len(converted) == 4 and (converted[2] < converted[0] or converted[3] < converted[1]):
        raise ValueError(f"bbox coordinate order is invalid after conversion: {box!r}")
    return converted


def convert_record(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return record, False
    bbox_type = objects.get("bbox_type", "real")
    if bbox_type != "norm1":
        return record, False
    boxes = objects.get("bbox") or []
    if not isinstance(boxes, list):
        raise ValueError("objects.bbox must be a list")
    objects["bbox"] = [norm1_to_norm1000_box(box) for box in boxes]
    objects["bbox_type"] = "norm1000"
    return record, True


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))
    raise FileNotFoundError(input_path)


def output_path_for(input_file: Path, input_root: Path, output_root: Path | None, suffix: str) -> Path:
    if output_root is None:
        return input_file.with_name(f"{input_file.stem}{suffix}{input_file.suffix}")
    if input_root.is_file():
        return output_root / input_file.name
    return output_root / input_file.relative_to(input_root)


def convert_file(input_file: Path, input_root: Path, output_root: Path | None, suffix: str) -> dict[str, Any]:
    out_file = output_path_for(input_file, input_root, output_root, suffix)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    converted = 0
    unchanged = 0
    errors: list[dict[str, Any]] = []
    with input_file.open("r", encoding="utf-8") as src, out_file.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                record, changed = convert_record(record)
                converted += int(changed)
                unchanged += int(not changed)
                dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as exc:
                errors.append({"line_no": line_no, "error": str(exc)})
                if len(errors) > 20:
                    raise RuntimeError(f"too many errors while converting {input_file}") from exc
    return {
        "input": str(input_file),
        "output": str(out_file),
        "rows": rows,
        "converted_norm1_to_norm1000": converted,
        "unchanged": unchanged,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/luojunkun/stage1/dataset_ms-swift/pixmo-points"),
        help="PixMo-Points jsonl file or directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. If omitted, writes sibling files with --suffix.",
    )
    parser.add_argument("--suffix", default=".norm1000", help="Suffix used when --output-root is omitted.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace input files after writing converted temporary files. A .bak is kept.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults to output root/input dir conversion_manifest.pixmo_points_norm1000.json.",
    )
    args = parser.parse_args()

    input_root = args.input
    files = iter_input_files(input_root)
    if not files:
        raise SystemExit(f"No jsonl files found under {input_root}")

    effective_output_root = args.output_root
    if args.replace and effective_output_root is None:
        effective_output_root = input_root.parent / f"{input_root.name}.norm1000_tmp" if input_root.is_dir() else input_root.parent

    results = [convert_file(path, input_root, effective_output_root, args.suffix) for path in files]

    if args.replace:
        for result in results:
            input_file = Path(result["input"])
            output_file = Path(result["output"])
            backup = input_file.with_suffix(input_file.suffix + ".bak_norm1")
            shutil.copy2(input_file, backup)
            shutil.move(str(output_file), str(input_file))
            result["backup"] = str(backup)
            result["replaced"] = True

    manifest = args.manifest
    if manifest is None:
        base = effective_output_root or (input_root if input_root.is_dir() else input_root.parent)
        manifest = base / "conversion_manifest.pixmo_points_norm1000.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(input_root),
        "output_root": str(effective_output_root) if effective_output_root else None,
        "replace": args.replace,
        "files": results,
        "total_rows": sum(item["rows"] for item in results),
        "total_converted_norm1_to_norm1000": sum(item["converted_norm1_to_norm1000"] for item in results),
        "total_errors": sum(len(item["errors"]) for item in results),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["total_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
