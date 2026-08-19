#!/usr/bin/env python3
"""Validate converted-dataset audit structure, assets, hashes, and HTML links."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


class ValidationError(Exception):
    pass


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sample_cards = 0
        self.references: list[str] = []
        self.asset_references: list[str] = []
        self.has_viewport = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if "sample" in values.get("class", "").split():
            self.sample_cards += 1
        if tag == "meta" and values.get("name", "").lower() == "viewport":
            self.has_viewport = True
        for attribute in ("href", "src"):
            if values.get(attribute):
                self.references.append(values[attribute])
                if attribute == "src":
                    self.asset_references.append(values[attribute])


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"invalid JSON: {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except Exception as exc:
        raise ValidationError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc
    return rows


def safe_path(root: Path, raw_value: str, label: str) -> Path:
    relative = Path(raw_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError(f"unsafe {label}: {raw_value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationError(f"escaping {label}: {raw_value}") from exc
    if not resolved.is_file():
        raise ValidationError(f"missing {label}: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        raise ValidationError(f"image decode failed: {path}: {exc}") from exc


def parse_html(path: Path) -> ReportParser:
    parser = ReportParser()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"invalid HTML: {path}: {exc}") from exc
    if not parser.has_viewport:
        raise ValidationError(f"missing viewport: {path}")
    for raw in parser.references:
        parts = urlsplit(raw)
        if parts.scheme or parts.netloc or raw.startswith("#"):
            if parts.scheme in {"http", "https"}:
                raise ValidationError(f"external HTML dependency/link: {path}: {raw}")
            continue
        decoded = unquote(parts.path)
        if not decoded:
            continue
        reference = Path(decoded)
        if reference.is_absolute() or ".." in reference.parts:
            raise ValidationError(f"unsafe HTML reference: {path}: {raw}")
        if not (path.parent / reference).resolve().exists():
            raise ValidationError(f"broken HTML reference: {path}: {raw}")
    return parser


def validate_dataset(root: Path, name: str, samples_expected: int, candidates_expected: int) -> dict:
    dataset_dir = root / "sub-dataset" / name
    required = (
        "sampling-summary.json",
        "source-schema.json",
        "source-layout.txt",
        "format-audit.json",
        "media-audit.json",
        "input-files.json",
        "sampling-manifest.jsonl",
        "sampling-candidate-decisions.jsonl",
        "displayed-samples.jsonl",
    )
    for filename in required:
        if not (dataset_dir / filename).is_file():
            raise ValidationError(f"missing dataset artifact: {dataset_dir / filename}")
    summary = load_json(dataset_dir / "sampling-summary.json")
    if not isinstance(summary, dict):
        raise ValidationError(f"summary is not an object: {name}")
    manifest = load_jsonl(dataset_dir / "sampling-manifest.jsonl")
    decisions = load_jsonl(dataset_dir / "sampling-candidate-decisions.jsonl")
    samples = load_jsonl(dataset_dir / "displayed-samples.jsonl")
    if len(manifest) != samples_expected or len(samples) != samples_expected:
        raise ValidationError(
            f"sample count mismatch: {name}: manifest={len(manifest)} displayed={len(samples)}"
        )
    if summary.get("selected_count") != samples_expected:
        raise ValidationError(f"selected_count mismatch: {name}")
    if summary.get("candidate_count") != candidates_expected:
        raise ValidationError(f"candidate_count mismatch: {name}")
    if len(decisions) != candidates_expected - samples_expected:
        raise ValidationError(f"candidate decisions mismatch: {name}")
    draw_orders = [row.get("draw_order") for row in manifest + decisions]
    if sorted(draw_orders) != list(range(candidates_expected)):
        raise ValidationError(f"candidate partition mismatch: {name}")
    if not summary.get("source_unchanged"):
        raise ValidationError(f"source fingerprint changed: {name}")
    if summary.get("source_fingerprint_before") != summary.get("source_fingerprint_after"):
        raise ValidationError(f"source fingerprint digest mismatch: {name}")

    sample_ids = set()
    archived_assets: set[Path] = set()
    status_counts: dict[str, int] = {}
    gt_overlay_counts: dict[str, int] = {}
    for row in samples:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValidationError(f"duplicate/missing sample id: {name}: {sample_id}")
        sample_ids.add(sample_id)
        safe_path(dataset_dir, str(row.get("record_archive_path") or ""), "record archive")
        status = str(row.get("media_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        assets = row.get("media_assets") or []
        available_assets = 0
        overlay_assets = 0
        for asset in assets:
            if not isinstance(asset, dict) or not asset.get("archive_path"):
                continue
            path = safe_path(dataset_dir, str(asset["archive_path"]), "media asset")
            archived_assets.add(path)
            available_assets += 1
            if asset.get("gt_overlay"):
                overlay_assets += 1
            expected = asset.get("sha256") or asset.get("preview_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValidationError(f"missing asset hash: {name}: {sample_id}: {path}")
            if sha256_file(path) != expected:
                raise ValidationError(f"asset hash mismatch: {path}")
            decode_image(path)
        if status in {"available", "partial"} and available_assets == 0:
            raise ValidationError(f"available sample lacks archived asset: {name}: {sample_id}")
        objects = row.get("objects")
        has_declared_image_gt = (
            isinstance(objects, dict)
            and isinstance(objects.get("bbox"), list)
            and bool(objects.get("bbox"))
            and any(isinstance(asset, dict) and asset.get("type") == "images" for asset in assets)
        )
        overlays = row.get("ground_truth_overlay") or []
        for overlay in overlays:
            if isinstance(overlay, dict):
                overlay_status = str(overlay.get("status") or "unknown")
                gt_overlay_counts[overlay_status] = gt_overlay_counts.get(overlay_status, 0) + 1
        if has_declared_image_gt and overlay_assets == 0:
            raise ValidationError(f"image grounding sample lacks GT overlay asset: {name}: {sample_id}")

    report = root / f"{name}.html"
    parser = parse_html(report)
    if parser.sample_cards != samples_expected:
        raise ValidationError(
            f"HTML sample-card mismatch: {name}: {parser.sample_cards}/{samples_expected}"
        )
    html_assets = {
        (report.parent / Path(unquote(urlsplit(raw).path))).resolve()
        for raw in parser.asset_references
        if not urlsplit(raw).scheme
    }
    if not archived_assets.issubset(html_assets):
        missing = sorted(str(path) for path in archived_assets - html_assets)
        raise ValidationError(f"archived assets absent from HTML: {name}: {missing[:3]}")
    return {
        "dataset": name,
        "samples": len(samples),
        "candidate_decisions": len(decisions),
        "archived_assets": len(archived_assets),
        "sample_media_statuses": status_counts,
        "ground_truth_overlay_statuses": gt_overlay_counts,
    }


def validate(root: Path, expected_datasets: int, samples_expected: int, candidates_expected: int) -> dict:
    root = root.resolve()
    status = load_json(root / "collection-status.json")
    if not isinstance(status, dict) or status.get("state") != "completed":
        raise ValidationError(f"collection is not complete: {status}")
    overall = load_json(root / "overall-summary.json")
    evidence = load_json(root / "source-readonly-evidence.json")
    if not isinstance(overall, dict) or not isinstance(overall.get("datasets"), list):
        raise ValidationError("invalid overall summary")
    datasets = overall["datasets"]
    if len(datasets) != expected_datasets:
        raise ValidationError(f"dataset count mismatch: {len(datasets)}/{expected_datasets}")
    if overall.get("displayed_rows_total") != expected_datasets * samples_expected:
        raise ValidationError("overall displayed row count mismatch")
    names = [str(item.get("name")) for item in datasets]
    if len(names) != len(set(names)):
        raise ValidationError("duplicate dataset names")
    evidence_entries = evidence.get("datasets") if isinstance(evidence, dict) else None
    if not isinstance(evidence_entries, list) or {row.get("dataset") for row in evidence_entries} != set(names):
        raise ValidationError("source evidence dataset set mismatch")
    for entry in evidence_entries:
        if not entry.get("unchanged") or entry.get("before_digest") != entry.get("after_digest"):
            raise ValidationError(f"source evidence mismatch: {entry.get('dataset')}")

    index_parser = parse_html(root / "index.html")
    linked_reports = {
        Path(unquote(urlsplit(raw).path)).name
        for raw in index_parser.references
        if urlsplit(raw).path.endswith(".html")
    }
    expected_reports = {f"{name}.html" for name in names}
    if not expected_reports.issubset(linked_reports):
        raise ValidationError(f"index missing reports: {sorted(expected_reports - linked_reports)}")
    results = [
        validate_dataset(root, name, samples_expected, candidates_expected) for name in names
    ]
    return {
        "status": "passed",
        "root": str(root),
        "datasets": expected_datasets,
        "reports": expected_datasets,
        "displayed_samples": expected_datasets * samples_expected,
        "candidate_rows": expected_datasets * candidates_expected,
        "source_readonly_evidence": "passed",
        "dataset_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-datasets", type=int, default=17)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=400)
    args = parser.parse_args()
    try:
        result = validate(args.root, args.expected_datasets, args.samples, args.candidates)
    except ValidationError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    output = args.root.resolve() / "validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS datasets={result['datasets']} displayed_samples={result['displayed_samples']} "
        f"candidate_rows={result['candidate_rows']} source_readonly=passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
