#!/usr/bin/env python3
"""Validate an analyze-source-datasets delivery without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REQUIRED_DATASET_FILES = (
    "sampling-manifest.jsonl",
    "sampling-candidate-decisions.jsonl",
    "sampling-summary.json",
    "source-layout.txt",
    "source-schema.json",
)
RAW_JSON_ORIGINS = {"source_exact", "derived_lossless_projection"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DeliveryError(Exception):
    pass


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sample_cards: list[list[int]] = []
        self.links: list[str] = []
        self.images: list[str] = []
        self.asset_references: list[str] = []
        self.has_viewport = False
        self.stack: list[dict] = []
        self.raw_blocks: list[dict] = []
        self.active_raw_block: int | None = None
        self.raw_child_tags = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        attr_names = {key for key, _ in attrs}
        classes = values.get("class", "").split()
        parent_card = self.stack[-1]["card"] if self.stack else None
        parent_hidden = self.stack[-1]["hidden"] if self.stack else False
        parent_closed_details = self.stack[-1]["closed_details"] if self.stack else 0
        compact_style = re.sub(r"\s+", "", values.get("style", "").casefold())
        hidden = parent_hidden or "hidden" in attr_names or any(
            rule in compact_style
            for rule in ("display:none", "visibility:hidden", "content-visibility:hidden", "opacity:0")
        )
        closed_details = parent_closed_details
        if tag == "details" and "open" not in attr_names:
            closed_details += 1
        card = parent_card
        if "sample" in classes:
            card = len(self.sample_cards)
            self.sample_cards.append([])
        if self.active_raw_block is not None:
            self.raw_child_tags += 1
        if tag == "pre" and "raw-json" in classes:
            block = {
                "attrs": values,
                "text": "",
                "card": card,
                "visible_by_default": not hidden and closed_details == 0,
                "stack_depth": len(self.stack) + 1,
            }
            self.raw_blocks.append(block)
            self.active_raw_block = len(self.raw_blocks) - 1
            if card is not None:
                self.sample_cards[card].append(self.active_raw_block)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag == "img" and values.get("src"):
            self.images.append(values["src"])
            self.asset_references.append(values["src"])
        if tag == "meta" and values.get("name", "").lower() == "viewport":
            self.has_viewport = True
        if tag == "link" and values.get("href"):
            self.links.append(values["href"])
            self.asset_references.append(values["href"])
        if tag in {"script", "source", "video"} and values.get("src"):
            self.links.append(values["src"])
            self.asset_references.append(values["src"])
        self.stack.append(
            {
                "tag": tag,
                "card": card,
                "hidden": hidden,
                "closed_details": closed_details,
            }
        )

    def handle_endtag(self, tag: str) -> None:
        if (
            tag == "pre"
            and self.active_raw_block is not None
            and self.raw_blocks[self.active_raw_block]["stack_depth"] == len(self.stack)
        ):
            self.active_raw_block = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self.active_raw_block is not None:
            self.raw_blocks[self.active_raw_block]["text"] += data


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DeliveryError(f"invalid JSON: {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except Exception as exc:
        raise DeliveryError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_archive_path(dataset_dir: Path, raw_path: str, label: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DeliveryError(f"unsafe {label}: {dataset_dir.name}: {raw_path}")
    resolved = (dataset_dir / relative).resolve()
    try:
        resolved.relative_to(dataset_dir.resolve())
    except ValueError as exc:
        raise DeliveryError(f"escaping {label}: {dataset_dir.name}: {raw_path}") from exc
    if not resolved.is_file():
        raise DeliveryError(f"missing {label}: {dataset_dir.name}: {raw_path}")
    return resolved


def raw_json_payload(dataset_dir: Path, row: dict) -> tuple[dict, str]:
    sample_id = str(row.get("sample_id", ""))
    descriptor = row.get("raw_json")
    if not isinstance(descriptor, dict):
        raise DeliveryError(f"missing or invalid raw_json descriptor: {dataset_dir.name}: {sample_id}")
    missing = {
        field
        for field in ("archive_path", "sha256", "origin", "encoding")
        if field not in descriptor
    }
    if missing:
        raise DeliveryError(
            f"raw_json descriptor lacks fields {sorted(missing)}: {dataset_dir.name}: {sample_id}"
        )
    archive_path = descriptor.get("archive_path")
    expected_hash = descriptor.get("sha256")
    origin = descriptor.get("origin")
    encoding = descriptor.get("encoding")
    if not isinstance(archive_path, str) or not archive_path:
        raise DeliveryError(f"invalid raw_json archive path: {dataset_dir.name}: {sample_id}")
    parts = urlsplit(archive_path)
    if parts.scheme or parts.netloc or parts.query or parts.fragment:
        raise DeliveryError(f"unsafe raw_json archive path: {dataset_dir.name}: {archive_path}")
    if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
        raise DeliveryError(f"invalid raw_json SHA256: {dataset_dir.name}: {sample_id}")
    if origin not in RAW_JSON_ORIGINS:
        raise DeliveryError(f"invalid raw_json origin: {dataset_dir.name}: {sample_id}: {origin}")
    if encoding != "utf-8":
        raise DeliveryError(f"invalid raw_json encoding: {dataset_dir.name}: {sample_id}: {encoding}")
    archive = safe_archive_path(dataset_dir, archive_path, "raw_json archive path")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise DeliveryError(f"raw_json SHA256 mismatch: {dataset_dir.name}: {sample_id}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeliveryError(f"raw_json is not UTF-8: {dataset_dir.name}: {sample_id}: {exc}") from exc
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeliveryError(
            f"raw_json is not one complete JSON value: {dataset_dir.name}: {sample_id}: {exc}"
        ) from exc
    return descriptor, text


def validate_raw_json_blocks(
    parser: ReportParser,
    expected_rows: list[tuple[str, dict, str]],
    report: Path,
) -> dict[tuple[str, str], str]:
    expected_ids = [sample_id for sample_id, _, _ in expected_rows]
    if parser.raw_child_tags:
        raise DeliveryError(f"raw_json block contains executable/markup child tags: {report}")
    if len(parser.raw_blocks) != len(expected_rows):
        raise DeliveryError(
            f"raw_json block count {len(parser.raw_blocks)} != {len(expected_rows)}: {report}"
        )
    if len(parser.sample_cards) != len(expected_rows):
        raise DeliveryError(
            f"sample card count {len(parser.sample_cards)} != {len(expected_rows)}: {report}"
        )

    observed_ids: list[str] = []
    observed: dict[tuple[str, str], dict] = {}
    for card_index, block_indexes in enumerate(parser.sample_cards):
        if len(block_indexes) != 1:
            raise DeliveryError(
                f"sample card must contain exactly one raw_json block: {report}: card={card_index}"
            )
        block = parser.raw_blocks[block_indexes[0]]
        attrs = block["attrs"]
        sample_id = attrs.get("data-sample-id", "")
        role = attrs.get("data-raw-json-role", "")
        if role != "sample":
            raise DeliveryError(f"invalid raw_json HTML role: {report}: {sample_id}: {role}")
        if not block["visible_by_default"]:
            raise DeliveryError(f"raw_json is not visible by default: {report}: {sample_id}")
        key = (sample_id, role)
        if key in observed:
            raise DeliveryError(f"duplicate raw_json HTML block: {report}: {sample_id}: {role}")
        observed[key] = block
        observed_ids.append(sample_id)

    if observed_ids != expected_ids:
        raise DeliveryError(f"raw_json sample card order/identity mismatch: {report}")
    if any(block["card"] is None for block in parser.raw_blocks):
        raise DeliveryError(f"raw_json block appears outside a sample card: {report}")

    browser_expected: dict[tuple[str, str], str] = {}
    for sample_id, descriptor, text in expected_rows:
        key = (sample_id, "sample")
        block = observed.get(key)
        if block is None:
            raise DeliveryError(f"missing raw_json HTML block: {report}: {sample_id}")
        attrs = block["attrs"]
        for attribute, field in (
            ("data-archive-path", "archive_path"),
            ("data-sha256", "sha256"),
            ("data-origin", "origin"),
        ):
            if attrs.get(attribute) != descriptor.get(field):
                raise DeliveryError(
                    f"raw_json HTML attribute mismatch: {report}: {sample_id}: {attribute}"
                )
        if block["text"] != text:
            raise DeliveryError(f"raw_json HTML text differs from archive: {report}: {sample_id}")
        browser_expected[key] = text
    return browser_expected


def media_path_and_hash(dataset_dir: Path, row: dict) -> tuple[Path, str]:
    if row.get("media_archive_path"):
        path = safe_archive_path(dataset_dir, str(row["media_archive_path"]), "media archive path")
        expected = row.get("sha256") or row.get("source_video_sha256")
    elif row.get("preview_archive_path"):
        path = safe_archive_path(dataset_dir, str(row["preview_archive_path"]), "preview archive path")
        expected = row.get("preview_sha256")
    else:
        raise DeliveryError(f"manifest row has no media or preview path: {dataset_dir.name}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise DeliveryError(f"manifest row has no valid archived-media SHA256: {dataset_dir.name}")
    return path, expected


def uniqueness_key(row: dict) -> tuple[str, str]:
    if row.get("source_video_sha256"):
        video_id = str(row.get("video_id", ""))
        if not video_id:
            raise DeliveryError("video row lacks video_id")
        return str(row["source_video_sha256"]), video_id
    if not row.get("sha256"):
        raise DeliveryError("image row lacks sha256")
    return str(row["sha256"]), ""


def verify_decodable(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise DeliveryError("Pillow is required to verify archived image decoding") from exc
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise DeliveryError(f"media decode failed: {path}: {exc}") from exc


def local_reference(html_path: Path, raw_value: str) -> Path | None:
    parts = urlsplit(raw_value)
    if parts.scheme or parts.netloc or raw_value.startswith("#"):
        return None
    decoded = unquote(parts.path)
    if not decoded:
        return None
    reference = Path(decoded)
    if reference.is_absolute() or ".." in reference.parts:
        raise DeliveryError(f"unsafe HTML reference: {html_path}: {raw_value}")
    return (html_path.parent / reference).resolve()


def parse_report(path: Path) -> ReportParser:
    parser = ReportParser()
    try:
        parser.feed(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DeliveryError(f"invalid HTML input: {path}: {exc}") from exc
    if not parser.has_viewport:
        raise DeliveryError(f"missing viewport meta tag: {path}")
    for raw_value in parser.asset_references:
        parts = urlsplit(raw_value)
        if parts.scheme or parts.netloc:
            raise DeliveryError(f"external asset dependency is not allowed: {path}: {raw_value}")
    for raw_value in parser.links + parser.images:
        reference = local_reference(path, raw_value)
        if reference is not None and not reference.exists():
            raise DeliveryError(f"broken internal reference: {path}: {raw_value}")
    return parser


def validate_dataset(
    dataset_dir: Path,
    report: Path,
    expected_count: int,
) -> tuple[int, int, dict[tuple[str, str], str]]:
    for name in REQUIRED_DATASET_FILES:
        if not (dataset_dir / name).is_file():
            raise DeliveryError(f"missing required file: {dataset_dir / name}")

    summary = load_json(dataset_dir / "sampling-summary.json")
    schema = load_json(dataset_dir / "source-schema.json")
    if not isinstance(summary, dict) or not isinstance(schema, (dict, list)):
        raise DeliveryError(f"summary/schema has wrong JSON type: {dataset_dir.name}")
    manifest = load_jsonl(dataset_dir / "sampling-manifest.jsonl")
    decisions = load_jsonl(dataset_dir / "sampling-candidate-decisions.jsonl")

    selected_count = summary.get("selected_count")
    candidate_count = summary.get("candidate_count")
    if selected_count != len(manifest):
        raise DeliveryError(f"selected_count differs from manifest rows: {dataset_dir.name}")
    if selected_count != expected_count:
        raise DeliveryError(
            f"expected {expected_count} accepted samples, found {selected_count}: {dataset_dir.name}"
        )
    if not isinstance(candidate_count, int) or candidate_count < selected_count:
        raise DeliveryError(f"invalid candidate_count: {dataset_dir.name}")
    population_total = summary.get("population_total", summary.get("video_population_total"))
    if not isinstance(population_total, int) or population_total < selected_count:
        raise DeliveryError(f"invalid population_total: {dataset_dir.name}")
    if not isinstance(summary.get("seed"), int):
        raise DeliveryError(f"missing integer seed: {dataset_dir.name}")

    sample_ids: set[str] = set()
    uniqueness: set[tuple[str, str]] = set()
    draw_orders: list[int] = []
    archived_media: set[Path] = set()
    expected_raw_json: list[tuple[str, dict, str]] = []
    for row in manifest:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise DeliveryError(f"missing or duplicate sample_id: {dataset_dir.name}: {sample_id}")
        sample_ids.add(sample_id)
        raw_descriptor, raw_text = raw_json_payload(dataset_dir, row)
        expected_raw_json.append((sample_id, raw_descriptor, raw_text))
        draw_order = row.get("draw_order")
        if not isinstance(draw_order, int):
            raise DeliveryError(f"missing integer draw_order: {dataset_dir.name}: {sample_id}")
        draw_orders.append(draw_order)

        key = uniqueness_key(row)
        if key in uniqueness:
            raise DeliveryError(f"duplicate accepted visual identity: {dataset_dir.name}: {sample_id}")
        uniqueness.add(key)

        media_path, expected_hash = media_path_and_hash(dataset_dir, row)
        archived_media.add(media_path)
        actual_hash = sha256_file(media_path)
        if actual_hash != expected_hash:
            raise DeliveryError(f"archived media SHA256 mismatch: {media_path}")
        verify_decodable(media_path)

        for field, raw_value in row.items():
            if field.endswith("_archive_path") and raw_value:
                safe_archive_path(dataset_dir, str(raw_value), field)

    for row in decisions:
        draw_order = row.get("draw_order")
        if not isinstance(draw_order, int) or not row.get("reason"):
            raise DeliveryError(f"candidate decision lacks draw_order/reason: {dataset_dir.name}")
        draw_orders.append(draw_order)

    if len(draw_orders) != candidate_count:
        raise DeliveryError(
            f"candidate partition size {len(draw_orders)} != {candidate_count}: {dataset_dir.name}"
        )
    if set(draw_orders) != set(range(candidate_count)) or len(set(draw_orders)) != candidate_count:
        raise DeliveryError(f"candidate draw orders do not form an exact partition: {dataset_dir.name}")

    parser = parse_report(report)
    if len(parser.sample_cards) != selected_count:
        raise DeliveryError(
            f"report card count {len(parser.sample_cards)} != {selected_count}: {report.name}"
        )
    browser_expected = validate_raw_json_blocks(parser, expected_raw_json, report)
    report_images = {
        reference
        for raw_value in parser.images
        if (reference := local_reference(report, raw_value)) is not None
    }
    report_media_links = {
        reference
        for raw_value in parser.links
        if (reference := local_reference(report, raw_value)) is not None
    }
    if not archived_media.issubset(report_images):
        missing = sorted(str(path) for path in archived_media - report_images)
        raise DeliveryError(f"manifest media missing from report images: {report.name}: {missing[:3]}")
    if not archived_media.issubset(report_media_links):
        missing = sorted(str(path) for path in archived_media - report_media_links)
        raise DeliveryError(f"manifest media missing from report links: {report.name}: {missing[:3]}")
    return selected_count, candidate_count, browser_expected


def validate_readonly_evidence(root: Path, dataset_names: set[str], required: bool) -> None:
    path = root / "source-readonly-evidence.json"
    if not path.is_file():
        if required:
            raise DeliveryError(f"missing source read-only evidence: {path}")
        print("WARN source-readonly-evidence.json absent; legacy read-only declarations are not proof")
        return
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
        raise DeliveryError(f"invalid source read-only evidence structure: {path}")
    entries = {entry.get("dataset"): entry for entry in payload["datasets"] if isinstance(entry, dict)}
    if set(entries) != dataset_names:
        raise DeliveryError("source read-only evidence dataset set does not match delivery")
    for name, entry in entries.items():
        before = entry.get("before_digest")
        after = entry.get("after_digest")
        if not before or before != after or entry.get("unchanged") is not True:
            raise DeliveryError(f"source fingerprint changed or is incomplete: {name}")
        if not entry.get("fingerprint_algorithm") or not entry.get("source_root"):
            raise DeliveryError(f"source evidence lacks algorithm/root: {name}")


def browser_check(
    html_files: list[Path],
    raw_expectations: dict[Path, dict[tuple[str, str], str]],
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise DeliveryError("Playwright is required for --browser-check") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for viewport in ({"width": 1440, "height": 900}, {"width": 390, "height": 844}):
                for html_path in html_files:
                    page = browser.new_page(viewport=viewport)
                    console_errors: list[str] = []
                    page_errors: list[str] = []
                    external_requests: list[str] = []
                    page.on(
                        "console",
                        lambda message: console_errors.append(message.text)
                        if message.type == "error"
                        else None,
                    )
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    page.on(
                        "request",
                        lambda request: external_requests.append(request.url)
                        if urlsplit(request.url).scheme in {"http", "https"}
                        else None,
                    )
                    try:
                        page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=60_000)
                        result = page.evaluate(
                            """async () => {
                              const images = Array.from(document.images);
                              images.forEach(img => { img.loading = 'eager'; });
                              await Promise.all(images.map(img => {
                                if (img.complete) return Promise.resolve();
                                return new Promise(resolve => {
                                  img.addEventListener('load', resolve, {once: true});
                                  img.addEventListener('error', resolve, {once: true});
                                  setTimeout(resolve, 10000);
                                });
                              }));
                              const rawBlocks = Array.from(document.querySelectorAll('pre.raw-json')).map(el => {
                                const style = getComputedStyle(el);
                                return {
                                  sampleId: el.dataset.sampleId || '',
                                  role: el.dataset.rawJsonRole || '',
                                  text: el.textContent,
                                  visible: el.getClientRects().length > 0 &&
                                    style.display !== 'none' && style.visibility !== 'hidden' &&
                                    style.opacity !== '0' && !el.closest('details:not([open])')
                                };
                              });
                              return {
                                overflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth,
                                brokenImages: images.filter(img => !img.complete || img.naturalWidth === 0).map(img => img.getAttribute('src')),
                                rawBlocks
                              };
                            }"""
                        )
                        if result["overflow"] > 1:
                            raise DeliveryError(
                                f"horizontal overflow at {viewport['width']}px: {html_path}: {result['overflow']}px"
                            )
                        if result["brokenImages"]:
                            raise DeliveryError(
                                f"browser found broken images: {html_path}: {result['brokenImages'][:3]}"
                            )
                        expected = raw_expectations.get(html_path.resolve())
                        if expected is not None:
                            observed: dict[tuple[str, str], dict] = {}
                            for block in result["rawBlocks"]:
                                key = (block["sampleId"], block["role"])
                                if key in observed:
                                    raise DeliveryError(
                                        f"browser found duplicate raw_json block: {html_path}: {key}"
                                    )
                                observed[key] = block
                            if set(observed) != set(expected):
                                raise DeliveryError(
                                    f"browser raw_json identities differ from manifest: {html_path}"
                                )
                            for key, expected_text in expected.items():
                                block = observed[key]
                                if not block["visible"]:
                                    raise DeliveryError(
                                        f"browser raw_json block is not visible: {html_path}: {key}"
                                    )
                                if block["text"] != expected_text:
                                    raise DeliveryError(
                                        f"browser raw_json text differs from archive: {html_path}: {key}"
                                    )
                        if external_requests:
                            raise DeliveryError(
                                f"browser made external requests: {html_path}: {external_requests[:3]}"
                            )
                        if console_errors or page_errors:
                            raise DeliveryError(
                                f"browser errors: {html_path}: console={console_errors[:3]} page={page_errors[:3]}"
                            )
                    finally:
                        page.close()
        finally:
            browser.close()


def validate(root: Path, expected_count: int, require_source_evidence: bool, run_browser: bool) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise DeliveryError(f"delivery root is not a directory: {root}")
    index = root / "index.html"
    sub_root = root / "sub-dataset"
    if not index.is_file() or not sub_root.is_dir():
        raise DeliveryError("delivery must contain index.html and sub-dataset/")

    dataset_dirs = sorted(path for path in sub_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not dataset_dirs:
        raise DeliveryError("no dataset archive directories found")

    index_parser = parse_report(index)
    report_paths = {path.name for path in root.glob("*.html") if path.name != "index.html"}
    index_report_links = {
        Path(urlsplit(unquote(link)).path).name
        for link in index_parser.links
        if urlsplit(link).path.lower().endswith(".html") and Path(urlsplit(unquote(link)).path).name != "index.html"
    }

    total_selected = 0
    total_candidates = 0
    reports = []
    raw_expectations: dict[Path, dict[tuple[str, str], str]] = {}
    for dataset_dir in dataset_dirs:
        report = root / f"{dataset_dir.name}.html"
        if not report.is_file():
            raise DeliveryError(f"missing dataset report: {report}")
        selected, candidates, report_raw_expectations = validate_dataset(
            dataset_dir, report, expected_count
        )
        total_selected += selected
        total_candidates += candidates
        reports.append(report)
        raw_expectations[report.resolve()] = report_raw_expectations

    expected_reports = {f"{dataset_dir.name}.html" for dataset_dir in dataset_dirs}
    if report_paths != expected_reports:
        raise DeliveryError(
            f"report files do not exactly match dataset directories: reports={sorted(report_paths)} datasets={sorted(expected_reports)}"
        )
    if not expected_reports.issubset(index_report_links):
        missing = sorted(expected_reports - index_report_links)
        raise DeliveryError(f"index does not link every dataset report: {missing}")

    dataset_names = {path.name for path in dataset_dirs}
    validate_readonly_evidence(root, dataset_names, require_source_evidence)
    if run_browser:
        browser_check([index, *reports], raw_expectations)

    print(
        f"PASS datasets={len(dataset_dirs)} reports={len(reports)} "
        f"selected={total_selected} candidates={total_candidates} "
        f"browser={'passed' if run_browser else 'not-run'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery_root", type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--require-source-evidence", action="store_true")
    parser.add_argument("--browser-check", action="store_true")
    args = parser.parse_args()
    try:
        validate(
            args.delivery_root,
            expected_count=args.expected_count,
            require_source_evidence=args.require_source_evidence,
            run_browser=args.browser_check,
        )
    except DeliveryError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
