#!/usr/bin/env python3
"""Validate an offline paired VLM benchmark-analysis delivery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


RAW_JSON_ROLES = ("sample", "baseline_prediction", "ours_prediction")
RAW_JSON_ORIGINS = {"source_exact", "derived_lossless_projection"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HtmlAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang = ""
        self.viewport = False
        self.samples = 0
        self.images = 0
        self.videos = 0
        self.references: list[str] = []
        self.pre_depth = 0
        self.pre_child_tags = 0
        self.stack: list[dict] = []
        self.sample_cards: list[list[int]] = []
        self.raw_blocks: list[dict] = []
        self.active_raw_block: int | None = None

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

        if self.pre_depth and tag != "pre":
            self.pre_child_tags += 1
        if tag == "pre":
            self.pre_depth += 1
            if "raw-json" in classes:
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
        if tag == "html":
            self.lang = values.get("lang", "")
        if tag == "meta" and values.get("name", "").casefold() == "viewport":
            self.viewport = True
        if "sample" in values.get("class", "").split():
            self.samples += 1
        if tag == "img":
            self.images += 1
        if tag == "video":
            self.videos += 1
        for field in ("src", "href", "poster"):
            if values.get(field):
                self.references.append(values[field])
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
        if tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self.active_raw_block is not None:
            self.raw_blocks[self.active_raw_block]["text"] += data


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{number}")
            rows.append(value)
    return rows


def raw_archive_path(report_dir: Path, raw: str) -> Path:
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"unsafe_raw_json_path:{report_dir}:{raw}")
    relative = Path(unquote(parsed.path))
    if not parsed.path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe_raw_json_path:{report_dir}:{raw}")
    resolved = (report_dir / relative).resolve()
    try:
        resolved.relative_to(report_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"escaping_raw_json_path:{report_dir}:{raw}") from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"missing_or_empty_raw_json:{report_dir}:{raw}")
    return resolved


def validate_raw_json(
    report_dir: Path,
    paired: list[dict],
    parser: HtmlAudit,
    errors: list[str],
) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], dict] = {}
    expected_text: dict[tuple[str, str], str] = {}

    for row in paired:
        sample_id = str(row.get("sample_id"))
        descriptors = row.get("raw_json")
        if not isinstance(descriptors, dict):
            errors.append(f"missing_or_invalid_raw_json:{sample_id}")
            continue
        actual_roles = set(descriptors)
        required_roles = set(RAW_JSON_ROLES)
        if actual_roles != required_roles:
            errors.append(
                f"raw_json_roles:{sample_id}:missing={sorted(required_roles - actual_roles)}:"
                f"extra={sorted(actual_roles - required_roles)}"
            )
        for role in RAW_JSON_ROLES:
            descriptor = descriptors.get(role)
            key = (sample_id, role)
            if not isinstance(descriptor, dict):
                errors.append(f"invalid_raw_json_descriptor:{sample_id}:{role}")
                continue
            missing = {
                field
                for field in ("archive_path", "sha256", "origin", "encoding")
                if field not in descriptor
            }
            if missing:
                errors.append(f"missing_raw_json_fields:{sample_id}:{role}:{sorted(missing)}")
                continue
            archive_path = descriptor.get("archive_path")
            expected_hash = descriptor.get("sha256")
            origin = descriptor.get("origin")
            encoding = descriptor.get("encoding")
            if not isinstance(archive_path, str) or not archive_path:
                errors.append(f"invalid_raw_json_archive_path:{sample_id}:{role}")
                continue
            if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
                errors.append(f"invalid_raw_json_sha256:{sample_id}:{role}")
                continue
            if origin not in RAW_JSON_ORIGINS:
                errors.append(f"invalid_raw_json_origin:{sample_id}:{role}:{origin}")
            if encoding != "utf-8":
                errors.append(f"invalid_raw_json_encoding:{sample_id}:{role}:{encoding}")
            try:
                archive = raw_archive_path(report_dir, archive_path)
                payload = archive.read_bytes()
                actual_hash = hashlib.sha256(payload).hexdigest()
                if actual_hash != expected_hash:
                    errors.append(f"raw_json_sha256_mismatch:{sample_id}:{role}")
                text = payload.decode("utf-8")
                json.loads(text)
            except UnicodeDecodeError as exc:
                errors.append(f"raw_json_not_utf8:{sample_id}:{role}:{exc}")
                continue
            except json.JSONDecodeError as exc:
                errors.append(f"raw_json_parse_error:{sample_id}:{role}:{exc}")
                continue
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                continue
            if key in expected:
                errors.append(f"duplicate_raw_json_expectation:{sample_id}:{role}")
                continue
            expected[key] = descriptor
            expected_text[key] = text

    if len(parser.raw_blocks) != len(paired) * len(RAW_JSON_ROLES):
        errors.append(
            f"html_raw_json_count:{len(parser.raw_blocks)}!={len(paired) * len(RAW_JSON_ROLES)}"
        )
    if len(parser.sample_cards) != len(paired):
        errors.append(f"html_raw_json_card_count:{len(parser.sample_cards)}!={len(paired)}")

    html_blocks: dict[tuple[str, str], dict] = {}
    card_ids: list[str] = []
    for card_index, block_indexes in enumerate(parser.sample_cards):
        blocks = [parser.raw_blocks[index] for index in block_indexes]
        ids = {block["attrs"].get("data-sample-id", "") for block in blocks}
        roles = [block["attrs"].get("data-raw-json-role", "") for block in blocks]
        if len(blocks) != len(RAW_JSON_ROLES) or set(roles) != set(RAW_JSON_ROLES):
            errors.append(f"html_raw_json_card_roles:{card_index}:{roles}")
        if len(ids) != 1 or not next(iter(ids), ""):
            errors.append(f"html_raw_json_card_sample_ids:{card_index}:{sorted(ids)}")
            card_ids.append("")
        else:
            card_ids.append(next(iter(ids)))

    paired_ids = [str(row.get("sample_id")) for row in paired]
    if card_ids != paired_ids:
        errors.append("html_raw_json_card_order_or_identity")

    for block in parser.raw_blocks:
        attrs = block["attrs"]
        sample_id = attrs.get("data-sample-id", "")
        role = attrs.get("data-raw-json-role", "")
        key = (sample_id, role)
        if block["card"] is None:
            errors.append(f"raw_json_outside_sample_card:{sample_id}:{role}")
        if not block["visible_by_default"]:
            errors.append(f"raw_json_not_visible_by_default:{sample_id}:{role}")
        if key in html_blocks:
            errors.append(f"duplicate_html_raw_json:{sample_id}:{role}")
            continue
        html_blocks[key] = block

    for key, descriptor in expected.items():
        sample_id, role = key
        block = html_blocks.get(key)
        if block is None:
            errors.append(f"missing_html_raw_json:{sample_id}:{role}")
            continue
        attrs = block["attrs"]
        for attribute, field in (
            ("data-archive-path", "archive_path"),
            ("data-sha256", "sha256"),
            ("data-origin", "origin"),
        ):
            if attrs.get(attribute) != descriptor.get(field):
                errors.append(f"html_raw_json_attribute_mismatch:{sample_id}:{role}:{attribute}")
        if block["text"] != expected_text[key]:
            errors.append(f"html_raw_json_text_mismatch:{sample_id}:{role}")
    for key in set(html_blocks) - set(expected):
        errors.append(f"unexpected_html_raw_json:{key[0]}:{key[1]}")
    return expected_text


def local_reference(html: Path, raw: str, root: Path) -> Path | None:
    parsed = urlsplit(raw)
    if raw.startswith("#"):
        return None
    if parsed.scheme or parsed.netloc:
        raise ValueError(f"external_or_unsafe_reference:{html}:{raw}")
    relative = Path(unquote(parsed.path))
    if relative.is_absolute():
        raise ValueError(f"absolute_reference:{html}:{raw}")
    resolved = (html.parent / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"escaping_reference:{html}:{raw}") from exc
    return resolved


def registry_count(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("benchmarks", "items", "registry"):
            if isinstance(value.get(key), list):
                return len(value[key])
    raise ValueError(f"cannot identify benchmark list in {path}")


def validate_report(report_dir: Path, root: Path) -> dict:
    errors = []
    required = (
        "report.html", "summary.json", "inference_config.json", "paired_samples.jsonl",
        "sample_manifest.jsonl", "badcases.csv", "validation.json",
    )
    for name in required:
        path = report_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{name}")
    if errors:
        return {"report_dir": str(report_dir), "errors": errors}

    summary = read_json(report_dir / "summary.json")
    paired = read_jsonl(report_dir / "paired_samples.jsonl")
    manifest = read_jsonl(report_dir / "sample_manifest.jsonl")
    expected = int(summary.get("displayed_sample_count", len(paired)))
    benchmark = str(summary.get("benchmark", ""))
    is_video_benchmark = benchmark in {"egoplan", "openeqa", "video-mme", "robovqa"}
    paired_ids = [str(row.get("sample_id")) for row in paired]
    manifest_ids = [str(row.get("sample_id")) for row in manifest]
    if expected <= 0 or expected > 100:
        errors.append(f"invalid_displayed_count:{expected}")
    if len(paired) != expected or len(set(paired_ids)) != expected:
        errors.append("paired_count_or_uniqueness")
    if manifest_ids != paired_ids or len(set(manifest_ids)) != expected:
        errors.append("manifest_count_order_or_uniqueness")

    for row in paired:
        sample_id = row.get("sample_id")
        for prefix in ("baseline", "ours"):
            for field in ("response", "parsed", "token_count", "error_tags"):
                if f"{prefix}_{field}" not in row:
                    errors.append(f"missing_field:{sample_id}:{prefix}_{field}")
            if f"{prefix}_correct" not in row and f"{prefix}_score" not in row:
                errors.append(f"missing_correct_or_score:{sample_id}:{prefix}")
        raw_asset = str(row.get("media_asset", ""))
        try:
            asset = local_reference(report_dir / "report.html", raw_asset, root)
            if asset is None or not asset.is_file() or asset.stat().st_size == 0:
                errors.append(f"missing_media_asset:{raw_asset}")
        except ValueError as exc:
            errors.append(str(exc))
        if is_video_benchmark:
            raw_video = str(row.get("video_asset", ""))
            unavailable = str(row.get("media_status", "")) in {"missing", "media_load_failure"}
            if not raw_video:
                if not unavailable:
                    errors.append(f"missing_video_asset:{sample_id}")
            else:
                try:
                    video = local_reference(report_dir / "report.html", raw_video, root)
                    if video is None or not video.is_file() or video.stat().st_size == 0:
                        errors.append(f"missing_video_asset:{raw_video}")
                    elif row.get("video_size_bytes") and int(row["video_size_bytes"]) != video.stat().st_size:
                        errors.append(f"video_size_mismatch:{raw_video}")
                except ValueError as exc:
                    errors.append(str(exc))

    with (report_dir / "badcases.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        list(csv.DictReader(stream))
    report = report_dir / "report.html"
    parser = HtmlAudit()
    parser.feed(report.read_text(encoding="utf-8"))
    if parser.lang != "zh-CN":
        errors.append("html_lang")
    if not parser.viewport:
        errors.append("missing_viewport")
    if parser.samples != expected:
        errors.append(f"html_sample_count:{parser.samples}")
    if parser.images < expected:
        errors.append(f"html_image_count:{parser.images}")
    expected_videos = sum(bool(row.get("video_asset")) for row in paired)
    if is_video_benchmark and parser.videos != expected_videos:
        errors.append(f"html_video_count:{parser.videos}!={expected_videos}")
    if parser.pre_child_tags:
        errors.append(f"raw_output_contains_markup:{parser.pre_child_tags}")
    validate_raw_json(report_dir, paired, parser, errors)
    return {
        "report_dir": str(report_dir),
        "benchmark": summary.get("benchmark"),
        "status": summary.get("status", "complete"),
        "displayed_samples": expected,
        "image_references": parser.images,
        "video_references": parser.videos,
        "raw_json_blocks": len(parser.raw_blocks),
        "errors": errors,
    }


def validate_all_html(root: Path) -> list[str]:
    errors = []
    for html in sorted(root.rglob("*.html")):
        parser = HtmlAudit()
        try:
            parser.feed(html.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError) as exc:
            errors.append(f"html_read_error:{html}:{exc}")
            continue
        for raw in parser.references:
            try:
                target = local_reference(html, raw, root)
                if target is not None and not target.exists():
                    errors.append(f"broken_reference:{html.relative_to(root)}:{raw}")
            except ValueError as exc:
                errors.append(str(exc))
    return errors


def browser_check(report_dirs: list[Path]) -> list[str]:
    errors: list[str] = []
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return [f"browser_check_unavailable:{exc}"]

    expectations: dict[Path, dict[tuple[str, str], str]] = {}
    for report_dir in report_dirs:
        try:
            paired = read_jsonl(report_dir / "paired_samples.jsonl")
            values: dict[tuple[str, str], str] = {}
            for row in paired:
                sample_id = str(row.get("sample_id"))
                descriptors = row.get("raw_json", {})
                for role in RAW_JSON_ROLES:
                    descriptor = descriptors[role]
                    archive = raw_archive_path(report_dir, str(descriptor["archive_path"]))
                    values[(sample_id, role)] = archive.read_bytes().decode("utf-8")
            expectations[(report_dir / "report.html").resolve()] = values
        except (KeyError, OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"browser_expectation_error:{report_dir}:{exc}")

    if errors:
        return errors

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for viewport in ({"width": 1440, "height": 900}, {"width": 390, "height": 844}):
                for html_path, expected in expectations.items():
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
                        page.goto(html_path.as_uri(), wait_until="load", timeout=60_000)
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
                    except Exception as exc:
                        errors.append(f"browser_page_error:{viewport['width']}:{html_path}:{exc}")
                        page.close()
                        continue
                    if result["overflow"] > 1:
                        errors.append(
                            f"browser_horizontal_overflow:{viewport['width']}:{html_path}:{result['overflow']}"
                        )
                    if result["brokenImages"]:
                        errors.append(
                            f"browser_broken_images:{viewport['width']}:{html_path}:{result['brokenImages'][:3]}"
                        )
                    observed: dict[tuple[str, str], dict] = {}
                    for block in result["rawBlocks"]:
                        key = (block["sampleId"], block["role"])
                        if key in observed:
                            errors.append(
                                f"browser_duplicate_raw_json:{viewport['width']}:{key[0]}:{key[1]}"
                            )
                        observed[key] = block
                    if set(observed) != set(expected):
                        errors.append(f"browser_raw_json_identity_mismatch:{viewport['width']}:{html_path}")
                    for key, expected_text in expected.items():
                        block = observed.get(key)
                        if block is None:
                            continue
                        if not block["visible"]:
                            errors.append(
                                f"browser_raw_json_not_visible:{viewport['width']}:{key[0]}:{key[1]}"
                            )
                        if block["text"] != expected_text:
                            errors.append(
                                f"browser_raw_json_text_mismatch:{viewport['width']}:{key[0]}:{key[1]}"
                            )
                    if external_requests:
                        errors.append(
                            f"browser_external_requests:{viewport['width']}:{html_path}:{external_requests[:3]}"
                        )
                    if console_errors or page_errors:
                        errors.append(
                            f"browser_errors:{viewport['width']}:{html_path}:"
                            f"console={console_errors[:3]}:page={page_errors[:3]}"
                        )
                    page.close()
        finally:
            browser.close()
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery_root", type=Path)
    parser.add_argument("--expected-benchmarks", type=int)
    parser.add_argument("--expected-reports", type=int)
    parser.add_argument("--browser-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.delivery_root.resolve()
    output = (args.output or root / "audit" / "delivery-validation.json").resolve()
    root_errors = []
    for name in (
        "index.html", "self-val/index.html", "external-benchmarks/index.html",
        "benchmark-registry.json", "overall-summary.json", "source-readonly-evidence.json",
    ):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            root_errors.append(f"missing_or_empty:{name}")
    benchmark_count = None
    if (root / "benchmark-registry.json").is_file():
        try:
            benchmark_count = registry_count(root / "benchmark-registry.json")
        except (ValueError, json.JSONDecodeError) as exc:
            root_errors.append(f"invalid_registry:{exc}")
    if args.expected_benchmarks is not None and benchmark_count != args.expected_benchmarks:
        root_errors.append(f"benchmark_count:{benchmark_count}!={args.expected_benchmarks}")
    evidence_path = root / "source-readonly-evidence.json"
    if evidence_path.is_file():
        evidence = read_json(evidence_path)
        if evidence.get("source_results_unchanged") is not True:
            root_errors.append("source_results_not_proven_unchanged")
    report_dirs = sorted(path.parent for path in root.glob("*/*/report.html"))
    reports = [validate_report(path, root) for path in report_dirs]
    if args.expected_reports is not None and len(reports) != args.expected_reports:
        root_errors.append(f"report_count:{len(reports)}!={args.expected_reports}")
    html_errors = validate_all_html(root)
    browser_errors = browser_check(report_dirs) if args.browser_check else []
    error_count = (
        len(root_errors)
        + len(html_errors)
        + len(browser_errors)
        + sum(len(report["errors"]) for report in reports)
    )
    result = {
        "status": "pass" if error_count == 0 else "fail",
        "delivery_root": str(root),
        "benchmark_count": benchmark_count,
        "report_count": len(reports),
        "status_page_count": len(list(root.glob("*/*/status.html"))),
        "root_errors": root_errors,
        "html_errors": html_errors,
        "browser_check": "passed" if args.browser_check and not browser_errors else "not-run" if not args.browser_check else "failed",
        "browser_errors": browser_errors,
        "reports": reports,
        "error_count": error_count,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "benchmark_count", "report_count", "status_page_count", "error_count")}, ensure_ascii=False))
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
