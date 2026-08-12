#!/usr/bin/env python3
"""Validate the source-only report delivery without optional image libraries."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = "/mnt/luojunkun/stage1/dataset"
DENIED_ROOT = SOURCE_ROOT + "_ms-swift"
DATASETS = [
    "COCO", "VQAv2", "VisualGenome", "GQA", "TextVQA", "ChartQA", "AI2D",
    "LLaVA-Instruct", "VLM-R1", "Robo2VLM", "RoboVQA", "SpatialVLM",
    "PixMo-Cap", "PixMo-Points", "Molmo2-VideoCapQA", "Molmo2-VideoPoint",
    "Molmo2-VideoSubtitleQA", "Molmo2-VideoTrack",
]
INCOMPLETE = {"Molmo2-VideoCapQA", "Molmo2-VideoSubtitleQA", "Molmo2-VideoTrack"}
REQUIRED_CSV = {
    "inventory.csv": 22,
    "dataset-capabilities.csv": 18,
    "task-distribution.csv": 1,
    "quality-audit.csv": 18,
    "training-recipes.csv": 15,
    "training-references.csv": 10,
    "recommended-datasets.csv": 16,
    "source-accounting.csv": 18,
    "quality-risk-matrix.csv": 18,
    "capability-gap.csv": 7,
    "embodied-task-distribution.csv": 7,
    "sampling-strata.csv": 54,
    "sample-length-distribution.csv": 288,
    "sample-length-summary.csv": 36,
    "sample-annotation-density.csv": 144,
    "sample-annotation-density-summary.csv": 18,
    "sample-media-summary.csv": 18,
    "sample-text-quality-summary.csv": 18,
    "sample-task-annotations.csv": 8810,
    "sample-task-distribution.csv": 18,
}


class Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str, str]] = []
        self.sample_count = 0
        self.record_count = 0
        self.qa_count = 0
        self.image_count = 0
        self.video_count = 0
        self.lazy_image_count = 0
        self.deferred_video_count = 0
        self.sample_record_ids: list[str] = []
        self.dataset = ""
        self.sample_id = ""
        self.viewport = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        classes = values.get("class", "").split()
        self.sample_count += int("sample" in classes)
        self.record_count += int("record-sample" in classes)
        self.qa_count += int("qa-item" in classes)
        self.image_count += int(tag == "img")
        self.video_count += int(tag == "video")
        self.lazy_image_count += int(tag == "img" and values.get("loading") == "lazy")
        self.deferred_video_count += int(tag == "video" and values.get("preload") == "none")
        if tag == "details" and "sample-record" in classes:
            self.sample_record_ids.append(values.get("data-sample-id", ""))
        if tag == "main":
            self.dataset = values.get("data-dataset", "")
            self.sample_id = values.get("data-sample-id", "")
        if tag == "meta" and values.get("name") == "viewport":
            self.viewport = True
        if tag in {"img", "video", "source", "script"} and values.get("src"):
            self.links.append((tag, "src", values["src"]))
        if tag == "video" and values.get("poster"):
            self.links.append((tag, "poster", values["poster"]))
        if tag in {"a", "link"} and values.get("href"):
            self.links.append((tag, "href", values["href"]))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_magic(path: Path) -> None:
    head = path.read_bytes()[:16]
    valid = (
        head.startswith(b"\xff\xd8\xff")
        or head.startswith(b"\x89PNG\r\n\x1a\n")
        or head.startswith((b"GIF87a", b"GIF89a"))
        or (len(head) >= 12 and head[4:12] in {b"ftypisom", b"ftypmp42", b"ftypavc1", b"ftypqt  "})
        or head.startswith(b"RIFF")
        or head.startswith(b"BM")
    )
    if not valid:
        raise AssertionError(f"unrecognized media signature: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(dataset: str) -> dict[str, int]:
    directory = ROOT / "sub-dataset" / dataset
    summary = load_json(directory / "sampling-summary.json")
    manifest = load_jsonl(directory / "sampling-manifest.jsonl")
    record_manifest = load_jsonl(directory / "record-sampling-manifest.jsonl")
    decisions = load_jsonl(directory / "sampling-candidate-decisions.jsonl")
    expected_selected = int(summary.get("selected_count", 0))
    expected_records = int(summary.get("record_sample_count", 0))
    candidate_count = int(summary["candidate_count"])
    if len(manifest) != expected_selected:
        raise AssertionError(f"{dataset}: manifest count {len(manifest)} != {expected_selected}")
    if expected_records and len(record_manifest) != expected_records:
        raise AssertionError(f"{dataset}: record manifest count {len(record_manifest)} != {expected_records}")
    source_root = str(summary.get("source_root", ""))
    if not source_root.startswith(SOURCE_ROOT) or source_root.startswith(DENIED_ROOT):
        raise AssertionError(f"{dataset}: sampling summary source root is outside allowlist")
    draw_orders = [int(row["draw_order"]) for row in manifest + decisions]
    if len(draw_orders) != candidate_count or set(draw_orders) != set(range(candidate_count)):
        raise AssertionError(f"{dataset}: candidate partition is not 0..{candidate_count - 1}")
    if len(draw_orders) != len(set(draw_orders)):
        raise AssertionError(f"{dataset}: duplicate draw order")

    hashes = []
    for row in manifest:
        source_uri = str(row.get("source_uri", ""))
        if source_uri and source_uri.startswith("/mnt/") and not source_uri.startswith(SOURCE_ROOT + "/"):
            raise AssertionError(f"{dataset}: manifest source URI outside allowlist")
        relative = row.get("media_archive_path") or row.get("preview_archive_path")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise AssertionError(f"{dataset}: unsafe/missing media archive path")
        path = directory / relative
        if not path.is_file():
            raise AssertionError(f"{dataset}: missing media {relative}")
        assert_magic(path)
        digest = row.get("sha256") or row.get("source_video_sha256")
        if digest:
            hashes.append(str(digest))
        record = row.get("record_archive_path")
        if record and not (directory / record).is_file():
            raise AssertionError(f"{dataset}: missing record {record}")
    for row in record_manifest:
        source_uri = str(row.get("source_uri", ""))
        if source_uri and not source_uri.startswith(SOURCE_ROOT + "/"):
            raise AssertionError(f"{dataset}: record manifest source URI outside allowlist")
        record = row.get("record_archive_path")
        if not record or not (directory / record).is_file():
            raise AssertionError(f"{dataset}: missing archived record evidence {record}")
    if len(hashes) != len(set(hashes)):
        raise AssertionError(f"{dataset}: accepted media uniqueness hash repeats")

    schema = load_json(ROOT / "source-schemas" / f"{dataset}.json")
    if schema.get("dataset") != dataset or not schema.get("source_only"):
        raise AssertionError(f"{dataset}: invalid source schema header")
    if not str(schema.get("source_root", "")).startswith(SOURCE_ROOT + "/"):
        raise AssertionError(f"{dataset}: source schema root outside allowlist")
    if len(schema.get("representative_source_records", [])) < 3:
        raise AssertionError(f"{dataset}: fewer than three source records in schema")

    page = ROOT / f"{dataset}.html"
    parser = Parser()
    parser.feed(page.read_text(encoding="utf-8"))
    if not parser.viewport:
        raise AssertionError(f"{dataset}: no viewport meta")
    if parser.sample_count or parser.record_count:
        raise AssertionError(f"{dataset}: legacy sample card classes are still present")
    detail_rows = manifest or record_manifest
    expected_ids = {str(row["sample_id"]) for row in detail_rows}
    if parser.dataset != dataset:
        raise AssertionError(f"{dataset}: report data-dataset identity is missing or incorrect")
    if len(parser.sample_record_ids) != 200 or set(parser.sample_record_ids) != expected_ids:
        raise AssertionError(f"{dataset}: embedded sample records do not match the 200-row manifest")
    expected_qa = sum(
        1 for row in read_csv("sample-task-annotations.csv") if row["dataset"] == dataset
    )
    if parser.qa_count != expected_qa:
        raise AssertionError(f"{dataset}: embedded QA count {parser.qa_count} != {expected_qa}")
    expected_videos = sum(bool(row.get("source_video_archive_path")) for row in detail_rows)
    expected_images = sum(
        not row.get("source_video_archive_path")
        and bool(row.get("media_archive_path") or row.get("preview_archive_path"))
        for row in detail_rows
    )
    if parser.video_count != expected_videos or parser.image_count != expected_images:
        raise AssertionError(
            f"{dataset}: embedded media count images={parser.image_count}/{expected_images}, "
            f"videos={parser.video_count}/{expected_videos}"
        )
    if parser.lazy_image_count != expected_images or parser.deferred_video_count != expected_videos:
        raise AssertionError(f"{dataset}: embedded media must use lazy images and deferred videos")
    validate_links(page, parser)
    return {
        "accepted": expected_selected,
        "record_evidence": expected_records if dataset in INCOMPLETE else 0,
    }


def validate_links(page: Path, parser: Parser) -> None:
    content = page.read_text(encoding="utf-8")
    if "url(http" in content.lower():
        raise AssertionError(f"{page.name}: external CSS dependency")
    for tag, attribute, value in parser.links:
        lower = value.lower()
        if tag in {"img", "script", "link"} and lower.startswith(("http://", "https://", "//")):
            raise AssertionError(f"{page.name}: external dependency {tag} {value}")
        if lower.startswith(("http://", "https://", "mailto:")) or value.startswith("#"):
            continue
        target = unquote(value.split("#", 1)[0])
        if not target:
            continue
        path = (page.parent / target).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise AssertionError(f"{page.name}: path escapes delivery root: {value}") from exc
        if not path.exists():
            raise AssertionError(f"{page.name}: broken {attribute}={value}")


def detail_manifest(dataset: str) -> list[dict]:
    directory = ROOT / "sub-dataset" / dataset
    visual = load_jsonl(directory / "sampling-manifest.jsonl")
    return visual or load_jsonl(directory / "record-sampling-manifest.jsonl")


def validate_sample_aggregates() -> dict[str, int]:
    if (ROOT / "sample-details").exists():
        raise AssertionError("sample-details must not be published")
    totals = Counter()
    for dataset in DATASETS:
        rows = detail_manifest(dataset)
        if len(rows) != 200 or len({row["sample_id"] for row in rows}) != 200:
            raise AssertionError(f"{dataset}: manifest must contain 200 unique sample IDs")
        totals["sample_records"] += len(rows)
        for row in rows:
            sample_id = str(row["sample_id"])
            video = row.get("source_video_archive_path")
            image = row.get("media_archive_path")
            preview = row.get("preview_archive_path")
            if video:
                video_path = ROOT / "sub-dataset" / dataset / str(video)
                if not video_path.is_file() or video_path.stat().st_size != int(row["source_video_byte_length"]):
                    raise AssertionError(f"{dataset}/{sample_id}: missing or wrong-sized source video")
                assert_magic(video_path)
                if sha256(video_path) != row["source_video_sha256"]:
                    raise AssertionError(f"{dataset}/{sample_id}: source video SHA256 mismatch")
                totals["source_videos"] += 1
            elif image:
                totals["source_images"] += 1
            elif preview:
                totals["preview_only"] += 1
            else:
                totals["media_unavailable"] += 1

    totals["qa_annotation_units"] = len(read_csv("sample-task-annotations.csv"))
    expected = {
        "sample_records": 3600,
        "qa_annotation_units": 8810,
        "source_images": 2600,
        "source_videos": 400,
        "preview_only": 0,
        "media_unavailable": 600,
    }
    for key, value in expected.items():
        if totals[key] != value:
            raise AssertionError(f"sample aggregates: {key}={totals[key]} != {value}")
    video_summary = load_json(ROOT / "artifacts" / "selected-video-archive-summary.json")
    if video_summary.get("archived_count") != 400 or video_summary.get("failure_count") or not video_summary.get("source_archives_unchanged"):
        raise AssertionError("selected-video-archive-summary.json: source video archive is incomplete")
    return dict(totals)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require_columns(name: str, rows: list[dict[str, str]], columns: set[str]) -> None:
    observed = set(rows[0]) if rows else set()
    missing = columns - observed
    if missing:
        raise AssertionError(f"{name}: missing required columns {sorted(missing)}")


def validate_csv() -> None:
    loaded: dict[str, list[dict[str, str]]] = {}
    for name, minimum in REQUIRED_CSV.items():
        rows = read_csv(name)
        loaded[name] = rows
        if len(rows) < minimum:
            raise AssertionError(f"{name}: {len(rows)} rows < {minimum}")

    inventory = loaded["inventory.csv"]
    require_columns("inventory.csv", inventory, {
        "entry", "status", "source_path", "source_version", "source_splits", "source_formats",
        "file_count", "source_bytes_observed", "source_records", "unique_media", "image_count",
        "video_count", "audio_count", "unique_scene_episode_trajectory", "supervision_units",
        "supervised_token_estimate", "media_storage", "media_and_lineage_identity",
        "reproducible_locator", "source_population_digest", "license", "evidence_scope", "evidence_time",
    })
    included = [row for row in inventory if row["status"] == "included"]
    if {row["entry"] for row in included} != set(DATASETS):
        raise AssertionError("inventory.csv: included dataset partition does not match expected datasets")
    for row in included:
        if not row["source_path"].startswith(SOURCE_ROOT + "/") or row["source_path"].startswith(DENIED_ROOT):
            raise AssertionError(f"inventory.csv: source path outside allowlist for {row['entry']}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_population_digest"] or ""):
            raise AssertionError(f"inventory.csv: invalid source digest for {row['entry']}")

    tasks = loaded["task-distribution.csv"]
    require_columns("task-distribution.csv", tasks, {
        "dataset", "task", "task_group", "count", "classification_basis",
        "embodied_capability_level", "estimated_supervised_tokens", "token_allocation_method",
        "source_path", "evidence_date", "simple_task_category", "simple_task_label",
        "task_family", "fine_task", "task_description", "input", "output", "domain",
        "taxonomy_version", "taxonomy_classification_basis", "source_classification_basis",
    })
    for row in tasks:
        if row["dataset"] not in DATASETS or not row["source_path"].startswith(SOURCE_ROOT + "/"):
            raise AssertionError("task-distribution.csv: invalid dataset or source path")

    sample_tasks = loaded["sample-task-annotations.csv"]
    require_columns("sample-task-annotations.csv", sample_tasks, {
        "dataset", "sample_id", "qa_index", "source_task_label", "simple_task_category",
        "simple_task_label", "task_family", "fine_task", "task_description", "input", "output",
        "domain", "classification_basis", "taxonomy_version", "question_preview",
    })
    if len(sample_tasks) != 8810:
        raise AssertionError(f"sample-task-annotations.csv: expected 8810 rows, got {len(sample_tasks)}")
    identities = {(row["dataset"], row["sample_id"], row["qa_index"]) for row in sample_tasks}
    if len(identities) != len(sample_tasks) or {row["dataset"] for row in sample_tasks} != set(DATASETS):
        raise AssertionError("sample-task-annotations.csv: duplicate identity or dataset partition mismatch")
    for row in sample_tasks:
        if not row["fine_task"] or not row["task_description"]:
            raise AssertionError(f"sample-task-annotations.csv: invalid task record for {row['sample_id']}")

    sample_distribution = loaded["sample-task-distribution.csv"]
    require_columns("sample-task-distribution.csv", sample_distribution, {
        "dataset", "source_task_label", "simple_task_category", "simple_task_label",
        "task_family", "fine_task", "task_description", "input", "output", "domain",
        "classification_basis", "taxonomy_version", "sample_count", "annotation_unit_count",
        "dataset_annotation_units", "within_dataset_percentage", "example_sample_id",
    })
    annotations_by_dataset = Counter(row["dataset"] for row in sample_tasks)
    distribution_by_dataset: Counter[str] = Counter()
    shares_by_dataset: dict[str, float] = defaultdict(float)
    for row in sample_distribution:
        distribution_by_dataset[row["dataset"]] += int(row["annotation_unit_count"])
        shares_by_dataset[row["dataset"]] += float(row["within_dataset_percentage"])
    if distribution_by_dataset != annotations_by_dataset:
        raise AssertionError("sample-task-distribution.csv: annotation totals do not match detail task rows")
    for dataset in DATASETS:
        if abs(shares_by_dataset[dataset] - 1.0) > 1e-9:
            raise AssertionError(f"sample-task-distribution.csv: {dataset} shares do not sum to one")

    accounting = loaded["source-accounting.csv"]
    require_columns("source-accounting.csv", accounting, {
        "dataset", "canonical_source_records", "confirmed_effective_records",
        "known_rejected_records", "unresolved_records", "accounting_check", "decision_basis",
    })
    if {row["dataset"] for row in accounting} != set(DATASETS):
        raise AssertionError("source-accounting.csv: dataset partition mismatch")
    for row in accounting:
        canonical = int(row["canonical_source_records"])
        parts = sum(int(row[key]) for key in (
            "confirmed_effective_records", "known_rejected_records", "unresolved_records"
        ))
        if canonical != parts or canonical != int(row["accounting_check"]):
            raise AssertionError(f"source-accounting.csv: unbalanced row for {row['dataset']}")

    gaps = loaded["capability-gap.csv"]
    if [row["embodied_level"] for row in gaps] != [f"E{index}" for index in range(7)]:
        raise AssertionError("capability-gap.csv: expected ordered E0-E6 rows")
    embodied = loaded["embodied-task-distribution.csv"]
    if [row["embodied_level"] for row in embodied] != [f"E{index}" for index in range(7)]:
        raise AssertionError("embodied-task-distribution.csv: expected ordered E0-E6 rows")

    recipes = loaded["training-recipes.csv"]
    require_columns("training-recipes.csv", recipes, {
        "recipe", "stage", "total_mix_percent", "capability_target", "datasets",
        "unique_media_target", "effective_supervision_target", "token_target", "per_media_cap",
        "entry_and_quota_rule", "validation_benchmarks", "evidence_basis", "reference_ids",
    })
    recipe_names = {row["recipe"] for row in recipes}
    if len(recipe_names) != 3:
        raise AssertionError("training-recipes.csv: expected exactly three recipes")
    for recipe in recipe_names:
        rows = [row for row in recipes if row["recipe"] == recipe]
        if {row["stage"] for row in rows} != {f"Stage {index}" for index in range(1, 6)}:
            raise AssertionError(f"training-recipes.csv: {recipe} lacks Stage 1-5")
        if sum(int(row["total_mix_percent"]) for row in rows) != 100:
            raise AssertionError(f"training-recipes.csv: {recipe} percentages do not sum to 100")

    references = loaded["training-references.csv"]
    require_columns("training-references.csv", references, {
        "citation_id", "authors", "year", "title", "arxiv_id", "url",
        "supported_principle", "used_in", "citation_scope", "metadata_source", "verified_date",
    })
    reference_ids = {int(row["citation_id"]) for row in references}
    if reference_ids != set(range(1, 11)):
        raise AssertionError("training-references.csv: citation IDs must be 1..10")
    for row in references:
        if row["url"] != f'https://arxiv.org/abs/{row["arxiv_id"]}':
            raise AssertionError(f"training-references.csv: invalid arXiv URL for [{row['citation_id']}]")
        if row["metadata_source"] != "arXiv official API" or not row["supported_principle"]:
            raise AssertionError(f"training-references.csv: incomplete evidence for [{row['citation_id']}]")
    used_reference_ids = {
        int(reference_id)
        for row in recipes
        for reference_id in row["reference_ids"].split(";")
        if reference_id
    }
    if used_reference_ids != reference_ids:
        raise AssertionError("training-recipes.csv: stage references do not cover bibliography IDs 1..10")

    length_summary = loaded["sample-length-summary.csv"]
    if sum(int(row["sample_total"]) for row in length_summary if row["side"] == "input") != 3600:
        raise AssertionError("sample-length-summary.csv: unexpected input sample total")
    media_summary = loaded["sample-media-summary.csv"]
    if sum(int(row["sample_total"]) for row in media_summary) != 3600:
        raise AssertionError("sample-media-summary.csv: unexpected sample total")


def validate_scope_and_evidence_paths() -> None:
    scope = load_json(ROOT / "scope-declaration.json")
    if scope.get("analysis_mode") != "source-only" or scope.get("server_allowlist") != [SOURCE_ROOT]:
        raise AssertionError("scope-declaration.json: invalid source-only allowlist")
    if DENIED_ROOT not in scope.get("server_denylist", []):
        raise AssertionError("scope-declaration.json: converted source denylist is missing")

    evidence = load_json(ROOT / "source-readonly-evidence.json")
    for row in evidence.get("datasets", []):
        source_root = str(row.get("source_root", ""))
        if not source_root.startswith(SOURCE_ROOT + "/") or source_root.startswith(DENIED_ROOT):
            raise AssertionError(f"source-readonly-evidence.json: source outside allowlist for {row.get('dataset')}")

    evidence_paths = list(ROOT.glob("*.html")) + list(ROOT.glob("*.csv"))
    evidence_paths += list((ROOT / "source-schemas").rglob("*.json"))
    for directory in (ROOT / "sub-dataset").iterdir():
        for name in (
            "sampling-manifest.jsonl", "record-sampling-manifest.jsonl",
            "sampling-candidate-decisions.jsonl", "sampling-summary.json",
            "source-layout.txt", "source-schema.json",
        ):
            path = directory / name
            if path.exists():
                evidence_paths.append(path)
    for path in evidence_paths:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if DENIED_ROOT in content:
            raise AssertionError(f"converted-data path used in report evidence: {path.relative_to(ROOT)}")


def validate_index_content() -> None:
    content = (ROOT / "index.html").read_text(encoding="utf-8")
    required_titles = [
        "简单任务形态分布",
        "标注单元加权任务族",
        "每个数据集的抽样任务族堆叠图",
        "抽样任务详细台账",
        "全量源任务族分布",
        "全量源任务详细台账",
        "数据集源规模柱状图",
        "记录数与唯一媒体数对比",
        "数据集 × 能力强度热力图（辅助视图）",
        "每媒体标注数量分布",
        "输入输出长度分布",
        "具身相关监督证据分布",
        "当前覆盖与目标差距",
        "配比依据",
        "通用能力保持",
        "通用-具身均衡",
        "具身推理优先",
        "收益与代价",
        "参考文献与方案映射",
        "引用边界",
    ]
    missing = [title for title in required_titles if title not in content]
    if missing:
        raise AssertionError(f"index.html: missing required chart titles {missing}")
    forbidden = [term for term in ("sample-details/", "3600 条抽检明细", "泄露", "跨源重叠") if term in content]
    if forbidden:
        raise AssertionError(f"index.html: removed content is still visible {forbidden}")
    if 'id="quality"' in content:
        raise AssertionError("index.html: removed quality section is still present")
    for dataset in DATASETS:
        marker = f'<td><a href="{dataset}.html">{dataset}</a></td>'
        if marker not in content:
            raise AssertionError(f"index.html: dataset table link missing for {dataset}")
    if "独立报告：" in content:
        raise AssertionError("index.html: duplicate standalone report link list is still present")


def main() -> int:
    required = [
        "index.html", "methodology.md", "source-readonly-evidence.json",
        "scope-declaration.json", "SOURCE_ANALYSIS_PROMPT.md",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            raise AssertionError(f"missing required file: {name}")
    validate_csv()
    validate_scope_and_evidence_paths()

    totals = Counter()
    for dataset in DATASETS:
        totals.update(validate_dataset(dataset))
    if totals["accepted"] != 3000 or totals["record_evidence"] != 600:
        raise AssertionError(f"unexpected sample totals: {dict(totals)}")
    sample_totals = validate_sample_aggregates()

    index_parser = Parser()
    index_path = ROOT / "index.html"
    index_parser.feed(index_path.read_text(encoding="utf-8"))
    if not index_parser.viewport:
        raise AssertionError("index.html: no viewport meta")
    validate_links(index_path, index_parser)
    validate_index_content()

    evidence = load_json(ROOT / "source-readonly-evidence.json")
    if not evidence.get("all_unchanged") or len(evidence.get("datasets", [])) != 18:
        raise AssertionError("source read-only evidence is incomplete or changed")

    result = {
        "status": "passed",
        "datasets": len(DATASETS),
        "accepted_visuals": totals["accepted"],
        "record_evidence": totals["record_evidence"],
        "sample_records": sample_totals["sample_records"],
        "sample_detail_pages": 0,
        "embedded_sample_records": sample_totals["sample_records"],
        "qa_annotation_units": sample_totals["qa_annotation_units"],
        "archived_source_videos": sample_totals["source_videos"],
        "source_fingerprints_unchanged": True,
        "source_accounting_balanced": True,
        "required_charts_present": 7,
        "converted_data_used_as_evidence": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
