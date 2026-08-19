---
name: analyze-source-datasets
description: Read-only source-dataset inventory, full-population deterministic random sampling, sample-material archiving, verbatim complete sample JSON display, per-dataset standalone HTML analysis, and evidence-based delivery validation. Use when the user asks to analyze or spot-check one dataset, a named list of datasets, or every dataset under a local or SSH root; especially when each source needs an input/output explanation, distribution analysis, fixed-count visual samples, complete unchanged JSON records, preserved source organization, manifests, and a total-to-detail HTML report.
---

# Analyze Source Datasets

## Purpose

Turn one dataset, several named datasets, or a dataset root into an auditable collection of source-level sample reports. Treat the source as immutable, sample from the complete declared population with a fixed seed, preserve original bytes and metadata where possible, and make every claim traceable to a source record or a documented derivation.

Read [references/delivery-contract.md](references/delivery-contract.md) before collecting data. Use [assets/report-template.html](assets/report-template.html) as the report structure and visual baseline. Run [scripts/validate_delivery.py](scripts/validate_delivery.py) before declaring completion.

## Establish The Contract

State these assumptions before implementation:

- Input mode: one dataset, an explicit dataset list, or automatic inventory under a root.
- Access mode: local filesystem or an explicitly named SSH host.
- Output root: use the user-provided path; otherwise ask because reports and copied media are material writes.
- Sampling unit: logical record, unique image, unique video, archive member, or joined instruction row.
- Target: default 200 accepted visuals per dataset.
- Candidate draw: default 400 entries from a deterministic permutation; extend the same permutation only when needed to reach the target.
- Seed: record one fixed integer per dataset. Never use an unrecorded random state.
- Dataset boundary: for root inventory, explain how immediate children or metadata files define datasets. Record every discovered child, including skipped or unreadable entries and reasons.

Stop and ask only when a choice changes the meaning of a sample, the dataset boundary, or the authorized source/output scope. Do not silently sample images when the source's actual training unit is a question row or instruction row.

## Enforce Read-Only Source Handling

The source roots are strictly read-only unless the user explicitly authorizes source writes.

- Never create caches, bytecode, temp files, lockfiles, indexes, virtual environments, previews, or derived data inside a source root.
- Set `PYTHONDONTWRITEBYTECODE=1`; point caches and temporary files to a local temporary or POSIX workspace outside the source.
- Do not install dependencies on OSSFS/FUSE. Install only in an authorized persistent system disk or a temporary/local POSIX environment.
- For SSH sources, send collection code through stdin and return artifacts through stdout. Keep all staging outside the source root.
- Use read-only operations for discovery and reads. Avoid tools whose normal behavior creates sidecars.
- Capture a before and after source inventory fingerprint over the declared sampling population. Store the evidence at the delivery root as `source-readonly-evidence.json` and require identical digests.
- Do not treat `"read_only_source": true` as sufficient proof. It is a declaration, not evidence.

On `ossfs2`, an unexplained empty listing is not proof of an empty dataset. Prime the mount with a read-only parent listing, retry, and record the anomaly. If it remains empty, report it as unreadable or unresolved rather than as a zero-sized dataset.

## Inventory And Model The Population

For each dataset:

1. Enumerate the entire source population before drawing samples.
2. Record source paths, file/container types, splits, counts, schema, and media-storage form.
3. Define a stable population index mapping each sampling unit to its source locator.
4. Write `source-layout.txt` from the observed hierarchy and `source-schema.json` from actual records.
5. Separate directly observed source facts, derived values, and unsupported fields.

Use structured readers for Parquet/Arrow, JSON/JSONL, ZIP/TAR, and video containers. Do not parse structured formats with ad hoc text matching.

Handle common layouts explicitly:

- Embedded Parquet/Arrow media: index all logical rows, preserve row/shard locators, and export source media bytes.
- Instruction and image tables: sample instruction rows, perform an explicit keyed join, and retain both row locators.
- ZIP/TAR datasets: build the population from valid media members; preserve member names and container identity.
- Video datasets: sample unique videos, archive source video bytes when practical, and label thumbnails as derived previews with generation parameters.
- Mixed visual/text-only records: keep text-only candidates in the decision log; never replace them invisibly.
- Missing optional metadata: display it as missing or unavailable. Never infer an annotation from pixels or dataset reputation.

For example, a COCO-named local Arrow conversion containing only image bytes and class labels does not contain bounding boxes. Report the schema actually read, not the capabilities of the canonical COCO release.

## Draw And Archive Samples

Generate a deterministic permutation over the complete population and inspect candidates in draw order. Accept an item only when the relevant source record and media are readable and the visual is unique.

- Images: require successful decoding and a unique SHA256 of original media bytes.
- Videos: require successful container inspection, a unique video ID, and a unique SHA256 of source video bytes.
- Reject broken, missing, or duplicate media with an explicit reason.
- Mark unused candidates as reserve entries.
- Ensure accepted, rejected, and reserve entries exactly partition every recorded candidate draw order.
- If 200 valid unique visuals do not exist, archive all valid visuals, mark the dataset incomplete, and do not fabricate or duplicate samples.

Archive original media bytes, original source records, and source metadata where possible. Preserve source-relative organization under `sub-dataset/{dataset}/`; use a clearly named `derived-preview/` directory for generated views. Do not resize or re-encode files presented as originals.

## Preserve And Display Complete Sample JSON

Read [references/raw-json-contract.md](references/raw-json-contract.md) before archiving records or building sample cards.

- For JSON/JSONL sources, archive each accepted record's exact UTF-8 payload without reserialization, field reordering, pretty-printing, normalization, or truncation.
- Add a `raw_json` descriptor with archive path, SHA-256, origin, and encoding to every accepted manifest row.
- Put the complete HTML-escaped payload directly in every sample card, visible by default. A fixed-height scroll area is allowed; excerpts, ellipses, and link-only access are not.
- Keep friendly field summaries in addition to the raw block, never instead of it.
- If the source is not JSON, preserve its native record and visibly label the complete JSON representation `derived_lossless_projection`; never call it source-exact.

For every accepted sample with ground-truth bounding boxes or points, generate a derived ground-truth overlay on a copy of the original-resolution image or the exact referenced video frame and use that overlay as the gallery's primary visual. Draw every declared box and point, visibly label the layer as `GT`, and link the clean archived original separately. Resolve box encoding, coordinate space, `image_id`, frame/timestamp, and orientation from source metadata; transform against decoded source dimensions rather than HTML display size. Preserve the source annotation and record the coordinate convention, target media/frame, overlay path/hash, and declared/rendered primitive counts in `sampling-manifest.jsonl`. Never infer coordinate semantics, silently clip invalid geometry, or choose an unrelated video keyframe. Mark unresolved mappings or count mismatches as `ground_truth_overlay_unresolved` and do not count those rows as visualization-valid; coordinate text or a clean image alone is not a valid substitute.

## Analyze And Write Reports

Write one standalone HTML report per dataset using a total-to-detail narrative:

1. Overall judgment: what the dataset fundamentally teaches and its strongest limitation.
2. Input, output, and source storage: exact fields, media format, source organization, and missing supervision.
3. Full-population sampling method: sampling unit, total population, seed, candidate partition, rejection logic, and uniqueness rule.
4. Sample distribution: dimensions, splits, task/answer/label categories, missingness, anomalies, and dataset-specific fields.
5. Complete per-sample gallery: all accepted visuals, source locators, input/output previews, hashes, links to archived originals/records, a visible verbatim full-record JSON block, and mandatory GT overlays for samples containing bounding boxes or points.

Describe sampled distributions as sampled evidence unless a statistic was computed over the full population. Label derived previews and interpretations. Do not turn missing values into negative labels, or infer bbox, segmentation, answer truth, or task suitability from images alone.

Create an `index.html` linking every dataset report and archive directory. Keep reports readable from `file://`, self-contained, responsive, and free of external asset dependencies.

## Validate Before Completion

Run:

```bash
python ~/.codex/skills/analyze-source-datasets/scripts/validate_delivery.py \
  /absolute/path/to/output \
  --expected-count 200 \
  --require-source-evidence \
  --browser-check
```

Also compare repository status captured before work with status after work. Only the requested output root and explicitly approved tooling may change. Do not remove or rewrite unrelated user changes.

Completion requires:

- every discovered/in-scope dataset has a report or an explicit unresolved entry;
- manifest, candidate partition, counts, hashes, uniqueness, media decoding, report cards, links, and index all validate;
- desktop and narrow browser checks show no horizontal overflow, broken images, console errors, or overlapping content;
- before/after source population fingerprints match;
- report claims match the archived source schema and records;
- every sample card contains the complete raw JSON payload, and its DOM text, archive bytes, manifest SHA-256, and sample ID agree;
- limitations and unresolved sources are stated, not hidden.

## Delivery Handoff

Report the absolute delivery path, dataset/report count, accepted sample total, validation command and result, source read-only evidence result, and any incomplete datasets. Never claim browser, remote, or decoding validation unless it was actually run.
