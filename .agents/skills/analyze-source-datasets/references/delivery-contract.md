# Source Dataset Analysis Delivery Contract

## Delivery Layout

```text
output-root/
├── index.html
├── {Dataset}.html
├── source-readonly-evidence.json
├── sub-dataset/
│   └── {Dataset}/
│       ├── media/ or source-sample/
│       ├── records/
│       ├── source-metadata/
│       ├── derived-preview/          # only when needed; never call derived media original
│       ├── sampling-manifest.jsonl
│       ├── sampling-candidate-decisions.jsonl
│       ├── sampling-summary.json
│       ├── source-layout.txt
│       └── source-schema.json
└── tools/                            # optional reproducibility scripts
```

Names may reflect native source directories, but every manifest archive path must resolve beneath its dataset archive directory. Reject absolute archive paths and `..` traversal.

## Sampling Definitions

### Population

The population is the complete set of declared sampling units, not the subset that is cheap to enumerate. Record:

- `sampling_unit`: one of `logical_record`, `unique_image`, `unique_video`, `archive_member`, or `joined_instruction_row`;
- `population_total` and counts by split/container when applicable;
- a stable population locator, such as shard plus row, archive plus member, or source-relative path;
- the exact rule for exclusions from the population, such as non-media members or a corrupt archive.

### Candidate draw

Use a deterministic permutation without replacement. The default target is 200 accepted visuals and the default recorded draw window is 400 candidates. If 400 is insufficient, extend the same permutation and update `candidate_count`; do not redraw with a new seed.

Every draw order from `0` through `candidate_count - 1` must appear exactly once in either:

- `sampling-manifest.jsonl` for accepted entries; or
- `sampling-candidate-decisions.jsonl` for rejected and reserve entries.

Common decision reasons include `reserve_candidate_not_needed`, `missing_media`, `decode_failure`, `duplicate_media_sha256`, `duplicate_video_id`, `text_only_record_without_image`, and `join_key_not_found`.

## Manifest Contract

Each accepted manifest row must contain:

- `sample_id`: stable and unique within the dataset;
- `draw_order`: integer candidate position;
- a population locator (`population_index`, source row, or archive/member identity);
- `media_archive_path` or `preview_archive_path`;
- original media `sha256`, or for video `source_video_sha256` plus `video_id`;
- dimensions/format when applicable;
- record or metadata archive paths when those artifacts exist;
- `raw_json` with `archive_path`, `sha256`, `origin`, and `encoding` for the complete sample record;
- source input and output previews or fields needed to trace them.

For a derived preview, also store the preview hash and a human-readable generation description. Never substitute a preview hash for the source video's uniqueness hash.

Follow [raw-json-contract.md](raw-json-contract.md). For JSON/JSONL, archive the exact record payload without reserialization. For non-JSON sources, preserve the native record and mark the complete reversible JSON representation `derived_lossless_projection`.

If the accepted source record contains ground-truth bbox or point annotations, also require:

- `ground_truth_overlay_path` and its SHA256 under `derived-preview/`;
- the source annotation locator and explicit bbox/point coordinate convention;
- the target image identity or video frame index/timestamp;
- declared and rendered bbox/point counts, which must match;
- `ground_truth_overlay_status`, with unresolved mappings excluded from visualization-valid totals.

Render the overlay on a copy of the decoded original-resolution image or exact referenced frame, label it visibly as `GT`, and retain a separate link to the clean original. Do not guess formats, silently clip invalid coordinates, or use coordinate text as the only visualization.

## Summary Contract

`sampling-summary.json` must contain at least:

```json
{
  "dataset": "DatasetName",
  "source_root": "/source/root",
  "sampling_unit": "logical_record",
  "population_total": 1000,
  "seed": 2026080401,
  "candidate_count": 400,
  "selected_count": 200,
  "invalid_or_duplicate_count": 0,
  "reserve_candidate_count": 200,
  "sampling_method": "uniform random permutation over all logical records",
  "read_only_source": true
}
```

Add dataset-specific distributions without forcing unrelated datasets into a common label taxonomy.

## Source Read-Only Evidence

The delivery-root `source-readonly-evidence.json` contains one entry per dataset:

```json
{
  "version": 1,
  "datasets": [
    {
      "dataset": "DatasetName",
      "access_mode": "ssh",
      "source_root": "/source/root",
      "fingerprint_algorithm": "sha256 over sorted population locators, sizes, and mtimes",
      "before_digest": "...",
      "after_digest": "...",
      "unchanged": true,
      "collector_transport": "code via stdin; tar/artifacts via stdout",
      "temporary_workspace": "/tmp/source-analysis-...",
      "read_operations": ["directory enumeration", "metadata read", "media read"]
    }
  ]
}
```

Do not include credentials, tokens, private keys, or secret-bearing command strings. A matching fingerprint supports the claim that the declared population did not change during collection; it is not a general proof that no unrelated process wrote elsewhere.

## Source Schema And Layout

`source-layout.txt` is an observed, compact tree or container listing with file roles and counts. `source-schema.json` records actual field names, types, nullability/missingness observations, and representative structural examples. It must distinguish:

- fields physically present in the source;
- joined fields from another source table;
- derived analysis fields;
- optional or missing fields.

Dataset names do not imply fields. In particular, never claim bbox because a source is named COCO; only claim it if the inspected conversion contains bbox coordinates or a traceable joined annotation source.

## HTML Acceptance

Each report must:

- include a viewport meta tag and no external CSS/JS/image dependency;
- follow the five-section total-to-detail narrative;
- contain exactly one `.sample` card per accepted manifest row;
- contain exactly one visible-by-default `.raw-json` block per sample card, with sample ID, role, archive path, SHA-256, and origin attributes matching the manifest;
- render the complete archived JSON text without truncation or reserialization; the parsed DOM `textContent` must equal the UTF-8 archive exactly;
- link each card to archived media and, where available, archived records;
- use the GT overlay as the primary visual whenever the source record contains bbox/point truth, while linking the clean source separately;
- require every GT overlay to decode, target the recorded source image/frame, and match its declared bbox/point counts;
- state the population, sampling unit, seed, candidate/rejection method, and evidence boundary;
- show missing fields explicitly rather than filling them in;
- work from `file://` at desktop and narrow widths.

The index must link every report. Browser acceptance checks desktop and narrow viewports for horizontal overflow, missing images, console/page errors, and incoherent overlap.

## Dataset-Specific Notes

- Parquet/Arrow: inspect footer/schema for all shards; preserve shard and row coordinates.
- Joined tables: preserve both instruction and media-table locators and record join failures.
- ZIP/TAR: reject unsafe extraction paths; preserve archive/member identity.
- Videos: require unique video IDs and source-video hashes; thumbnails are derived previews.
- Mixed records: text-only rows remain part of the logical population when that is the training unit.
- Broken media: record rejection; never silently replace it outside draw order.
- OSSFS/FUSE: retry transient empty enumerations after a read-only mount prime; unresolved zero listings are errors, not empty datasets.
