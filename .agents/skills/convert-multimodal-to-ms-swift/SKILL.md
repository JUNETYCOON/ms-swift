---
name: convert-multimodal-to-ms-swift
description: Inspect and convert image, video, audio, caption, VQA, grounding, point, and tracking datasets into deterministic ms-swift JSONL for SFT. Use for schema discovery, media-level train/val splitting, parallel conversion, rejection reporting, media validation, and leakage audits.
---

# Multimodal ms-swift Conversion

## Workflow

1. Inspect source files and real records before designing adapters.
2. Produce a schema report covering fields, types, missing values, media IDs and task distributions.
3. Define one adapter per source schema.
4. Build media identities and lineage before creating derived samples.
5. Split leakage groups, then expand QA/caption/grounding records.
6. Convert in parallel and write deterministically.
7. Validate schema, media, coordinates and split isolation.
8. Update conversion documentation and report exact counts.

## Split Contract

Treat the original media family as the minimum split unit.

- Keep all QA and captions for one image in one split.
- Keep all clips, frames and windows from one original video in one split.
- Join multi-media samples into connected components when they share any media.
- Use namespaced source IDs, lineage roots and available content hashes.
- Protect official validation/test media from entering training.
- Assign unanchored groups with versioned SHA-256, never row-level randomness or `hash()`.
- Quarantine records whose media identity or lineage cannot be established.
- Persist the mapping in `media_groups.tsv`.

## Format Contract

Use JSONL with `messages` and ordered `images`, `videos` or `audios`.

For SFT, normally emit user/assistant message pairs. Ensure every media placeholder has exactly one matching resource.

For grounding:

- Replace each `<ref-object>` from `objects.ref`.
- Replace each `<bbox>` from `objects.bbox`.
- Treat ref and bbox counts independently.
- Accept only 2-value points or 4-value boxes.
- Set `bbox_type` explicitly.
- Provide `image_id` for multi-image real-coordinate boxes.
- Reject structurally invalid or materially out-of-range coordinates.

Do not assume `QWENVL_BBOX_FORMAT=new` converts generic objects into cookbook JSON.

## Implementation Contract

Stream large inputs, use bounded multi-process workers, preserve source ordering, and write through temporary files. Never fabricate missing media or annotations. Record every rejection with source identity and reason.

## Required Validation

Assert:

- each retained record belongs to exactly one split;
- each leakage group has exactly one split;
- train and val share no media ID, lineage root or content hash;
- message roles and media placeholders are valid;
- object placeholder counts and coordinate ranges are valid;
- local media exists and decodes;
- source accounting balances;
- repeated runs and different worker counts produce identical split mappings.

Attempt an ms-swift loader and template-render smoke test. Report any environment limitation instead of claiming validation succeeded.
