---
name: analyze-vlm-benchmarks
description: Audit, pair, score, sample, and present VLM benchmark predictions as traceable Chinese HTML reports with verbatim complete sample/prediction JSON, complete local video access, and browser-compatibility validation. Use when Codex needs to compare baseline and trained checkpoints across self-built validation sets and external benchmarks; inventory existing eval outputs; distinguish official, local-conversion, and diagnostic metrics; compute paired statistics; inspect bad cases with images, keyframes, full videos, or exact JSON records; or validate an offline benchmark-analysis delivery.
---

# Analyze VLM Benchmarks

Build an evidence-backed benchmark delivery from existing predictions whenever possible. Keep source results immutable, separate metric scopes, and make headline results distinct from the qualitative sample gallery.

Read [references/delivery-contract.md](references/delivery-contract.md) before collecting results. Read [references/scoring-and-statistics.md](references/scoring-and-statistics.md) before scoring. Use [assets/report-template.html](assets/report-template.html) as the per-benchmark visual baseline.

## Establish Scope

Record before implementation:

- source host and result roots;
- output root;
- baseline and ours model identities;
- in-scope self-built validation sets and external benchmarks;
- canonical data version, split, prompt, media policy, generation settings, and scorer per benchmark;
- fixed sampling seed, default `20260806` unless the user supplies one.

Treat each discovered benchmark as `complete`, `partial`, `unresolved`, or `excluded`. Never omit an unusable result silently. Stop only when the benchmark boundary, model identity, or authorized output scope cannot be discovered safely.

## Choose The Presentation Policy

Record who controls the headline values before building pages.

- Use `reproducible_primary` when the user asks the agent to score and establish the benchmark result from artifacts.
- Use `user_specified_primary` when the user explicitly declares a supplied result table authoritative for presentation.
- Under `user_specified_primary`, display the supplied Baseline/Ours values as the report's main results. Do not call them historical, replace them with recomputed values, or place shared-subset, smoke, sampled, or diagnostic rerun aggregates in the headline block.
- Preserve exact displayed precision from the supplied table and write a machine-readable `presented-results.json` with `source: user_specified_primary`.
- A user-specified presentation source does not automatically make a metric official. Keep labels such as diagnostic Token-F1 or local conversion accurate when known.
- Keep selected sample evidence separate: media, questions, references, raw Baseline/Ours outputs, parsed answers, per-sample correctness or scores, diagnostic tags, and filters may remain visible.
- If a supplied benchmark has no numerical value, show `结果表未提供数值` and retain its qualitative samples when available.
- If paired samples are unavailable, show the supplied headline result on a status page and state only that the paired sample report is unavailable.

## Classify Evidence Before Comparing

Assign both a run status and an evidence tier. These are different dimensions:

- `official_full`: complete official split and scorer, with scorer parity evidence;
- `reproducible_full`: complete paired split with immutable inputs and a documented benchmark scorer;
- `diagnostic_full`: complete population but only lexical, heuristic, or local-conversion metrics;
- `shared_subset`: a verified ID intersection from unequal partial runs;
- `smoke`: a small paired execution that does not cover the declared benchmark population;
- `historical_verified`: a persisted aggregate artifact whose exact run population is known;
- `historical_unverified`: a user/table/log value without matching prediction and scorer artifacts;
- `unresolved`: no valid paired comparison can be formed.

Under `reproducible_primary`, never replace a reproducible result with a more favorable unsupported summary. Under `user_specified_primary`, follow the explicit presentation contract above and keep any recomputed audit values out of the user-facing headline display. In both modes, never fabricate scorer provenance or claim an official metric without evidence.

## Audit Read-Only Inputs

1. Inventory every candidate prediction, result, config, scorer output, and media root.
2. Capture source fingerprints before and after work outside the source roots:

```bash
python scripts/fingerprint_tree.py snapshot SOURCE_ROOT before.json --label results
python scripts/fingerprint_tree.py snapshot SOURCE_ROOT after.json --label results
python scripts/fingerprint_tree.py compare before.json after.json --output source-readonly-evidence.json
```

3. Prefer a complete prediction with `failed_samples=0` when model hashes and all evaluation settings match. Re-run inference only for missing or incompatible coverage.
4. Compare sharded model files by content hash when paths or checkpoint names differ. Record both path identity and content identity.
5. Do not mix runs with different prompts, sample IDs, media, frame sampling, generation parameters, dataset versions, or scorers.
6. For SSH data, stage code and derived assets in `/tmp` or an authorized output root. Never create caches or dependencies in model, dataset, or historical result directories.
7. Audit train/eval overlap at three levels when training data is available: exact media hash, normalized question/answer text, and near-duplicate media. Record each result as observed, user-provided, or unknown.
8. Treat media leakage, QA leakage, format defects, and fallback media as first-class `data_risk` fields. A technically complete run can still be invalid as clean-generalization evidence.

## Normalize A Registry

Create one canonical registry row per benchmark containing group, task type, run status, evidence tier, data/split, declared population, scored population, paired coverage, model/config identity, metric scope, primary metric, scorer provenance, source files, media state, data-risk fields, and limitations.

Separate:

- `self-val`: local or converted validation data, reported as local/diagnostic unless it exactly reproduces an official release and scorer;
- `external-benchmarks`: independent suites, using the official scorer where available;
- `unresolved`: status pages when a valid baseline/ours pair cannot be formed.

Use distinct count names. `declared_population_n`, `scored_population_n`, `paired_n`, and `displayed_n` must not share a generic “full n” label. For unequal partial runs, report baseline population, ours population, intersection size, one-sided counts, input mismatches, and whether either ID set is a strict subset.

Use [references/task-and-metric-taxonomy.md](references/task-and-metric-taxonomy.md) when assigning task labels or interpreting common benchmark names.

## Score Full Populations

When scoring is in scope, run the benchmark's official scorer on the complete valid split before computing local diagnostics. Preserve the scorer input, raw scorer output, package/repository version, and commit. When the user says no new inference or scoring is needed, reuse the supplied primary table and existing qualitative samples.

- Label official metrics as official only after parity checking the scorer output.
- Label transformed-data metrics as local conversion.
- Label lexical, heuristic, or auto-tagging measures as diagnostic.
- Never substitute sample-gallery accuracy for the full-set score.
- Never compare scalar results from incompatible populations.
- If unequal partial runs have a verified common subset, recompute both sides on that exact intersection and label the result `shared_subset`; preserve the incompatible original aggregates only as historical context.
- A complete scorer run over a smoke file is still `smoke`, not a complete benchmark result.

For binary paired correctness rows, run:

```bash
python scripts/paired_statistics.py full-paired.jsonl --output paired-statistics.json \
  --baseline-field baseline_correct --ours-field ours_correct --category-field category \
  --seed 20260806 --bootstrap-iterations 10000
```

Report accuracy delta, paired bootstrap 95% CI, the 2x2 table, exact McNemar p-value, parse success, and category results. Do not attach McNemar or accuracy semantics to non-binary metrics.

## Pair And Select Samples

Join baseline and ours by the stable benchmark sample ID. Confirm identical question, options, reference, media locator, prompt, frame policy, and generation configuration for every pair.

Select up to 100 qualitative rows from complete paired predictions:

```bash
python scripts/select_paired_samples.py full-paired.jsonl sample_manifest.jsonl \
  --representative 50 --targeted 50 --seed 20260806
```

Use 50 proportionally stratified representative rows and 50 targeted rows, prioritizing ours regressions, then ours improvements, then both-wrong rows. Continue category stratification within each outcome. Mark every row `representative` or `targeted_badcase`. If fewer than 100 valid pairs exist, report the shortfall instead of duplicating rows.

## Preserve And Display Complete JSON

Read [references/raw-json-contract.md](references/raw-json-contract.md) before archiving selected rows or building HTML. For every displayed pair, preserve and display three complete records: the source benchmark sample, the Baseline prediction record, and the Ours prediction record.

- For JSON/JSONL inputs, archive the exact UTF-8 record payload without reserialization, field reordering, pretty-printing, normalization, or truncation.
- Put the three artifact descriptors under `raw_json` in `paired_samples.jsonl`, including archive path, SHA-256, origin, and encoding.
- Render all three complete texts inside the same sample card. The source record belongs in the common section; prediction records belong in their respective Baseline/Ours columns.
- Use HTML escaping only. Keep each block in the DOM, visible by default, and scrollable when long; never replace it with an excerpt or only a download link.
- If a source is not JSON, preserve the native record and label the complete JSON projection `derived_lossless_projection`; never call generated JSON source-exact.

## Recover Media And Diagnose Errors

Decode every selected image or video. For videos, generate 6-8 uniformly sampled keyframes, retain frame indices/timestamps, and copy or extract the complete source clip into `assets/videos/`. Every available video sample must expose the complete local file through an in-page `<video controls>` player with the contact sheet as poster; a contact sheet alone is incomplete. Make the primary video action play in the current sample instead of navigating to a raw file or triggering a browser download. Probe the container and codec before claiming browser playback. For codecs unsupported by Chromium, generate an H.264 browser proxy when practical while retaining the untouched original locally. If a proxy is not feasible, provide a copy-local-path action for an external player, mark browser playback as unavailable, and never describe file existence as successful playback. Mark missing media, decode failures, empty-video fallback, and frame-sampling failures explicitly; never present repeated fallback frames as valid temporal evidence.

For every displayed sample that contains ground-truth bounding boxes or points, make a ground-truth overlay on a copy of the original-resolution image or the exact referenced video frame the primary visual. Draw every declared box and point, visibly label the layer as `GT`, and keep the clean source media available separately. Resolve `xyxy`/`xywh`/`cxcywh`, pixel/normalized/model-scale coordinates, `image_id`, frame index, timestamp, and any orientation transform from traceable metadata before rendering. Transform against decoded source dimensions, not CSS display dimensions; never guess coordinate semantics, silently clip invalid geometry, or render an annotation on a convenient but unrelated keyframe. Record the source annotation, coordinate convention, target media/frame, overlay path/hash, and declared/rendered primitive counts in the paired row or manifest. If any mapping is unresolved or the counts differ, mark `ground_truth_overlay_unresolved` and do not count that sample as visually validated; coordinate text or a clean image alone is not an acceptable fallback.

Keep raw model output unchanged and HTML-escape it. This response view does not replace the mandatory complete prediction-record JSON block. Assign only evidence-supported diagnostic tags. Use `unknown` when pixels, frames, references, or model reasoning do not support a narrower label. Read [references/media-and-browser-validation.md](references/media-and-browser-validation.md) for the media and browser acceptance checks.

## Build The Delivery

Create:

```text
OUTPUT/
  index.html
  self-val/index.html
  external-benchmarks/index.html
  benchmark-registry.json
  overall-summary.json
  source-readonly-evidence.json
  audit/
  GROUP/BENCHMARK/
    report.html
    summary.json
    inference_config.json
    paired_samples.jsonl
    sample_manifest.jsonl
    badcases.csv
    validation.json
    raw-json/
    assets/
      videos/
```

Generate one status page instead of a paired report for unresolved sample coverage. In every report, keep Baseline left and Ours right on desktop, stacked in that order on mobile. Show the selected headline-result policy above the gallery and make the sampled nature of displayed rows explicit.

The root report must include:

- an evidence-tier explanation;
- the active headline source, either `reproducible_primary` or `user_specified_primary`;
- under `user_specified_primary`, only the supplied result table in headline and aggregate sections;
- contamination and format-risk warnings when the requested presentation policy includes them; otherwise retain those fields only in machine-readable audit artifacts;
- partial/shared/smoke coverage ratios;
- improvements, regressions, uncertainty, and unresolved items;
- no cross-benchmark arithmetic mean over heterogeneous metrics.

## Validate And Hand Off

Run structural validation:

```bash
python scripts/validate_delivery.py OUTPUT --expected-benchmarks N --expected-reports M --browser-check
```

Then use Playwright with `1440x900` and `390x844` viewports. Require no broken media or links, external requests, console/page errors, horizontal page overflow, clipped controls, overlapping model panels, or unescaped model-output markup. For every sample, assert exactly one `sample`, one `baseline_prediction`, and one `ours_prediction` raw-JSON block; compare each block's `textContent` with its archived UTF-8 payload and require it to be visible by default. Exercise every filter and keyword search, save screenshots, and visually inspect representative desktop/mobile pages. Click the in-page video action and require that the URL remains unchanged, playback starts, `currentTime` advances, and only the selected player is active. Require metadata decoding for browser-compatible sources or proxies; record unsupported source codecs as an explicit diagnostic and validate the complete untouched original independently.

Finish only when source before/after fingerprints match and every registry row has a report or explicit status page. Report the absolute output path, counts, displayed primary metrics, regressions, incomplete items, scorer limitations, requested risk disclosures, validation files, and browser result. Never claim official parity, read-only integrity, media decoding, train/eval decontamination, complete video recovery, or browser validation unless it was actually checked.
