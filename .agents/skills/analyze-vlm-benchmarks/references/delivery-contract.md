# Delivery Contract

## Contents

1. Source and model integrity
2. Benchmark registry
3. Report artifacts
4. Status rules
5. Completion gates

## Source And Model Integrity

Treat models, datasets, prompts, historical predictions, scorer outputs, and completed results as immutable. Put generated previews, caches, temporary environments, screenshots, and reports outside those roots.

Record a before/after fingerprint for every declared result collection. A useful scalable fingerprint hashes sorted tuples of relative path, byte size, and nanosecond mtime. This proves the inventoried result tree was not changed; state clearly that it does not cover unrelated paths. Use content hashes for model identity and selected media integrity.

A reusable run must match:

- model content identity;
- dataset release and split;
- exact paired sample population;
- prompt/template and answer-format request;
- image preprocessing or video frame policy;
- seed, decoding, max tokens, dtype, attention implementation, and batch-independent generation settings;
- scorer code/version and metric scope;
- zero failed samples or an explicitly documented valid subset.

Path equality alone is insufficient. Conversely, different checkpoint paths may be equivalent when every model shard hash matches.

## Benchmark Registry

Store one JSON object per canonical benchmark with at least:

```json
{
  "benchmark": "stable-slug",
  "display_name": "Human name",
  "group": "self-val or external-benchmarks",
  "task_type": "VQA, DESCRIPTION, GROUNDING, or mixed",
  "status": "complete, partial, unresolved, or excluded",
  "headline_source": "reproducible_primary or user_specified_primary",
  "evidence_tier": "official_full, reproducible_full, diagnostic_full, shared_subset, smoke, historical_verified, historical_unverified, or unresolved",
  "data_version": "observed release or local conversion",
  "split": "observed split",
  "metric_scope": "official, local conversion, or diagnostic identifier",
  "primary_metric": "metric identifier",
  "declared_population_n": 0,
  "scored_population_n": 0,
  "paired_n": 0,
  "displayed_n": 0,
  "presented_metrics": [],
  "computed_aggregates_rendered": false,
  "baseline_source": "prediction/result locator",
  "ours_source": "prediction/result locator",
  "scorer_source": "code/result/version locator",
  "data_risk": {
    "media_leakage": "observed, user_provided, absent, or unknown",
    "qa_leakage": "observed, user_provided, absent, or unknown",
    "format_defect": "description or null"
  },
  "limitations": []
}
```

Include every discovered result directory in an inventory appendix, but choose only one canonical comparable run per benchmark. Explain why duplicate, debug, partial, stale, or incompatible runs were not selected.

## Report Artifacts

Each paired benchmark directory must contain:

- `report.html`: UTF-8, offline HTML/CSS/JS with relative media;
- `summary.json`: presented primary results, presentation policy, optional audit metrics, displayed counts, and limitations;
- `inference_config.json`: model identity, prompt, media policy, generation config, scorer provenance, and source paths;
- `paired_samples.jsonl`: one row per displayed sample containing both raw outputs and all displayed fields;
- `sample_manifest.jsonl`: the same unique IDs and order, seed, cohort, stratum, and source locators;
- `badcases.csv`: error rows and evidence-supported diagnostic tags;
- `validation.json`: machine-readable structural result;
- `raw-json/`: exact source-sample, Baseline prediction, and Ours prediction JSON payloads for every displayed pair;
- `assets/`: copied previews, keyframes, and contact sheets;
- `assets/videos/`: complete report-local source videos for every available displayed video sample.

Minimum paired row fields:

```text
sample_id, category, question, options, reference_answer,
media_asset, media_status, selection_cohort,
baseline_response, baseline_parsed, baseline_correct or baseline_score,
baseline_token_count, baseline_error_tags,
ours_response, ours_parsed, ours_correct or ours_score,
ours_token_count, ours_error_tags
```

Keep raw responses verbatim. Store derived labels separately and identify their method.

Every paired row must also contain a `raw_json` object with `sample`, `baseline_prediction`, and `ours_prediction` descriptors. Each descriptor contains `archive_path`, `sha256`, `origin`, and `encoding`. Follow [raw-json-contract.md](raw-json-contract.md): archive source-exact JSON/JSONL payloads without reserialization and render all three complete texts, visible by default, in the same HTML sample card.

For rows with ground-truth bbox or point annotations, also store `ground_truth_overlay_asset`, `ground_truth_overlay_sha256`, source annotation locator, coordinate convention, target image/frame identity, declared/rendered primitive counts, and `ground_truth_overlay_status`. The overlay must be a separate derived copy of the original-resolution image or exact referenced frame, visibly labeled `GT`; retain the clean source independently.

## Status Rules

- `complete`: full declared population is valid and the required scorer completed.
- `partial`: a precisely identified subset is valid; denominators and missing coverage are prominent.
- `unresolved`: no meaningful paired comparison can be formed.
- `excluded`: outside scope or deliberately rejected with a reason.

Unequal partial populations may still produce a `shared_subset` report when the ID intersection is verified, immutable inputs match, and both sides are rescored on exactly that intersection. Under `reproducible_primary`, keep incompatible aggregates in a separate audit block. Under `user_specified_primary`, do not render shared-subset or smoke aggregates in headline or aggregate sections. A technically complete scorer run on a small smoke file remains `partial` at the benchmark level.

Create a status page when one side is absent, no verified intersection exists, configs differ materially, source media is unavailable, or scorer parity cannot be established. Do not fill missing values with another run.

## Completion Gates

Require all of the following:

- every registry row links to a report or status page;
- IDs are unique and paired in identical order;
- every paired row has three valid raw-JSON descriptors; archived payload hashes match, each payload parses as complete JSON, and HTML text matches the archive exactly without truncation or reserialization;
- the manifest and report sample counts agree;
- displayed media exists, decodes, and uses relative paths;
- every row with ground-truth bbox/point data has a decoded GT overlay on the recorded original image/frame, a separate clean source, and matching declared/rendered primitive counts;
- HTML contains no external CDN or executable model output;
- official scores trace to official scorer inputs and outputs;
- sampled scores are never presented as full-set scores;
- the active headline-result source is explicit and qualitative samples are structurally separate from it;
- `user_specified_primary` pages do not visibly substitute recomputed, sampled, shared-subset, or smoke aggregates;
- contamination, leakage, and format defects follow the explicit presentation policy while remaining preserved in machine-readable audit data;
- every available video row has keyframes, a complete local video player, and an in-page playback action that does not navigate away; smoke/shared-subset coverage remains visible;
- desktop and mobile browser checks pass;
- source fingerprints match;
- unresolved and partial data issues remain visible.
