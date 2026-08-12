# Verbatim Raw JSON Contract

Apply this contract to every paired sample rendered in a benchmark report.

## Preserve Record Payloads

For JSONL, define the record payload as the exact bytes between record separators. Exclude only the trailing `LF` or `CRLF`; preserve all other whitespace, key order, numeric lexemes, escape sequences, and Unicode bytes. For a standalone JSON object, preserve the complete file bytes. Do not regenerate source-exact text with `json.dumps` or another serializer.

Archive three UTF-8 payloads per displayed pair:

- `sample`: the benchmark input/reference record used for both models;
- `baseline_prediction`: the complete persisted Baseline prediction row;
- `ours_prediction`: the complete persisted Ours prediction row.

If the input is not JSON, archive the native record separately and create a complete deterministic JSON projection containing every source field. Set `origin` to `derived_lossless_projection`; do not describe it as unchanged source JSON. Unsupported values such as bytes must use a documented reversible representation.

## Paired Row Schema

Store descriptors in `paired_samples.jsonl`:

```json
{
  "sample_id": "example-001",
  "raw_json": {
    "sample": {
      "archive_path": "raw-json/example-001.sample.json",
      "sha256": "64 lowercase hex characters",
      "origin": "source_exact",
      "encoding": "utf-8"
    },
    "baseline_prediction": {
      "archive_path": "raw-json/example-001.baseline.json",
      "sha256": "64 lowercase hex characters",
      "origin": "source_exact",
      "encoding": "utf-8"
    },
    "ours_prediction": {
      "archive_path": "raw-json/example-001.ours.json",
      "sha256": "64 lowercase hex characters",
      "origin": "source_exact",
      "encoding": "utf-8"
    }
  }
}
```

Paths are relative to the benchmark report directory and must not escape it. Each archived payload must decode as UTF-8, parse as one complete JSON value, and match its SHA-256.

## HTML Contract

Render these blocks inside every `.sample` card:

```html
<details class="raw-json-panel" open>
  <summary>Complete source sample JSON</summary>
  <pre class="raw-json raw-sample-json"
       data-sample-id="example-001"
       data-raw-json-role="sample"
       data-archive-path="raw-json/example-001.sample.json"
       data-sha256="..."
       data-origin="source_exact">EXACT HTML-ESCAPED PAYLOAD</pre>
</details>
```

Use roles `sample`, `baseline_prediction`, and `ours_prediction`. Put the sample block in the common context and each prediction block in its matching model column. A scrollable `max-height` is allowed; omission, ellipsis, substring previews, lazy text injection, field reordering, and a link-only presentation are not.

HTML escaping is the only allowed rendering transform. After parsing the page, each block's `textContent` must equal the decoded archived payload exactly. Keep the block visible by default, either outside `<details>` or inside `<details open>`.
