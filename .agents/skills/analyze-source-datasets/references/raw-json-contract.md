# Verbatim Sample JSON Contract

Apply this contract to every accepted sample in a dataset inspection report.

## Preserve The Payload

For JSONL, define the record payload as the exact bytes between record separators. Exclude only the trailing `LF` or `CRLF`; preserve all other whitespace, key order, numeric lexemes, escape sequences, and Unicode bytes. For a standalone JSON object, preserve the complete file bytes. Do not recreate source-exact text with `json.dumps` or another serializer.

If the source is not JSON, archive the native source record and create a complete deterministic JSON projection containing every field. Mark it `derived_lossless_projection`, document reversible encodings for non-JSON values, and never describe it as unchanged source JSON.

## Manifest Schema

Every accepted row in `sampling-manifest.jsonl` contains:

```json
{
  "sample_id": "example-001",
  "raw_json": {
    "archive_path": "records/example-001.raw.json",
    "sha256": "64 lowercase hex characters",
    "origin": "source_exact",
    "encoding": "utf-8"
  }
}
```

The path is relative to the dataset archive directory and must not escape it. The file must decode as UTF-8, parse as one complete JSON value, and match the declared SHA-256.

## HTML Contract

Every `.sample` card contains one block:

```html
<details class="raw-json-panel" open>
  <summary>Complete sample JSON</summary>
  <pre class="raw-json raw-sample-json"
       data-sample-id="example-001"
       data-raw-json-role="sample"
       data-archive-path="records/example-001.raw.json"
       data-sha256="..."
       data-origin="source_exact">EXACT HTML-ESCAPED PAYLOAD</pre>
</details>
```

HTML escaping is the only allowed rendering transform. The parsed block's `textContent` must equal the decoded archive exactly. Keep it visible by default, either outside `<details>` or inside `<details open>`. A scrollable `max-height` is allowed; omission, ellipsis, substring previews, lazy text injection, pretty-printing, field reordering, and link-only presentation are not.
