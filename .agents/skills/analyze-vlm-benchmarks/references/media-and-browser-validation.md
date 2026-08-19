# Media And Browser Validation

## Contents

1. Media recovery
2. Diagnostic labels
3. HTML safety
4. Browser acceptance

## Media Recovery

Resolve media from the canonical evaluation record before trying historical or guessed paths. Record the original locator and the resolved path.

For images:

- verify existence and non-zero size;
- decode pixels;
- preserve aspect ratio;
- copy a report-local preview and record its SHA256;
- retain overlays separately from the unmodified image.

For ground-truth spatial annotations:

- render every ground-truth bbox and point directly on a copy of the original-resolution image or exact referenced video frame, and use that overlay as the sample's primary visual;
- keep the unmodified source separately and label the overlay visibly as `GT` so it cannot be confused with model predictions;
- resolve bbox encoding, coordinate space, image identity, frame/timestamp, and orientation before rendering; scale from decoded source dimensions rather than rendered HTML dimensions;
- use distinct, legible marks for overlapping objects and a visible dot/crosshair for points; preserve labels or stable indices when present;
- store the annotation source, coordinate convention, target media/frame, overlay path/hash, and declared/rendered bbox and point counts;
- reject silent clipping, guessed coordinate semantics, unrelated keyframes, and coordinate-only text as substitutes;
- mark unresolved mapping or primitive-count mismatch as `ground_truth_overlay_unresolved` and fail visual acceptance for that sample.

For videos:

- inspect the container, duration, FPS, frame count, and decodability;
- record the video codec and test browser support instead of inferring support from the `.mp4` extension;
- sample 6-8 uniformly spaced timestamps across the valid duration;
- record frame index and timestamp on the contact sheet;
- copy or extract the complete source video into report-local `assets/videos/` without clipping its duration;
- render a local `<video controls>` element with the contact sheet as poster and make the primary action play that element in the current page;
- do not make users navigate to a raw video URL or depend on a browser download action for normal inspection;
- for browser-incompatible codecs such as `mp4v`, create an H.264 browser proxy when practical and retain the untouched original locally;
- when transcoding is not feasible, retain the poster and original, provide a copy-local-path action for VLC/system playback, mark browser playback unavailable, and do not claim metadata/playback validation passed;
- store `video_asset` and `video_size_bytes` in both `paired_samples.jsonl` and `sample_manifest.jsonl`;
- do not embed full videos as base64 in HTML.

Keyframes are navigation aids, not a replacement for the complete video. Deduplicate repeated source videos by reusing one relative asset. If a source video is genuinely unavailable, omit the player, mark the media status, and keep the failure visible.

If only one frame decodes, do not repeat it eight times without warning. Mark `empty_video_fallback`, `decode_failure`, or `frame_sampling_failure` and highlight the sample. A visible frame does not prove temporal evidence was available to inference.

## Diagnostic Labels

Use labels such as:

```text
visual_perception, spatial_reasoning, commonsense, temporal_reasoning,
action_planning, option_confusion, hallucination, output_format,
truncation, media_load_failure, frame_sampling_failure, unknown
```

Assign a label only from observable evidence: reference, options, media, parsed answer, raw output, and logged media state. A wrong answer alone does not prove hallucination or perception failure. Mark automatic labels as diagnostic in the report.

## HTML Safety

- Set `<meta charset="utf-8">` and `<html lang="zh-CN">`.
- Use only local CSS, JS, images, and videos.
- Convert all model text with an HTML escaping function before template insertion.
- Render raw output inside plain `<pre>` text with no child markup.
- Reject absolute paths, `..` traversal, remote URLs, and inline event handlers in model-controlled fields.
- Keep Baseline consistently left/top and Ours right/bottom.
- Do not rely on color alone; print explicit outcome and correctness labels.

## Browser Acceptance

Use Chromium/Playwright at `1440x900` and `390x844` for every report type and navigation/status page. Check:

- page and heading are Chinese UTF-8;
- expected sample and model-panel counts;
- every image has non-zero natural dimensions;
- every sample with ground-truth bbox/point data loads a non-empty `GT` overlay, links the clean source separately, and has matching declared/rendered primitive counts;
- every available video row has a local player, a complete non-empty asset, and an in-page playback action;
- clicking the playback action leaves the report URL unchanged, starts the selected video, advances `currentTime`, and pauses other players;
- browser-compatible videos or proxies load non-zero metadata dimensions and duration;
- browser-incompatible originals have an explicit codec diagnostic and are not reported as successfully playable;
- no failed or external requests, console errors, or page exceptions;
- `documentElement.scrollWidth <= clientWidth + 1`;
- desktop model columns share a top edge and do not overlap;
- mobile Baseline precedes Ours vertically and panels do not overlap;
- controls, tags, IDs, long words, and raw outputs are not clipped;
- outcome, category, error, cohort, media, and keyword filters update counts correctly;
- report navigation links resolve, while normal video inspection does not navigate to a raw media URL;
- raw-output `<pre>` elements contain text, not injected child elements.

When a page is closed or a test deliberately resets video sources, Chromium may report `net::ERR_ABORTED` for pending metadata requests. Ignore only requests whose resource type is `media`, whose local path is under `assets/videos/`, and whose exact error is `net::ERR_ABORTED`. Never apply that exception to images, HTML, scripts, styles, or other failure codes. Validate original video existence and size independently even when this narrow exception is used.

Save viewport and first-sample screenshots. Automated geometry checks are necessary but not sufficient; visually inspect at least the overall index, one image benchmark, one grounding report, and one video report on both viewport sizes.
