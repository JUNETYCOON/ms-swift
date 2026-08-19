# Molmo2, PixMo, and SpatialVLM conversion

`prepare_molmo_pixmo_spatial_swift.py` converts these local Hugging Face
Parquet exports into explicit ms-swift `train.jsonl` and `val.jsonl` files:

- `Molmo2-VideoCapQA`
- `Molmo2-VideoPoint`
- `Molmo2-VideoSubtitleQA`
- `Molmo2-VideoTrack`
- `pixmo-cap`
- `pixmo-points`
- `spatialvlm`

The defaults are specific to the stage-1 server:

```text
input:  /mnt/luojunkun/stage1/dataset/<dataset>
output: /mnt/luojunkun/stage1/dataset_ms-swift/<dataset>
```

## Split guarantee

The converter does not use ms-swift's row-level `split_dataset_ratio`. It hashes
`seed + canonical_media_key` with SHA-256 and writes the split explicitly. The
same image/video therefore cannot appear in both files, even when its QA rows
are in different Parquet shards.

The media keys are:

| Dataset | Group key |
| --- | --- |
| VideoCapQA / VideoSubtitleQA | YouTube `video_id` |
| VideoPoint | `(video_source, video_id)`; YouTube IDs share the same namespace as the QA datasets |
| VideoTrack | `(video_dataset, video)` |
| pixmo-cap | canonical URL with query and fragment removed |
| pixmo-points | SHA-256/URL connected component, covering both same-SHA/different-URL and same-URL/different-SHA anomalies |
| spatialvlm | ordered SHA-256 hashes of the embedded image bytes |

Every conversion report contains `unique_media.cross_split_leakage: 0`, and the
root summary performs the same audit across all selected datasets. A conflict
aborts the merge. Overwrites use backups and restore the previous complete file
set if installation of any new output fails.

## Quick check

Run a small conversion first. This exercises real Parquet schemas, parallel
workers, image extraction, placeholder validation, and split validation:

```bash
python /mnt/workspace/stage1/scripts/data-process/prepare_molmo_pixmo_spatial_swift.py \
  --datasets all \
  --num-workers 8 \
  --max-source-rows 200 \
  --allow-unverified-remote-media \
  --allow-empty-dataset \
  --max-reject-ratio 1 \
  --output-root /mnt/luojunkun/stage1/dataset_ms-swift-smoke
```

`--max-source-rows` is distributed across every authoritative source file, so
the check covers LongCapQA and all 16 VideoTrack sources instead of sampling
only the first Parquet.

Convert the six datasets whose annotations have at least some bundled/direct
media. This command explicitly allows unresolved remote references and partial
sources; inspect every report before training:

```bash
python /mnt/workspace/stage1/scripts/data-process/prepare_molmo_pixmo_spatial_swift.py \
  --datasets Molmo2-VideoCapQA Molmo2-VideoPoint Molmo2-VideoSubtitleQA \
             pixmo-cap pixmo-points spatialvlm \
  --num-workers 24 \
  --val-ratio 0.05 \
  --seed 42 \
  --extract-generated-videos \
  --allow-unverified-remote-media \
  --max-reject-ratio 1
```

Omit the two `allow` options for a readiness-enforcing run. In that mode,
remote media references, an empty result, or any source above the default 25%
reject threshold aborts without installing final JSONL files. Failed fragments
are retained in the reported hidden directory for diagnosis.

Use `--overwrite` only when replacing an earlier `train.jsonl`, `val.jsonl`,
`rejected.jsonl`, and report. Existing unrelated datasets below the output root
are never removed.

## Output

Each dataset directory contains:

```text
<dataset>/
  train.jsonl
  val.jsonl
  rejected.jsonl
  media_groups.tsv
  conversion_report.json
  images/                 # image datasets
  videos/                 # video datasets
```

`spatialvlm` images are extracted from Parquet bytes, deduplicated by content
SHA-256, and referenced by absolute paths. With
`--allow-unverified-remote-media`, PixMo and Molmo rows retain HTTP(S) resource
URLs unless an override index materializes them. Both forms are accepted by
ms-swift, although the report marks URL-based output as
`unverified_remote_references` rather than training-ready.

Local references are checked for absolute path and file existence, not decoded
or probed. Their report state is `local_files_exist_not_decoded`; verify image
decoding and video stream/frame metadata before starting SFT. A dataset with no
converted media is reported as `no_converted_media`.

Rejected rows contain a compact source identity and error, never the full large
annotation. Common expected reasons are unresolved licensed video media and
missing LongCapQA URL mappings.

## Verified full server run

The non-VideoTrack datasets were run on `2026-08-03` with 24 workers, seed 42,
a 5% validation ratio, generated-video extraction, and the explicit
diagnostic-only remote/partial-output flags. VideoTrack was independently
materialized and converted on `2026-08-11`. The installed output is
`/mnt/luojunkun/stage1/dataset_ms-swift`:

| Dataset | Train | Val | Rejected source rows |
| --- | ---: | ---: | ---: |
| Molmo2-VideoCapQA | 905,548 | 47,586 | 49,485 |
| Molmo2-VideoPoint | 603,085 | 31,525 | 23,730 |
| Molmo2-VideoSubtitleQA | 444,963 | 23,539 | 0 |
| Molmo2-VideoTrack | 23,620 | 1,014 | 40 |
| pixmo-cap | 681,313 | 35,729 | 0 |
| pixmo-points | 2,257,510 | 118,659 | 53 |
| spatialvlm | 82,892 | 4,584 | 0 |

The seven non-empty datasets contain 5,261,567 records. The VideoTrack
converter audit covers 1,319 canonical video lineages with zero train/val
conflicts.
Generated VideoPoint extraction produced 15,509 files; all 15,204 generated
source rows use local absolute paths. SpatialVLM produced 27,328 deduplicated
local images. The 53 rejected PixMo Points rows have empty labels.

These counts describe a schema-valid partial export, not a fully media-ready
training corpus. The Molmo requester-pays URLs remain unreadable without an
authorized GCP quota project, 49,485 LongCapQA rows have no mapping, and 23,730
VideoPoint rows have no available source media. VideoTrack is media-ready only
for the five sources documented below; the other 11 sources remain skipped.
PixMo media remains URL-based. Each report records these states and rejection
counts.

## Dataset mapping

### VideoCapQA

`CapQA` rows become one `<video>` question/answer sample. Each LongCapQA
`qa_list` is expanded into independent samples. The category and negative
answers are not training targets. The authoritative CapQA and LongCapQA files
are both read, and their overlapping video IDs receive the same split.

The bundled mapping covers all 190,631 CapQA videos but only 511 of 49,996
LongCapQA videos. The default per-source reject threshold therefore fails this
dataset rather than silently calling the approximately 1% LongCap conversion
complete. Supply the missing files with `--video-media-index`, or use
`--max-reject-ratio 1` only when an explicitly partial annotation export is
acceptable.

### VideoSubtitleQA

The default prompt contains the video, timestamped transcript, and question.
The upstream task explicitly requires both visual content and the audio
transcript. `--exclude-subtitles` is an intentional opt-out that changes the
task to video-only QA; do not use it for a faithful SubtitleQA conversion.

### VideoPoint

Only `data/train-00000-of-00001.parquet` is read. The eight category Parquets
are duplicate partitions of that table and are intentionally excluded.

Source points are percentages (`0..100`), not pixels. Targets retain raw video
timestamps/frame numbers and use textual points on a `0..1000` grid. They do
not use ms-swift `objects`: the current template implementation normalizes
`objects` only against `images` and has no frame/timestamp binding for videos.
Empty point rows are retained as negative examples.

Generated videos can be extracted from the local tar archives:

```bash
python /mnt/workspace/stage1/scripts/data-process/prepare_molmo_pixmo_spatial_swift.py \
  --datasets Molmo2-VideoPoint \
  --extract-generated-videos \
  --num-workers 3
```

MammalNet media is not bundled and unresolved rows are recorded as rejected.
Use `--video-media-index` to supply downloaded MammalNet/generated/YouTube
files and missing LongCapQA media. It is a JSON object from `video_id` to an
absolute local path (or a direct URL); local relative paths resolve against the
index file.

### VideoTrack

The Parquets are authoritative; colocated `point_tracks.json` files duplicate
them. Actual videos are not bundled and several of the 16 upstream sources have
separate license/download requirements. Supply a JSON resolver after acquiring
them legally:

```json
{
  "dataset_name::clip_000_239": {
    "source_video_id": "dataset_name::source_video_001",
    "lineage_key": "dataset_name::source_video_001",
    "windows": [
      {
        "mode": "cropped",
        "path": "/absolute/media/clip_000_239-window-000-119.mp4",
        "source_start_frame": 0,
        "source_end_frame": 119
      },
      {
        "mode": "cropped",
        "path": "/absolute/media/clip_000_239-window-120-239.mp4",
        "source_start_frame": 120,
        "source_end_frame": 239
      }
    ]
  }
}
```

The example annotation spans source frames `0..239`; both files together cover
that exact inclusive range.

Materialized indexes should declare `source_video_id` and `lineage_key`
together. The converter verifies that both identify the annotation's `video`
before using `lineage_key` for the split key. Legacy entries without either
field use the dataset/video fallback; `mose` and `mosev2` both fall back to
`mose-family::<video>`.

For the `2026-08-11` partial training run, real media was available and
materialized for `dancetrack`, `soccernet`, `mose`, `mosev2`, and `vipseg`.
`APTv2`, `animaltrack`, `bdd100k`, `bft`, `mot2020`, `personpath22`, `sav`,
`seadrones`, `sportsmot`, `teamtrack`, and `uavdt` were skipped because no
reliable local media could be joined. `Molmo2-VideoCapQA`,
`Molmo2-VideoPoint`, and `Molmo2-VideoSubtitleQA` were not read or modified by
this run.

```bash
python /mnt/workspace/stage1/scripts/data-process/materialize_molmo2_videotrack_media.py \
  --scratch-dir /mnt/workspace/stage1/.videotrack-scratch \
  --workers 8 \
  --max-inflight-videos 16 \
  --ffmpeg-threads 2 \
  --max-scratch-bytes 16106127360

python /mnt/workspace/stage1/scripts/data-process/prepare_molmo_pixmo_spatial_swift.py \
  --datasets Molmo2-VideoTrack \
  --input-root /mnt/luojunkun/stage1/dataset \
  --output-root /mnt/luojunkun/stage1/dataset_ms-swift \
  --video-track-sources dancetrack mose mosev2 soccernet vipseg \
  --video-track-index /mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/video_track_media.json \
  --num-workers 12 \
  --val-ratio 0.05 \
  --seed 42 \
  --max-reject-ratio 0.01 \
  --overwrite
```

For a partial media inventory, add `--video-track-sources mose mosev2` to read
only those case-insensitive `data/<source>` Parquet groups. Unknown source names
fail with the available source list, and the normalized selection is recorded
in `conversion_report.json`.

String-only index values and full source videos are rejected deliberately.
Track points are normalized with the source width/height and retain object ID,
frame, and timestamp in assistant text. A resolver value may be one cropped
window, a list of windows, or an object containing `windows`. The declared
inclusive source-frame ranges must form a gap-free, non-overlapping partition
of the annotation's full `start_frame..end_frame` range. Each media file must be
the real clip for its declared window and may contain at most 128 frames by
default; use `--max-track-window-frames` to change that bound. Output frame
numbers and timestamps are local to each window and start at zero. Every window
from the same original video retains the same video-level split key. Rows with
unresolved, oversized, gapped, overlapping, or mismatched windows are rejected.

The verified five-source result is:

| Source | Source rows | Retained rows | Train records | Val records | Rejected rows | Clips | Unique MP4 windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dancetrack | 3,735 | 3,723 | 8,095 | 391 | 12 | 704 | 1,225 |
| soccernet | 4,420 | 4,392 | 8,088 | 259 | 28 | 610 | 978 |
| mose | 880 | 880 | 964 | 53 | 0 | 337 | 390 |
| mosev2 | 1,168 | 1,168 | 1,258 | 60 | 0 | 463 | 527 |
| vipseg | 5,466 | 5,466 | 5,215 | 251 | 0 | 675 | 675 |
| **Total** | **15,669** | **15,629** | **23,620** | **1,014** | **40** | **2,789** | **3,795** |

The 40 rejected rows are explicit cleaning failures: 26 have no point tracks
and 14 contain visible points outside the declared source dimensions. One
source row may expand into multiple SFT records when its frame range crosses a
128-frame media window, which is why 15,629 retained source rows produce 24,634
records. The media index contains 4,046 window references to 3,795 unique MP4s.
All windows from the same canonical original video share one of 1,319 media
group assignments; `mose` and `mosev2` use the shared `mose-family` lineage.

The GT visualization audit renders points on the decoded source frames. It
contains three sampled clips per source, 45 PNG overlays, and 108 declared and
rendered points with no unresolved source. The MOSEv2 multipart SHA-256 values
for all three archive parts matched `SHA256SUMS`.

Run the strict end-to-end audit with the same source allowlist:

```bash
python /mnt/workspace/stage1/scripts/data-process/audit_molmo2_videotrack_swift.py \
  --source-root /mnt/luojunkun/stage1/dataset/Molmo2-VideoTrack \
  --output-dir /mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack \
  --video-track-sources dancetrack mose mosev2 soccernet vipseg \
  --media-workers 24 \
  --report /mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/audit_report.json
```

The `2026-08-11` audit passed with zero hard failures: all 3,795 MP4s passed
ffprobe plus first/last-frame decoding, all 15,669 source rows were accounted
for as 15,629 retained and 40 rejected rows, and both canonical-lineage and
source-video cross-split leakage counts were zero.

### Molmo2-VideoTrack norm1000 enrichment

The original VideoTrack JSONL stores track points only in assistant text. The
`convert_molmo2_videotrack_norm1000.py` adapter parses those lines, replaces
each coordinate with a `<bbox>` token, and attaches `objects.bbox` with
`bbox_type="norm1000"` while reusing the existing train/val and media mapping.
It streams both JSONL files, preserves source order, and validates placeholder
counts, coordinate ranges, media existence, and train/val video isolation.

```bash
python scripts/data-process/convert_molmo2_videotrack_norm1000.py \
  --train-input /mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/ready_train.jsonl \
  --eval-input /mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/ready_eval.jsonl \
  --output-dir /mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/norm1000 \
  --num-workers 8
```

On `2026-08-14` this produced 23,567 train rows (22,975 with points, 592
empty-track negatives) and 1,011 eval rows (1,003 with points, 8 negatives):
8,001,880 points total, zero rejections, zero placeholder/range/media errors,
and zero train/val video overlap. Negative rows keep the original
`No visible track points were annotated.` answer with an empty bbox list.

### pixmo-cap

The consolidated long caption is the assistant target. Raw transcripts are not
duplicated as extra targets. Source images are URL-only. Query parameters are
removed for grouping but the original URL is retained in `images`.

### pixmo-points

The source-provided SHA-256 is authoritative. Points are clamped to `0..100`,
divided by 100, and emitted as length-two entries in `objects.bbox` with
`bbox_type: "norm1"`. The converter enforces exact `<ref-object>`/`ref` and
`<bbox>`/`bbox` counts. Empty-point negatives keep a non-empty assistant answer
and an empty bbox list.

### spatialvlm

Embedded image bytes are extracted and content-addressed. Structured message
content is flattened in its original order. Normalized four-coordinate region
lists are replaced by `<bbox>` and stored in `objects.bbox` with
`bbox_type: "norm1"`. Independent user/assistant pairs in a source row are
expanded into single-round records. Each derived prompt receives its own
`<image>` tag and inherits the source image's split.

## Media constraints on the stage-1 server

The Molmo mapping points to a Google Cloud Storage requester-pays bucket. On
the checked server, unauthenticated HEAD/range requests return
`400 UserProjectMissing`; neither `gcloud`/`gsutil` nor a billing/quota project
is configured. `--allow-unverified-remote-media` can preserve those direct URLs
in schema-valid ms-swift JSONL, but they are not readable on the checked server.
Local video materialization requires a separately authorized GCP
project and downloader. Do not replace them with YouTube watch-page URLs,
which are not direct video resources.

PixMo contains a mixture of PixMo S3 and third-party image URLs. The checked
server can reach the S3 resources, while some third-party hosts are blocked,
expired, or return different bytes. For stable repeated training, download
reachable images separately, verify pixmo-points bytes against `image_sha256`,
and rewrite `images` to absolute local paths.

Use the resumable materializer for PixMo URL media:

```bash
python scripts/data-process/download_pixmo_media.py \
  --source-root /path/to/source-datasets \
  --output-root /path/to/converted-ms-swift-datasets \
  --state-root /path/on-a-local-filesystem/pixmo-download-state \
  --datasets pixmo-cap pixmo-points \
  --workers 32
```

`--state-root` must be on a filesystem that supports SQLite locking; do not put
it on an object-store FUSE mount. A restorable database checkpoint is copied
to each dataset directory. Downloads use temporary files and atomic renames,
decode every retained image with Pillow, and enforce the source SHA-256 values
for pixmo-points. Original `train.jsonl` and `val.jsonl` files are unchanged.
After all URLs have been attempted, the script writes `local_train.jsonl` and
`local_val.jsonl` with absolute local paths and excludes unavailable media into
the corresponding `local_*_rejected.jsonl` files. Progress and exact outcome
counts are recorded in `media_download_report.json`, `localization_report.json`,
and `media_download_manifest.jsonl`.

Inspect a running or resumed job without downloading more media:

```bash
python scripts/data-process/download_pixmo_media.py \
  --source-root /path/to/source-datasets \
  --output-root /path/to/converted-ms-swift-datasets \
  --state-root /path/on-a-local-filesystem/pixmo-download-state \
  --datasets pixmo-cap pixmo-points \
  --phase status
```

For a partial-image training run, rebuild the ordinary local splits after each
download batch. The DLC decontamination pass consumes these local files later:

```bash
python scripts/data-process/download_pixmo_media.py \
  --source-root /mnt/luojunkun/stage1/dataset \
  --output-root /mnt/luojunkun/stage1/dataset_ms-swift \
  --state-root /tmp/pixmo-download-state \
  --datasets pixmo-cap pixmo-points \
  --phase rewrite \
  --allow-incomplete-rewrite \
  --rewrite-splits train val
```

This writes `local_train.jsonl` and `local_val.jsonl`. The DLC manifest uses
`local_train.jsonl` as source and materializes `local_dlc_train.jsonl` after
removing every train row whose media occurs in any eval set. Do not use
`local_global_train.jsonl` for this run: that older entrypoint also removes
cross-dataset train media. Missing media remains listed in the matching
`local_*_rejected.jsonl`; rerunning the command after more downloads atomically
expands the local files without changing the source JSONL.

## Training

Pass the files explicitly so ms-swift does not split rows again:

```bash
swift sft \
  --model /absolute/path/to/model \
  --dataset /mnt/luojunkun/stage1/dataset_ms-swift/pixmo-points/local_dlc_train.jsonl \
  --val_dataset /mnt/luojunkun/stage1/dataset_ms-swift/pixmo-points/local_val.jsonl
```

In the checked server's default Python environment, importing the complete
`swift.dataset` stack currently fails before dataset loading because
`libhggcrt1.so` is missing. The underlying Hugging Face JSON loader successfully
loaded one streaming sample from every non-empty output. Install/activate the
server's intended accelerator runtime before invoking `swift sft`.

Grounding outputs use ms-swift's generic `objects` format and are directly
usable with the default `QWENVL_BBOX_FORMAT=legacy`. Setting
`QWENVL_BBOX_FORMAT=new` only changes placeholder rendering; it does not add the
`bbox_2d`/`label` JSON structure required by the official Qwen cookbook. A
separate output transformation is required for that cookbook representation,
and the point schema must be chosen explicitly for PixMo Points.
