# Scoring And Paired Statistics

## Contents

1. Metric hierarchy
2. Paired binary statistics
3. Non-binary metrics
4. Scorer parity
5. Interpretation

## Metric Hierarchy

Use this precedence:

1. Official benchmark scorer on the official split.
2. Faithful local reproduction with scorer/version parity evidence.
3. Local conversion metric on transformed data.
4. Diagnostic heuristic or lexical metric.

Never relabel levels 2-4 as official without evidence. A benchmark may publish an official judge that is unavailable locally; in that case, report the local diagnostic and state that it is not the benchmark score.

Metric authority and population compatibility are independent. An official scalar from one partial population cannot be subtracted from an official scalar on another population. Recompute on a verified intersection or report the two values without a delta.

For exact-choice VQA, parse the requested option and score exact correctness after the benchmark's normalization. For open answers, use the benchmark's answer normalization and consensus policy. For grounding, use the official box/point coordinate convention and threshold. For captioning, use corpus-level scorers, not an average of ad hoc per-row similarities.

## Paired Binary Statistics

For paired correctness variables `B_i` and `O_i`:

```text
baseline accuracy = sum(B_i) / n
ours accuracy     = sum(O_i) / n
delta             = sum(O_i - B_i) / n
```

Build the 2x2 table:

| Outcome | Condition |
|---|---|
| both correct | `B=1, O=1` |
| ours only | `B=0, O=1` |
| baseline only | `B=1, O=0` |
| both wrong | `B=0, O=0` |

Bootstrap paired rows, not model sides independently. Resample row indices with replacement, compute the paired delta per replicate, and report the 2.5th and 97.5th percentiles with the seed and iteration count.

McNemar's exact test uses only discordant counts `b=baseline_only` and `c=ours_only`. Under the null, `min(b,c)` follows the lower tail of `Binomial(b+c, 0.5)`; use the two-sided p-value capped at 1. Report the test even when the direction is small, but do not describe `p>=0.05` as proof of equivalence.

Category accuracy is diagnostic unless the benchmark defines category aggregation as an official score. Include sample counts so small strata are visible.

## Non-Binary Metrics

For graded per-row scores, pair and bootstrap score differences when the scorer legitimately produces additive row scores. Do not fabricate per-row CIDEr, corpus BLEU, or other corpus-coupled scores merely to obtain a paired CI. Re-run the corpus scorer within bootstrap samples only when the implementation and compute budget support it.

Do not use McNemar for token-F1, IoU values, CIDEr, LLM-judge ratings, or macro category averages. If converting a grounding score to a binary `IoU >= threshold` hit, name that exact threshold and coordinate convention.

## Scorer Parity

Preserve and compare:

- official scorer repository/package version and commit;
- exact scorer input artifact;
- official output artifact;
- expected total and category counts;
- macro/micro aggregation and scale;
- answer normalization, language branch, and invalid-output behavior.

Spreadsheet-based scorers require special care. The workbook consumed by the scorer may differ from raw JSONL predictions because of truncation, normalization, manual columns, or resumed runs. Compare row IDs and raw outputs between both artifacts, quantify mismatches, and score the actual official input while displaying the unmodified raw model output. Document this split explicitly.

## Interpretation

Lead with external, official, full-set evidence. Use self-built validation to diagnose training-distribution fit, not as an external generalization claim. Large self-val gains paired with external regressions can indicate task-format adaptation, contamination, over-specialization, decoding differences, or scorer mismatch; benchmark results alone do not prove the cause.

Separate observed leakage evidence from user-provided leakage claims and unknown audit status. Any confirmed or user-declared media leakage must be prominent enough that a reader cannot mistake the score for clean holdout generalization.

Treat the 100-row gallery as qualitative evidence. Its purpose is to reveal mechanisms and bad cases, not estimate the leaderboard score.
