# Task And Metric Taxonomy

Use the task label that matches the evaluated output, not only the source dataset's reputation.

| Task | Evaluated output | Typical metrics |
|---|---|---|
| VQA | short answer, open answer, or choice | exact/normalized accuracy, consensus accuracy, official judge |
| DESCRIPTION | caption or free-form scene description | CIDEr, BLEU, METEOR, ROUGE-L, SPICE |
| GROUNDING | box, point, region, or referred target | IoU, IoU-threshold hit, point accuracy |
| OCR | recognized or reasoned text answer | benchmark-specific exact/fuzzy/category score |
| ACTION PLANNING | next action or plan, often multiple choice | choice accuracy or official planner score |

Common benchmark interpretations:

- **RealWorldQA**: image VQA; official answer-matching accuracy.
- **EgoPlan-Bench**: egocentric video VQA/action planning; official A-D choice accuracy.
- **Flickr30k**: image DESCRIPTION/captioning; official corpus caption metrics when the caption track is used. It can also support a separate retrieval/grounding task, so identify the evaluated track.
- **OpenEQA**: open-ended embodied VQA/description. Use its official LLM-judge protocol when available. Token-F1 against references is diagnostic, not the official benchmark score.
- **RoboSpatial**: spatial grounding/point prediction; use the benchmark's official coordinate and accuracy implementation.
- **Video-MME**: video multiple-choice VQA; report the subtitle/no-subtitle scope and official accuracy breakdowns.
- **OCRBench v2**: mixed VQA/OCR; use the official language/category workbook or scorer and its macro aggregation.
- **Visual Genome**: task depends on conversion. Region description is DESCRIPTION, question-answer rows are VQA, and box prediction is GROUNDING.
- **LLaVA-Instruct**: a mixed instruction corpus, not one homogeneous benchmark; split by task or declare a mixed diagnostic.

For self-built validation conversions, prefix the metric scope with `local_` or `diagnostic_` and record the conversion code. Do not infer official comparability from a familiar dataset name.
