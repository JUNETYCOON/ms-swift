"""Dataset processor registry for the unified data-process CLI.

Every dataset conversion is registered as a BaseDatasetProcessor subclass.
Existing standalone scripts remain the source of truth for conversion logic;
the base class owns the common run contract, and subclasses carry
dataset-specific metadata and hooks. To add a new dataset, add one subclass
below and it will appear in ``convert_datasets.py list`` automatically.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence


PROCESSORS: Dict[str, "BaseDatasetProcessor"] = {}


class ProcessingError(RuntimeError):
    """Raised when a registered processor fails."""


class BaseDatasetProcessor(ABC):
    name = ""
    description = ""
    script = ""
    task_types: tuple = ()
    bbox_type: Optional[str] = None

    def __init__(self) -> None:
        if not self.name or not self.script:
            raise ValueError("Dataset processor needs name and script")

    @abstractmethod
    def run(self, argv: Sequence[str], cwd: Path) -> Dict[str, object]:
        """Run the conversion and return a JSON-serializable summary."""


class LegacyScriptProcessor(BaseDatasetProcessor):
    """Run one existing data-process script through the unified registry."""

    def run(self, argv: Sequence[str], cwd: Path) -> Dict[str, object]:
        script_path = Path(__file__).resolve().parent / self.script
        if not script_path.exists():
            raise ProcessingError("script not found: {}".format(script_path))
        command = [sys.executable, str(script_path), *list(argv)]
        completed = subprocess.run(command, cwd=str(cwd), check=False)
        if completed.returncode != 0:
            raise ProcessingError(
                "{} failed with exit code {}".format(self.name, completed.returncode)
            )
        return {
            "processor": self.name,
            "script": self.script,
            "exit_code": completed.returncode,
        }


def register_processor(cls):
    instance = cls()
    if instance.name in PROCESSORS:
        raise ValueError("duplicate processor: {}".format(instance.name))
    PROCESSORS[instance.name] = instance
    return cls


@register_processor
class Ai2dProcessor(LegacyScriptProcessor):
    name = "ai2d"
    description = "AI2D diagram QA -> ms-swift JSONL"
    script = "prepare_ai2d_swift.py"
    task_types = ("vqa",)


@register_processor
class ChartQaProcessor(LegacyScriptProcessor):
    name = "chartqa"
    description = "ChartQA JSON/Parquet -> ms-swift JSONL"
    script = "prepare_chartqa_swift.py"
    task_types = ("vqa", "document_qa")


@register_processor
class GqaProcessor(LegacyScriptProcessor):
    name = "gqa"
    description = "GQA balanced train/val -> ms-swift JSONL"
    script = "prepare_gqa_swift.py"
    task_types = ("vqa",)


@register_processor
class LlavaInstructProcessor(LegacyScriptProcessor):
    name = "llava-instruct"
    description = "LLaVA-Instruct 150K -> ms-swift JSONL"
    script = "prepare_llava_instruct_swift.py"
    task_types = ("vqa", "caption")


@register_processor
class VisualGenomeQaProcessor(LegacyScriptProcessor):
    name = "visualgenome-qa"
    description = "Visual Genome QA -> ms-swift JSONL"
    script = "prepare_visualgenome_swift.py"
    task_types = ("vqa",)


@register_processor
class VisualGenomeRegionsProcessor(LegacyScriptProcessor):
    name = "visualgenome-regions"
    description = "Visual Genome regions -> grounding JSONL"
    script = "prepare_visualgenome_regions_grounding.py"
    task_types = ("grounding",)
    bbox_type = "norm1000"


@register_processor
class VisualGenomeGraphProcessor(LegacyScriptProcessor):
    name = "visualgenome-graph"
    description = "Visual Genome grounded graph -> ms-swift JSONL"
    script = "prepare_visualgenome_grounded_graph.py"
    task_types = ("grounding", "caption")
    bbox_type = "norm1000"


@register_processor
class Vqav2Processor(LegacyScriptProcessor):
    name = "vqav2"
    description = "VQAv2 train -> ms-swift JSONL"
    script = "prepare_vqav2_swift.py"
    task_types = ("vqa",)


@register_processor
class TextVqaProcessor(LegacyScriptProcessor):
    name = "textvqa"
    description = "TextVQA train -> ms-swift JSONL"
    script = "prepare_textvqa_swift.py"
    task_types = ("vqa", "ocr")


@register_processor
class CocoMultiLabelProcessor(LegacyScriptProcessor):
    name = "coco-multilabel"
    description = "COCO multi-label QA -> ms-swift JSONL"
    script = "prepare_coco_multilabel_swift.py"
    task_types = ("vqa", "classification")


@register_processor
class CocoCaptionProcessor(LegacyScriptProcessor):
    name = "coco-caption"
    description = "COCO caption -> ms-swift JSONL"
    script = "prepare_cococaption_swift.py"
    task_types = ("caption",)


@register_processor
class MolmoPixmoSpatialProcessor(LegacyScriptProcessor):
    name = "molmo-pixmo-spatial"
    description = "Molmo/PixMo/SpatialVLM -> ms-swift JSONL"
    script = "prepare_molmo_pixmo_spatial_swift.py"
    task_types = ("grounding", "caption")
    bbox_type = "norm1000"


@register_processor
class PixmoPointsNorm1000Processor(LegacyScriptProcessor):
    name = "pixmo-points-norm1000"
    description = "Clean PixMo points and enforce norm1000"
    script = "clean_pixmo_points_norm1000.py"
    task_types = ("point", "grounding")
    bbox_type = "norm1000"


@register_processor
class Molmo2VideoTrackProcessor(LegacyScriptProcessor):
    name = "molmo2-videotrack"
    description = "Molmo2 video tracking -> norm1000 ms-swift JSONL"
    script = "convert_molmo2_videotrack_norm1000.py"
    task_types = ("video", "tracking", "point")
    bbox_type = "norm1000"


@register_processor
class Robo2VlmProcessor(LegacyScriptProcessor):
    name = "robo2vlm"
    description = "Robo2VLM -> ms-swift JSONL"
    script = "prepare_robo2vlm_swift.py"
    task_types = ("vqa", "robot")


@register_processor
class RoboVqaProcessor(LegacyScriptProcessor):
    name = "robovqa"
    description = "RoboVQA -> ms-swift JSONL"
    script = "prepare_robovqa_swift.py"
    task_types = ("vqa", "robot")


@register_processor
class VlmR1Processor(LegacyScriptProcessor):
    name = "vlmr1"
    description = "VLM-R1 grounding -> ms-swift JSONL"
    script = "prepare_vlmr1_swift.py"
    task_types = ("grounding",)
    bbox_type = "norm1000"


@register_processor
class LlavaCurateProcessor(LegacyScriptProcessor):
    name = "llava-curate"
    description = "Curate/deduplicate LLaVA-Instruct SFT JSONL"
    script = "curate_llava_sft.py"
    task_types = ("curate",)


@register_processor
class RoboVqaCurateProcessor(LegacyScriptProcessor):
    name = "robovqa-curate"
    description = "Curate RoboVQA SFT JSONL"
    script = "curate_robovqa_sft.py"
    task_types = ("curate",)


@register_processor
class SanitizeProcessor(LegacyScriptProcessor):
    name = "sanitize"
    description = "Sanitize and validate ms-swift JSONL"
    script = "sanitize_sft_jsonl.py"
    task_types = ("audit",)


@register_processor
class SmokeSampleProcessor(LegacyScriptProcessor):
    name = "smoke-sample"
    description = "Sample norm1000 smoke subsets"
    script = "smoke/sample_norm1000_smoke_300.py"
    task_types = ("sampling",)


@register_processor
class SmokeVideoLocalProcessor(LegacyScriptProcessor):
    name = "smoke-video-local"
    description = "Prepare local video files for smoke runs"
    script = "smoke/prepare_local_video_dataset.py"
    task_types = ("video", "sampling")


@register_processor
class DownloadPixmoMediaProcessor(LegacyScriptProcessor):
    name = "download-pixmo-media"
    description = "Download PixMo media"
    script = "download_pixmo_media.py"
    task_types = ("media",)


@register_processor
class MaterializeMolmo2MediaProcessor(LegacyScriptProcessor):
    name = "materialize-molmo2-media"
    description = "Materialize Molmo2 video tracking media"
    script = "materialize_molmo2_videotrack_media.py"
    task_types = ("video", "media")


@register_processor
class GlobalMediaDedupProcessor(LegacyScriptProcessor):
    name = "global-media-dedup"
    description = "Global media deduplication for ms-swift JSONL"
    script = "global_media_dedup.py"
    task_types = ("media", "audit")


@register_processor
class PrepareGroundingIouEvalProcessor(LegacyScriptProcessor):
    name = "prepare-grounding-iou-eval"
    description = "Prepare grounding C-IoU eval set"
    script = "prepare_grounding_iou_eval_set.py"
    task_types = ("grounding", "eval")
    bbox_type = "norm1000"


@register_processor
class SampleRoboVqaEvalProcessor(LegacyScriptProcessor):
    name = "sample-robovqa-eval"
    description = "Sample RoboVQA eval JSONL"
    script = "sample_robovqa_eval.py"
    task_types = ("eval", "sampling")


@register_processor
class AuditDlcSftProcessor(LegacyScriptProcessor):
    name = "audit-dlc-sft"
    description = "Audit DLC SFT entrypoints"
    script = "audit_dlc_sft.py"
    task_types = ("audit",)


@register_processor
class AuditMolmo2VideoTrackProcessor(LegacyScriptProcessor):
    name = "audit-molmo2-videotrack"
    description = "Audit Molmo2 video tracking ms-swift JSONL"
    script = "audit_molmo2_videotrack_swift.py"
    task_types = ("video", "audit")


@register_processor
class AuditPairedDescriptionProcessor(LegacyScriptProcessor):
    name = "audit-paired-description"
    description = "Audit paired description metrics"
    script = "audit_paired_description_metrics.py"
    task_types = ("audit", "description")


@register_processor
class AuditRoboVqaContaminationProcessor(LegacyScriptProcessor):
    name = "audit-robovqa-contamination"
    description = "Audit RoboVQA contamination"
    script = "audit_robovqa_contamination.py"
    task_types = ("audit",)


@register_processor
class ValidateEntrypointsProcessor(LegacyScriptProcessor):
    name = "validate-entrypoints"
    description = "Validate ms-swift entrypoints/manifests"
    script = "validate_sft_entrypoints.py"
    task_types = ("audit", "manifest")


@register_processor
class VerifyStage1RemediationProcessor(LegacyScriptProcessor):
    name = "verify-stage1-remediation"
    description = "Verify Stage1 data remediation"
    script = "verify_stage1_data_remediation.py"
    task_types = ("audit",)


@register_processor
class Stage1PreparePerfProcessor(LegacyScriptProcessor):
    name = "stage1-prepare-perf"
    description = "Prepare Stage1 performance dataset"
    script = "stage1_prepare_perf_dataset.py"
    task_types = ("dataset",)


@register_processor
class DiagnoseRobo2VlmInvalidRowProcessor(LegacyScriptProcessor):
    name = "diagnose-robo2vlm"
    description = "Diagnose invalid Robo2VLM rows"
    script = "diagnose_robo2vlm_invalid_row.py"
    task_types = ("diagnose",)

