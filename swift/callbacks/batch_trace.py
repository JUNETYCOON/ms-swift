#!/usr/bin/env python3
"""Per-step sample/loss trace callback for ms-swift SFT."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List

import torch
from transformers import TrainerControl, TrainerState

from swift.utils import get_logger
from .base import TrainerCallback

try:
    from swift.sequence_parallel import sequence_parallel
    from swift.utils.transformers_utils import get_packed_seq_params
except Exception:  # noqa: BLE001
    sequence_parallel = None
    get_packed_seq_params = None

logger = get_logger()


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(_as_str(item) for item in value)
    return str(value)


def _pick_id(record: Dict[str, Any]) -> str:
    for key in ("id", "sample_id", "video_id", "image_id", "uid"):
        if record.get(key) not in (None, ""):
            return _as_str(record[key])
    meta = record.get("metadata") or record.get("meta") or {}
    if isinstance(meta, dict):
        for key in ("id", "sample_id", "video_id", "image_id", "uid", "source_id"):
            if meta.get(key) not in (None, ""):
                return _as_str(meta[key])
    for key in ("videos", "images"):
        value = record.get(key) or []
        if isinstance(value, list) and value:
            return Path(_as_str(value[0])).name
    return ""


def _media_head(record: Dict[str, Any]) -> str:
    for key in ("videos", "images"):
        value = record.get(key) or []
        if isinstance(value, list) and value:
            return _as_str(value[0])
    return ""


def _assistant_text(record: Dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    parts = []
    for item in messages:
        if isinstance(item, dict) and item.get("role") == "assistant":
            parts.append(_as_str(item.get("content", "")))
    return "\n".join(parts)


def _gt_points(record: Dict[str, Any]) -> List[List[float]]:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return []
    bbox = objects.get("bbox") or []
    if not isinstance(bbox, list):
        return []
    points = []
    for item in bbox:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                points.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    return points


def _bbox_count(record: Dict[str, Any]) -> int:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return 0
    return len(objects.get("bbox") or []) if isinstance(objects.get("bbox"), list) else 0


def _sample_meta(record: Dict[str, Any], line_no: int, source_jsonl: str) -> Dict[str, Any]:
    return {
        "source_jsonl": source_jsonl,
        "source_line_no": line_no,
        "sample_id": _pick_id(record),
        "media_head": _media_head(record),
        "bbox_count": _bbox_count(record),
        "target_assistant_text": _assistant_text(record),
        "gt_points": _gt_points(record),
    }


def _decode_tokens(tokenizer: Any, token_ids: Any) -> str:
    if tokenizer is None or token_ids is None:
        return ""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    try:
        return tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True).strip()
    except Exception:  # noqa: BLE001
        return ""


class BatchTraceCallback(TrainerCallback):
    """Write micro-batch samples, micro-loss and a concise step summary."""

    def __init__(self, args: "TrainingArguments", trainer: "Trainer"):
        super().__init__(args, trainer)
        self.trace_dir = Path(os.environ.get("SWIFT_BATCH_TRACE_DIR", ""))
        self.fsync_interval = int(os.environ.get("SWIFT_BATCH_TRACE_FSYNC_INTERVAL", "5"))
        self.source_jsonl = os.environ.get("SWIFT_TRACE_SOURCE_JSONL", "")
        self.decode_preds = os.environ.get("SWIFT_BATCH_TRACE_DECODE_PREDS", "1").lower() in {"1", "true", "yes"}
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
        self._pending_meta: List[List[Dict[str, Any]]] = []
        self._trace_handle = None
        self._trace_writes = 0
        self._last_optimizer_step = None
        self._micro_step = 0

    def _open_trace_file(self) -> None:
        if not self.trace_dir or self._trace_handle is not None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"batch_trace_rank{self.rank}.jsonl"
        self._trace_handle = path.open("w", encoding="utf-8")

    def _should_write_event(self) -> bool:
        if sequence_parallel is None:
            return True
        try:
            if sequence_parallel.enabled():
                try:
                    return int(sequence_parallel.dp_rank) == 0
                except Exception:  # noqa: BLE001
                    return int(sequence_parallel.sp_rank) == 0
        except Exception:  # noqa: BLE001
            pass
        return True

    def _write_event(self, event: Dict[str, Any]) -> None:
        self._open_trace_file()
        if self._trace_handle is None:
            return
        self._trace_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._trace_writes += 1
        self._trace_handle.flush()
        if self._trace_writes % max(1, self.fsync_interval) == 0:
            self._trace_handle.flush()

    def on_init_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self._patch_dataset()
        self._patch_data_collator()
        self._patch_compute_loss()

    def _patch_dataset(self) -> None:
        dataset = self.trainer.train_dataset
        module_name = type(dataset).__module__ if dataset is not None else ""
        class_name = type(dataset).__name__ if dataset is not None else ""
        if dataset is None or class_name != "LazyLLMDataset" or "swift.dataset" not in module_name:
            return
        if getattr(type(dataset), "_codex_batch_trace_patched", False):
            return

        def getitem_with_trace(self, idx: int) -> Dict[str, Any]:
            actual_idx = idx
            for attempt in range(getattr(self, "n_try_fetch", 1)):
                if attempt > 0:
                    actual_idx = self._idx_list[self._idx]
                    self._idx = (self._idx + 1) % len(self)
                data = self.dataset[actual_idx]
                try:
                    encoded = self.encode_func(data, return_length=True)
                except Exception:
                    if self.strict:
                        raise
                    continue
                if isinstance(encoded, dict):
                    encoded["_sample_meta"] = _sample_meta(
                        data,
                        int(actual_idx) + 1,
                        getattr(self, "_codex_source_jsonl", ""),
                    )
                return encoded
            raise ValueError("Failed to retrieve the dataset for batch trace.")

        dataset._codex_source_jsonl = self.source_jsonl
        type(dataset)._codex_batch_trace_patched = True
        type(dataset).__getitem__ = getitem_with_trace
        logger.info(f"[batch-trace] patched LazyLLMDataset rank={self.rank}")

    def _patch_data_collator(self) -> None:
        original = self.trainer.data_collator
        if getattr(original, "_codex_batch_trace_wrapped", False):
            return

        def collate_with_trace(batch, *args, **kwargs):
            metas = []
            if isinstance(batch, list):
                for item in batch:
                    if isinstance(item, dict) and item.get("_sample_meta") is not None:
                        metas.append(item.pop("_sample_meta"))
            self._pending_meta.append(metas)
            return original(batch, *args, **kwargs)

        collate_with_trace._codex_batch_trace_wrapped = True
        self.trainer.data_collator = collate_with_trace
        logger.info(f"[batch-trace] patched data collator rank={self.rank}")

    def _patch_compute_loss(self) -> None:
        original = self.trainer.compute_loss
        if getattr(original, "_codex_batch_trace_wrapped", False):
            return

        def compute_loss_with_trace(trainer_self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            text_position_ids = inputs.get("text_position_ids")
            if text_position_ids is None:
                text_position_ids = inputs.get("position_ids")
            request_outputs = bool(return_outputs or self.decode_preds)
            result = original(
                model,
                inputs,
                return_outputs=request_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            if request_outputs:
                loss, outputs = result
            else:
                loss = result
                outputs = None
            loss_value = None
            if isinstance(loss, torch.Tensor):
                try:
                    loss_value = float(loss.detach().float().item())
                except Exception:
                    loss_value = None
            meta = self._pending_meta.pop(0) if self._pending_meta else []
            if meta or loss_value is not None:
                prediction_texts, target_texts = self._decode_batch_texts(
                    inputs,
                    outputs,
                    labels,
                    text_position_ids,
                )
                if len(prediction_texts) == len(meta):
                    for sample, prediction_text, target_text in zip(meta, prediction_texts, target_texts):
                        if prediction_text:
                            sample["prediction_text"] = prediction_text
                        if target_text:
                            sample["target_text"] = target_text
                state = trainer_self.state
                optimizer_step = int(state.global_step) + 1
                if state.global_step != self._last_optimizer_step:
                    self._last_optimizer_step = state.global_step
                    self._micro_step = 0
                micro_step = self._micro_step
                self._micro_step += 1
                event = {
                    "event": "step_loss",
                    "trace_id": f"step-{optimizer_step}-rank-{self.rank}-micro-{micro_step}-{time.time_ns()}",
                    "optimizer_step": optimizer_step,
                    "micro_step_in_optimizer_step": micro_step,
                    "global_step": int(state.global_step),
                    "rank": self.rank,
                    "micro_loss": loss_value,
                    "samples": meta,
                }
                if self._should_write_event():
                    self._write_event(event)
                if self.rank == 0 and self._should_write_event():
                    ids = [sample.get("sample_id") or "" for sample in meta]
                    logger.info(
                        "[batch-trace] step=%s micro=%s loss=%s samples=%s ids=%s",
                        optimizer_step,
                        micro_step,
                        "none" if loss_value is None else f"{loss_value:.6f}",
                        len(ids),
                        ",".join(ids[:8]),
                    )
            return (loss, outputs) if return_outputs else loss

        compute_loss_with_trace._codex_batch_trace_wrapped = True
        self.trainer.compute_loss = MethodType(compute_loss_with_trace, self.trainer)
        logger.info(f"[batch-trace] patched compute_loss rank={self.rank}")

    def _gather_preds_and_labels(
        self,
        inputs: Dict[str, Any],
        outputs: Any,
        labels: Any,
    ) -> tuple[Any, Any]:
        if outputs is None or labels is None:
            return None, None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)) and outputs:
            logits = outputs[0]
        else:
            logits = None
        if logits is None:
            return None, None
        if not isinstance(logits, torch.Tensor):
            return None, None
        if not isinstance(labels, torch.Tensor):
            return None, None

        preds = logits.detach().argmax(dim=-1)
        labels = labels.detach()
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
            labels = labels.unsqueeze(0)
        sp_size = int(getattr(getattr(self.trainer, "template", None), "sequence_parallel_size", 1) or 1)
        if sp_size > 1 and sequence_parallel is not None:
            position_ids = sequence_parallel.real_position_ids
            if sequence_parallel.rp_world_size > 1 and position_ids is not None:
                position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
            else:
                position_ids = None
            preds = sequence_parallel.gather(preds, dim=1, position_ids=position_ids)
            labels = sequence_parallel.gather(labels, dim=1, position_ids=position_ids)
            labels = torch.roll(labels, shifts=1, dims=1)
        return preds, labels

    def _decode_batch_texts(
        self,
        inputs: Dict[str, Any],
        outputs: Any,
        labels: Any,
        text_position_ids: Any,
    ) -> tuple[list[str], list[str]]:
        if not self.decode_preds:
            return [], []
        try:
            preds, gathered_labels = self._gather_preds_and_labels(inputs, outputs, labels)
            if preds is None or gathered_labels is None:
                return [], []
            tokenizer = getattr(getattr(self.trainer, "template", None), "tokenizer", None)
            if tokenizer is None:
                return [], []

            segments_preds: list[Any] = []
            segments_labels: list[Any] = []
            padding_free = bool(getattr(getattr(self.trainer, "template", None), "padding_free", False))
            if padding_free and preds.shape[0] == 1 and get_packed_seq_params is not None and text_position_ids is not None:
                cu_seqlens = get_packed_seq_params(text_position_ids.detach())["cu_seq_lens_q"]
                cu_seqlens = [int(value) for value in cu_seqlens.tolist()]
                for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                    segments_preds.append(preds[0, start:end])
                    segments_labels.append(gathered_labels[0, start:end])
            else:
                for pred_row, label_row in zip(preds, gathered_labels):
                    segments_preds.append(pred_row)
                    segments_labels.append(label_row)

            prediction_texts: list[str] = []
            target_texts: list[str] = []
            for pred_row, label_row in zip(segments_preds, segments_labels):
                target_mask = label_row != -100
                if int(label_row.numel()) > 1:
                    target_ids = label_row[1:][target_mask[1:]]
                    pred_ids = pred_row[:-1][target_mask[1:]]
                else:
                    target_ids = label_row[target_mask]
                    pred_ids = pred_row[target_mask]
                target_text = _decode_tokens(tokenizer, target_ids)
                target_token_count = int(target_ids.numel()) if isinstance(target_ids, torch.Tensor) else len(target_ids)
                if target_token_count <= 0:
                    prediction_texts.append("")
                    target_texts.append(target_text)
                    continue
                prediction_texts.append(_decode_tokens(tokenizer, pred_ids))
                target_texts.append(target_text)
            return prediction_texts, target_texts
        except Exception:  # noqa: BLE001
            if self.rank == 0:
                logger.warning("[batch-trace] logits-to-text decode failed; point accuracy will use target fallback")
            return [], []

    def on_train_end(self, args: "TrainingArguments", state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self._trace_handle is not None:
            self._trace_handle.flush()
            self._trace_handle.close()
            self._trace_handle = None
        if self.trace_dir:
            logger.info(
                f"[batch-trace] trace_dir={self.trace_dir} rank={self.rank} writes={self._trace_writes}"
            )
