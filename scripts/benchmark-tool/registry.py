from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class MetricSpec:
    name: str
    aliases: tuple
    higher_is_better: bool = True
    percentage: bool = False


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    aliases: tuple
    metrics: Mapping[str, MetricSpec]
    default_metrics: Tuple[str, ...]
    backend: str
    dataset_name: Optional[str]
    task: str
    output: str
    reference: str


RECALL_AT_1 = MetricSpec(
    name='recall_at_1',
    aliases=('recall_at_1', 'recall@1', 'recall_1', 'r@1', 'r_1', 'r1'),
    percentage=True)
RECALL_AT_5 = MetricSpec(
    name='recall_at_5',
    aliases=('recall_at_5', 'recall@5', 'recall_5', 'r@5', 'r_5', 'r5'),
    percentage=True)
RECALL_AT_10 = MetricSpec(
    name='recall_at_10',
    aliases=('recall_at_10', 'recall@10', 'recall_10', 'r@10', 'r_10', 'r10'),
    percentage=True)
ACCURACY = MetricSpec(
    name='accuracy',
    aliases=('accuracy', 'acc', 'overall_accuracy', 'overall_acc', 'top1_accuracy'),
    percentage=True)
IOU = MetricSpec(
    name='iou',
    aliases=('iou', 'mean_iou', 'avg_iou', 'average_iou'),
    percentage=False)
LLM_JUDGE = MetricSpec(
    name='llm_judge',
    aliases=('llm_judge', 'llm_match', 'llm-match', 'llm_score', 'llm score'),
    percentage=False)
HUMAN_JUDGE = MetricSpec(
    name='human_judge',
    aliases=('human_judge', 'human_score', 'human score'),
    percentage=False)
OCR_SCORE = MetricSpec(
    name='score',
    aliases=('score', 'total_score', 'final_score', 'final score', 'ocrbench_score'),
    percentage=False)


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    'flickr30k': BenchmarkSpec(
        name='flickr30k',
        display_name='Flickr30k',
        aliases=('flickr30k', 'flickr_30k'),
        metrics={
            'recall_at_1': RECALL_AT_1,
            'recall_at_5': RECALL_AT_5,
            'recall_at_10': RECALL_AT_10,
        },
        default_metrics=('recall_at_1', 'recall_at_5', 'recall_at_10'),
        backend='external',
        dataset_name=None,
        task='Caption phrase grounding to one or more bounding boxes',
        output='bbox',
        reference='https://aclanthology.org/Q14-1006/'),
    'openeqa': BenchmarkSpec(
        name='openeqa',
        display_name='OpenEQA',
        aliases=('openeqa', 'open_eqa'),
        metrics={
            'llm_judge': LLM_JUDGE,
            'human_judge': HUMAN_JUDGE,
        },
        default_metrics=('llm_judge', ),
        backend='external',
        dataset_name=None,
        task='Embodied first-person environment question answering',
        output='free-form text',
        reference='https://open-eqa.github.io/'),
    'egoplan_bench': BenchmarkSpec(
        name='egoplan_bench',
        display_name='EgoPlan-Bench',
        aliases=('egoplan_bench', 'egoplan', 'ego_plan_bench'),
        metrics={'accuracy': ACCURACY},
        default_metrics=('accuracy', ),
        backend='external',
        dataset_name=None,
        task='Next-step planning in first-person videos',
        output='action choice or text',
        reference='https://arxiv.org/abs/2312.06722'),
    'robospatial': BenchmarkSpec(
        name='robospatial',
        display_name='RoboSpatial',
        aliases=('robospatial', 'robo_spatial'),
        metrics={
            'accuracy': ACCURACY,
            'iou': IOU,
        },
        default_metrics=('accuracy', 'iou'),
        backend='external',
        dataset_name=None,
        task='Robot-scene spatial relations and localization',
        output='QA or bbox',
        reference='https://arxiv.org/abs/2411.16537'),
    'ocrbench': BenchmarkSpec(
        name='ocrbench',
        display_name='OCRBench',
        aliases=('ocrbench', 'ocr_bench'),
        metrics={
            'score': OCR_SCORE,
            'accuracy': ACCURACY,
        },
        default_metrics=('score', ),
        backend='ms_swift_vlmevalkit',
        dataset_name='OCRBench',
        task='OCR and document-oriented visual question answering',
        output='free-form text',
        reference='https://arxiv.org/abs/2501.00321'),
    'realworldqa': BenchmarkSpec(
        name='realworldqa',
        display_name='RealWorldQA',
        aliases=('realworldqa', 'real_world_qa', 'realworld_qa'),
        metrics={'accuracy': ACCURACY},
        default_metrics=('accuracy', ),
        backend='ms_swift_vlmevalkit',
        dataset_name='RealWorldQA',
        task='Real-world image question answering',
        output='choice token or short answer',
        reference='https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/RealWorldQA'),
    'video_mme': BenchmarkSpec(
        name='video_mme',
        display_name='Video-MME',
        aliases=('video_mme', 'videomme', 'video_mme_bench'),
        metrics={'accuracy': ACCURACY},
        default_metrics=('accuracy', ),
        backend='ms_swift_vlmevalkit',
        dataset_name='Video-MME',
        task='Video multiple-choice understanding with optional subtitles',
        output='choice token',
        reference='https://arxiv.org/abs/2405.21075'),
}


CUSTOM_DATASET_METRICS: Dict[str, MetricSpec] = {
    'acc': MetricSpec(
        name='acc',
        aliases=('acc', 'accuracy', 'exact_match', 'exact_match_acc'),
        percentage=True),
    'answer_accuracy': MetricSpec(
        name='answer_accuracy',
        aliases=('answer_accuracy', 'type_aware_accuracy', 'semantic_accuracy'),
        percentage=True),
    'semantic_similarity': MetricSpec(
        name='semantic_similarity',
        aliases=('semantic_similarity', 'answer_similarity'),
        percentage=False),
    'yes_no_accuracy': MetricSpec(
        name='yes_no_accuracy',
        aliases=('yes_no_accuracy', 'binary_accuracy'),
        percentage=True),
    'freeform_accuracy': MetricSpec(
        name='freeform_accuracy',
        aliases=('freeform_accuracy', 'free_form_accuracy'),
        percentage=True),
    'freeform_similarity': MetricSpec(
        name='freeform_similarity',
        aliases=('freeform_similarity', 'free_form_similarity'),
        percentage=False),
    'rouge_1': MetricSpec(
        name='rouge_1',
        aliases=('rouge_1', 'rouge-1'),
        percentage=False),
    'rouge_2': MetricSpec(
        name='rouge_2',
        aliases=('rouge_2', 'rouge-2'),
        percentage=False),
    'rouge_l': MetricSpec(
        name='rouge_l',
        aliases=('rouge_l', 'rouge-l'),
        percentage=False),
    'bleu_4': MetricSpec(
        name='bleu_4',
        aliases=('bleu_4', 'bleu-4'),
        percentage=False),
    'mean_iou': MetricSpec(
        name='mean_iou',
        aliases=('mean_iou', 'iou', 'avg_iou', 'average_iou'),
        percentage=False),
    'acc_iou_0_5': MetricSpec(
        name='acc_iou_0_5',
        aliases=('acc_iou_0_5', 'iou@0.5', 'iou_0.5', 'grounding_acc', 'grounding_accuracy'),
        percentage=True),
    'num_samples': MetricSpec(
        name='num_samples',
        aliases=('num_samples', 'sample_count', 'count'),
        percentage=False),
}


def normalize_name(name: str) -> str:
    return name.strip().lower().replace('-', '_').replace('@', '_').replace(' ', '_')


def parse_names(values: Optional[Iterable[str]]) -> List[str]:
    names: List[str] = []
    if values is None:
        return names
    for value in values:
        for item in value.split(','):
            item = item.strip()
            if item:
                names.append(item)
    return names


def canonical_benchmark_name(name: str) -> str:
    normalized = normalize_name(name)
    for benchmark_name, spec in BENCHMARKS.items():
        candidates = {normalize_name(alias) for alias in spec.aliases}
        candidates.update({normalize_name(spec.display_name), benchmark_name})
        if normalized in candidates:
            return benchmark_name
    return normalized


def get_benchmark(name: str) -> BenchmarkSpec:
    key = canonical_benchmark_name(name)
    if key not in BENCHMARKS:
        return BenchmarkSpec(
            name=key,
            display_name=name,
            aliases=(key, ),
            metrics=CUSTOM_DATASET_METRICS,
            default_metrics=('acc', ),
            backend='custom',
            dataset_name=None,
            task='Custom dataset',
            output='model response',
            reference='')
    return BENCHMARKS[key]


def require_benchmark(name: str) -> BenchmarkSpec:
    key = canonical_benchmark_name(name)
    if key not in BENCHMARKS:
        supported = ', '.join(spec.display_name for spec in BENCHMARKS.values())
        raise ValueError(f'Unknown benchmark `{name}`. Supported benchmarks: {supported}.')
    return BENCHMARKS[key]


def resolve_metrics(benchmarks: Iterable[str], requested_metrics: Optional[Iterable[str]] = None) -> List[str]:
    benchmark_names = list(benchmarks)
    requested = parse_names(requested_metrics)
    if requested:
        metrics: List[str] = []
        for metric in requested:
            canonical = None
            for benchmark in benchmark_names:
                spec = metric_spec_for(benchmark, metric)
                if spec is not None:
                    canonical = spec.name
                    break
            metrics.append(canonical or normalize_name(metric))
        return list(dict.fromkeys(metrics))

    metrics = []
    seen = set()
    for benchmark in benchmark_names:
        for metric in get_benchmark(benchmark).default_metrics:
            if metric not in seen:
                metrics.append(metric)
                seen.add(metric)
    return metrics


def metric_spec_for(benchmark: str, metric: str) -> Optional[MetricSpec]:
    normalized = normalize_name(metric)
    benchmark_spec = get_benchmark(benchmark)
    spec = benchmark_spec.metrics.get(normalized)
    if spec is not None:
        return spec
    for candidate in benchmark_spec.metrics.values():
        aliases = {normalize_name(alias) for alias in candidate.aliases}
        if normalized in aliases:
            return candidate
    return None


def canonical_metric_name(raw_name: str, benchmarks: Iterable[str]) -> Optional[str]:
    normalized = normalize_name(raw_name)
    for benchmark in benchmarks:
        for metric_name, spec in get_benchmark(benchmark).metrics.items():
            candidates = {normalize_name(alias) for alias in spec.aliases}
            candidates.add(metric_name)
            if normalized in candidates:
                return metric_name
    return None
