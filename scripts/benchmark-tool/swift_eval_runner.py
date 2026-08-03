import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def _install_transformers_compat() -> None:
    import transformers

    if not hasattr(transformers, 'AutoModelForVision2Seq') and hasattr(transformers, 'AutoModelForImageTextToText'):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run one VLMEvalKit dataset through the ms-swift API.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--base-model')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--infer-backend', default='transformers')
    parser.add_argument('--eval-output-dir', required=True)
    parser.add_argument('--report-file', required=True)
    parser.add_argument('--eval-limit', type=int)
    parser.add_argument('--eval-num-proc', type=int, default=16)
    parser.add_argument('--eval-generation-config')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    _install_transformers_compat()

    from swift import EvalArguments, eval_main
    _install_transformers_compat()

    generation_config = json.loads(args.eval_generation_config) if args.eval_generation_config else None
    eval_kwargs = {
        'model': args.base_model or args.model_path,
        'eval_dataset': [args.dataset],
        'eval_backend': 'VLMEvalKit',
        'infer_backend': args.infer_backend,
        'eval_output_dir': args.eval_output_dir,
        'eval_limit': args.eval_limit,
        'eval_num_proc': args.eval_num_proc,
        'eval_generation_config': generation_config,
    }
    if args.base_model:
        eval_kwargs['adapters'] = [args.model_path]

    report = eval_main(EvalArguments(**eval_kwargs))
    report_file = Path(args.report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
