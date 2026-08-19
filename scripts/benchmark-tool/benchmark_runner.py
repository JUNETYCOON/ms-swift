import glob
import json
import os
import re
import selectors
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from .official_results import extract_official_metrics, write_score_csv
    from .records import EvaluationRecord
    from .registry import BenchmarkSpec, canonical_benchmark_name, require_benchmark
except ImportError:
    from official_results import extract_official_metrics, write_score_csv
    from records import EvaluationRecord
    from registry import BenchmarkSpec, canonical_benchmark_name, require_benchmark


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


@dataclass(frozen=True)
class RunnerConfig:
    base_dir: Path
    defaults: Mapping[str, Any]
    benchmarks: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PreparedRun:
    model: ModelSpec
    benchmark: BenchmarkSpec
    command: Sequence[str]
    work_dir: Path
    run_dir: Path
    stdout_log: Path
    stderr_log: Path
    result_pattern: str
    result_config: Mapping[str, Any]
    env: Mapping[str, str]
    timeout: Optional[float]


class BenchmarkRunError(RuntimeError):
    pass


def _derived_model_name(path: str) -> str:
    name = re.split(r'[/\\]+', path.rstrip('/\\'))[-1]
    return name or 'model'


def parse_model_specs(values: Sequence[str]) -> List[ModelSpec]:
    models: List[ModelSpec] = []
    names = set()
    for raw_value in values:
        value = raw_value.strip()
        if not value:
            continue
        if '=' in value:
            name, path = value.split('=', 1)
            name, path = name.strip(), path.strip()
        else:
            path = value
            name = _derived_model_name(path)
        if not name or not path:
            raise ValueError(f'Invalid model specification `{raw_value}`. Use NAME=WEIGHT_PATH or WEIGHT_PATH.')
        if name in names:
            raise ValueError(f'Duplicate model name `{name}`. Model names must be unique.')
        names.add(name)
        models.append(ModelSpec(name=name, path=path))
    if not models:
        raise ValueError('At least one trained model weight is required.')
    return models


def load_runner_config(path: Optional[Path]) -> RunnerConfig:
    if path is None:
        return RunnerConfig(base_dir=Path.cwd(), defaults={}, benchmarks={})
    if not path.is_file():
        raise ValueError(f'Runner config does not exist: {path}.')
    with path.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError('Runner config must be a JSON object.')
    defaults = data.get('defaults', {})
    benchmark_entries = data.get('benchmarks', {})
    if not isinstance(defaults, Mapping) or not isinstance(benchmark_entries, Mapping):
        raise ValueError('Runner config `defaults` and `benchmarks` must be JSON objects.')
    benchmarks: Dict[str, Mapping[str, Any]] = {}
    for name, entry in benchmark_entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f'Runner config for `{name}` must be a JSON object.')
        benchmarks[canonical_benchmark_name(str(name))] = entry
    return RunnerConfig(base_dir=path.resolve().parent, defaults=defaults, benchmarks=benchmarks)


def _merge_entry(config: RunnerConfig, benchmark: str) -> Dict[str, Any]:
    entry = dict(config.defaults)
    benchmark_entry = dict(config.benchmarks.get(benchmark, {}))
    default_env = entry.pop('env', {})
    benchmark_env = benchmark_entry.pop('env', {})
    entry.update(benchmark_entry)
    if default_env or benchmark_env:
        entry['env'] = {**dict(default_env), **dict(benchmark_env)}
    return entry


def _safe_dir_name(name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._')
    return safe or 'model'


def _expand(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format_map(context)
    except KeyError as e:
        raise ValueError(f'Unknown runner config placeholder `{{{e.args[0]}}}` in `{value}`.') from e


def _external_command(entry: Mapping[str, Any], context: Mapping[str, str]) -> List[str]:
    raw_command = entry.get('command')
    if isinstance(raw_command, str):
        raw_command = shlex.split(raw_command, posix=os.name != 'nt')
    if not isinstance(raw_command, list) or not raw_command or not all(isinstance(arg, str) for arg in raw_command):
        raise ValueError('External benchmark config requires a non-empty `command` string array.')
    return [_expand(argument, context) for argument in raw_command]


def _builtin_command(spec: BenchmarkSpec, context: Mapping[str, str], infer_backend: str,
                     eval_limit: Optional[int], eval_num_proc: int,
                     eval_generation_config: Optional[str]) -> List[str]:
    helper = Path(__file__).with_name('swift_eval_runner.py').resolve()
    command = [
        sys.executable,
        str(helper),
        '--model-path',
        context['model_path'],
        '--dataset',
        str(spec.dataset_name),
        '--infer-backend',
        infer_backend,
        '--eval-output-dir',
        context['run_dir'],
        '--report-file',
        context['official_result'],
        '--eval-num-proc',
        str(eval_num_proc),
    ]
    if context.get('base_model'):
        command.extend(['--base-model', context['base_model']])
    if eval_limit is not None:
        command.extend(['--eval-limit', str(eval_limit)])
    if eval_generation_config:
        command.extend(['--eval-generation-config', eval_generation_config])
    return command


def prepare_run(model: ModelSpec, benchmark_name: str, config: RunnerConfig, output_dir: Path,
                base_model: Optional[str] = None, infer_backend: str = 'transformers',
                eval_limit: Optional[int] = None, eval_num_proc: int = 16,
                eval_generation_config: Optional[str] = None,
                timeout: Optional[float] = None) -> PreparedRun:
    spec = require_benchmark(benchmark_name)
    run_dir = (output_dir / _safe_dir_name(model.name) / spec.name).resolve()
    stdout_log = run_dir / 'stdout.log'
    stderr_log = run_dir / 'stderr.log'
    official_result = run_dir / 'official_result.json'
    entry = _merge_entry(config, spec.name)

    context = {
        'python': sys.executable,
        'model': model.name,
        'model_name': model.name,
        'model_path': model.path,
        'base_model': base_model or '',
        'benchmark': spec.name,
        'benchmark_name': spec.display_name,
        'output_dir': str(output_dir.resolve()),
        'run_dir': str(run_dir),
        'official_result': str(official_result),
        'stdout_log': str(stdout_log),
        'stderr_log': str(stderr_log),
    }

    command_override = 'command' in entry
    if spec.backend == 'external' or command_override:
        if not entry:
            raise ValueError(
                f'Benchmark `{spec.display_name}` requires an external official runner config. '
                'Pass `--runner-config benchmark-runners.json`.')
        raw_root = str(entry.get('root', '.'))
        root_text = _expand(raw_root, context)
        work_dir = Path(root_text).expanduser()
        if not work_dir.is_absolute():
            work_dir = config.base_dir / work_dir
        work_dir = work_dir.resolve()
        context['benchmark_root'] = str(work_dir)
        command = _external_command(entry, context)
        result_config = entry.get('result')
        if not isinstance(result_config, Mapping) or not result_config.get('file'):
            raise ValueError(
                f'External benchmark `{spec.display_name}` requires `result.file` in the runner config.')
        result_config = dict(result_config)
        result_pattern = _expand(str(result_config.pop('file')), context)
    else:
        work_dir = Path(__file__).resolve().parents[2]
        command = _builtin_command(
            spec,
            context,
            infer_backend=infer_backend,
            eval_limit=eval_limit,
            eval_num_proc=eval_num_proc,
            eval_generation_config=eval_generation_config)
        raw_result_config = entry.get('result', {})
        if not isinstance(raw_result_config, Mapping):
            raise ValueError(f'Runner `result` for `{spec.display_name}` must be a JSON object.')
        result_config = dict(raw_result_config)
        result_pattern = _expand(str(result_config.pop('file', official_result)), context)
        result_config.setdefault('format', 'json')

    raw_env = entry.get('env', {})
    if not isinstance(raw_env, Mapping):
        raise ValueError(f'Runner `env` for `{spec.display_name}` must be a JSON object.')
    env = os.environ.copy()
    env.update({str(key): _expand(str(value), context) for key, value in raw_env.items()})
    run_timeout = entry.get('timeout', timeout)
    return PreparedRun(
        model=model,
        benchmark=spec,
        command=command,
        work_dir=work_dir,
        run_dir=run_dir,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        result_pattern=result_pattern,
        result_config=result_config,
        env=env,
        timeout=float(run_timeout) if run_timeout is not None else None)


def _locate_result(pattern: str, work_dir: Path) -> Path:
    path = Path(pattern).expanduser()
    if not path.is_absolute():
        path = work_dir / path
    matches = [Path(match) for match in glob.glob(str(path), recursive=True)]
    matches = [match for match in matches if match.is_file()]
    if not matches:
        raise BenchmarkRunError(f'Official benchmark result not found: {path}.')
    return max(matches, key=lambda item: item.stat().st_mtime)


def _mkdir_with_retries(path: Path, attempts: int = 3, delay: float = 5.0) -> None:
    for attempt in range(1, attempts + 1):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except OSError as e:
            if attempt == attempts:
                raise BenchmarkRunError(
                    f'Cannot create benchmark output directory: {path}. Original error: {e}. '
                    'If this path is on mounted storage, rerun with a local output directory such as '
                    '`/tmp/ms-swift-benchmark` and copy results to mounted storage after the run.') from e
            print(
                f'  warning: cannot create output directory yet ({e}); retrying in {delay:g}s '
                f'({attempt}/{attempts})...',
                file=sys.stderr,
                flush=True)
            time.sleep(delay)


def _write_live_chunk(target: Any, chunk: bytes) -> None:
    if not chunk:
        return
    try:
        target.buffer.write(chunk)
        target.buffer.flush()
        return
    except (AttributeError, OSError):
        pass
    try:
        target.write(chunk.decode(errors='replace'))
        target.flush()
    except OSError:
        pass


def _copy_log_cache(cache_path: Path, target_path: Path) -> None:
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_path, target_path)
    except OSError as e:
        print(f'  warning: cannot copy log to {target_path}: {e}', file=sys.stderr, flush=True)


def _run_process_with_live_logs(prepared: PreparedRun) -> int:
    env = dict(prepared.env)
    env.setdefault('PYTHONUNBUFFERED', '1')
    with tempfile.TemporaryDirectory(prefix='benchmark-live-logs-', dir='/tmp') as temp_dir:
        stdout_cache = Path(temp_dir) / 'stdout.log'
        stderr_cache = Path(temp_dir) / 'stderr.log'
        print(f'  live log cache: {temp_dir}', flush=True)
        try:
            with stdout_cache.open('wb') as stdout_log, stderr_cache.open('wb') as stderr_log:
                process = subprocess.Popen(
                    list(prepared.command),
                    cwd=prepared.work_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0)
                assert process.stdout is not None
                assert process.stderr is not None

                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ, (stdout_log, sys.stdout))
                selector.register(process.stderr, selectors.EVENT_READ, (stderr_log, sys.stderr))
                started_at = time.monotonic()
                try:
                    while selector.get_map():
                        wait_time = 0.5
                        if prepared.timeout is not None:
                            remaining = prepared.timeout - (time.monotonic() - started_at)
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(list(prepared.command), prepared.timeout)
                            wait_time = min(wait_time, remaining)
                        for key, _ in selector.select(wait_time):
                            chunk = os.read(key.fileobj.fileno(), 8192)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            log_file, live_target = key.data
                            log_file.write(chunk)
                            _write_live_chunk(live_target, chunk)
                    return process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    selector.close()
                    process.stdout.close()
                    process.stderr.close()
        finally:
            if stdout_cache.exists():
                _copy_log_cache(stdout_cache, prepared.stdout_log)
            if stderr_cache.exists():
                _copy_log_cache(stderr_cache, prepared.stderr_log)


def run_prepared(prepared: PreparedRun, dry_run: bool = False) -> Optional[EvaluationRecord]:
    if dry_run:
        return None
    if not prepared.work_dir.is_dir():
        raise BenchmarkRunError(
            f'Benchmark working directory does not exist for `{prepared.benchmark.display_name}`: '
            f'{prepared.work_dir}.')

    print(f'  cwd: {prepared.work_dir}', flush=True)
    print(f'  run dir: {prepared.run_dir}', flush=True)
    print(f'  command: {shlex.join(list(prepared.command))}', flush=True)
    print(f'  stdout log: {prepared.stdout_log}', flush=True)
    print(f'  stderr log: {prepared.stderr_log}', flush=True)
    if prepared.timeout is not None:
        print(f'  timeout: {prepared.timeout:g}s', flush=True)
    _mkdir_with_retries(prepared.run_dir)
    try:
        returncode = _run_process_with_live_logs(prepared)
    except subprocess.TimeoutExpired as e:
        raise BenchmarkRunError(
            f'Benchmark `{prepared.benchmark.display_name}` timed out after {prepared.timeout} seconds. '
            f'Logs: {prepared.stdout_log}, {prepared.stderr_log}.') from e
    except OSError as e:
        raise BenchmarkRunError(
            f'Benchmark `{prepared.benchmark.display_name}` failed while running or streaming logs: {e}. '
            f'Command: {prepared.command!r}. Logs: {prepared.stdout_log}, {prepared.stderr_log}.') from e

    if returncode != 0:
        raise BenchmarkRunError(
            f'Benchmark `{prepared.benchmark.display_name}` exited with code {returncode}. '
            f'Logs: {prepared.stdout_log}, {prepared.stderr_log}.')

    result_path = _locate_result(prepared.result_pattern, prepared.work_dir)
    metrics = extract_official_metrics(result_path, prepared.benchmark.name, prepared.result_config)
    record = EvaluationRecord(model=prepared.model.name, benchmark=prepared.benchmark.name, metrics=metrics)
    write_score_csv(record, prepared.run_dir / 'score.csv')
    return record
