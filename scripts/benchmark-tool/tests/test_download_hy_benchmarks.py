import argparse
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import download_hy_benchmarks as downloader


class _Response:

    def __init__(self, content: bytes, status: int = 200):
        self._stream = io.BytesIO(content)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def getcode(self) -> int:
        return self.status


class HyBenchmarkManifestTest(unittest.TestCase):

    def test_manifest_covers_all_requested_benchmarks(self):
        manifest = downloader.load_manifest(downloader.DEFAULT_MANIFEST)
        benchmarks = manifest['benchmarks']
        counts = {}
        for benchmark in benchmarks:
            counts[benchmark['category']] = counts.get(benchmark['category'], 0) + 1

        self.assertEqual(len(benchmarks), 38)
        self.assertEqual(counts, {
            'action_relevant_state_understanding': 23,
            'action_transition_reasoning': 8,
            'sequential_adaptive_reasoning': 7,
        })
        by_name = {benchmark['name']: benchmark for benchmark in benchmarks}
        self.assertEqual(
            by_name['SITE-Bench-Image']['source_keys'],
            by_name['SITE-Bench-Video']['source_keys'])
        self.assertEqual(
            by_name['ShareRobot-Bench-Affordance']['source_keys'],
            by_name['ShareRobot-Bench-Trajectory']['source_keys'])
        self.assertEqual(
            by_name['RoboBench-MCQ']['source_keys'],
            by_name['RoboBench-Planning']['source_keys'])
        self.assertEqual(by_name['RoboSpatial-Home']['availability'], 'complete')
        self.assertEqual(by_name['EgoPlan2']['availability'], 'restricted')
        self.assertEqual(by_name['Depth-InHouse']['availability'], 'unavailable')

    def test_verified_source_provider_policy_and_pinned_revisions(self):
        manifest = downloader.load_manifest(downloader.DEFAULT_MANIFEST)
        sources = {source['key']: source for source in manifest['sources']}

        self.assertEqual(sources['ms-blink']['provider'], 'modelscope')
        self.assertEqual(sources['ms-vsi-bench']['provider'], 'modelscope')
        self.assertEqual(sources['hf-pointbench']['provider'], 'huggingface')
        self.assertEqual(sources['existing-openeqa']['provider'], 'existing')
        self.assertEqual(sources['existing-robospatial-home']['provider'], 'existing')
        for source in sources.values():
            if source['provider'] in {'modelscope', 'huggingface', 'git'}:
                self.assertRegex(source['revision'], r'^[0-9a-f]{8,40}$')

    def test_validation_rejects_traversal_and_unknown_source(self):
        manifest = {
            'schema_version': 1,
            'sources': [{
                'key': 'bad',
                'provider': 'huggingface',
                'repo_id': 'owner/repo',
                'revision': '01234567',
                'target_dir': '../escape',
                'availability': 'public',
            }],
            'benchmarks': [{
                'name': 'Example',
                'category': 'action_transition_reasoning',
                'source_keys': ['missing'],
                'availability': 'complete',
            }],
        }

        errors = downloader.validate_manifest(manifest)

        self.assertTrue(any('safe non-empty relative path' in error for error in errors))
        self.assertTrue(any('unknown source' in error for error in errors))
        self.assertTrue(any('not referenced' in error for error in errors))

    def test_shared_source_is_selected_once(self):
        manifest = downloader.load_manifest(downloader.DEFAULT_MANIFEST)
        selected = downloader.select_benchmarks(
            manifest, ['SITE-Bench-Image', 'SITE-Bench-Video'], [])

        sources = downloader.selected_sources(manifest, selected, [])

        self.assertEqual([source['key'] for source in sources], ['hf-site-bench'])


class ModelScopeDownloadTest(unittest.TestCase):

    def test_tree_listing_recurses_and_paginates(self):
        calls = []

        def fake_request(url, _timeout, _retries):
            query = parse_qs(urlparse(url).query, keep_blank_values=True)
            root = query['Root'][0]
            page = int(query['PageNumber'][0])
            calls.append((root, page))
            if root == '' and page == 1:
                entries = [
                    {'Path': 'nested', 'Type': 'tree', 'Size': 0},
                    {'Path': 'README.md', 'Type': 'blob', 'Size': 10, 'Sha256': 'a' * 64},
                ]
                total = 101
            elif root == '' and page == 2:
                entries = [
                    {'Path': 'metadata.json', 'Type': 'blob', 'Size': 20, 'Sha256': 'b' * 64},
                ]
                total = 101
            elif root == 'nested' and page == 1:
                entries = [
                    {'Path': 'nested/data.bin', 'Type': 'blob', 'Size': 30, 'Sha256': 'c' * 64},
                ]
                total = 1
            else:
                self.fail(f'Unexpected request root={root!r} page={page}')
            return {
                'Code': 200,
                'Data': {'Files': entries, 'TotalCount': total},
            }

        source = {'repo_id': 'owner/repo', 'revision': '01234567'}
        with mock.patch.object(downloader, '_request_json', side_effect=fake_request):
            files = downloader.list_modelscope_files(source, timeout=1, retries=0)

        self.assertEqual([item['path'] for item in files], [
            'README.md', 'metadata.json', 'nested/data.bin'])
        self.assertEqual(calls, [('', 1), ('', 2), ('nested', 1)])

    def test_partial_file_is_resumed_and_verified(self):
        full_content = b'abcdef'
        source = {
            'repo_id': 'owner/repo',
            'revision': '01234567',
        }
        item = {
            'path': 'data/file.bin',
            'size': len(full_content),
            'sha256': hashlib.sha256(full_content).hexdigest(),
        }
        seen_range = []

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 1)
            seen_range.append(request.get_header('Range'))
            return _Response(full_content[3:], status=206)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            partial = root / 'data' / 'file.bin.part'
            partial.parent.mkdir(parents=True)
            partial.write_bytes(full_content[:3])
            with mock.patch.object(downloader, 'urlopen', side_effect=fake_urlopen):
                result = downloader._download_modelscope_file(
                    source, item, root, timeout=1, retries=0)

            self.assertEqual((root / 'data' / 'file.bin').read_bytes(), full_content)
            self.assertFalse(partial.exists())

        self.assertEqual(seen_range, ['bytes=3-'])
        self.assertEqual(result['status'], 'resumed')


class LocalVerificationTest(unittest.TestCase):

    def test_new_no_checkout_git_clone_skips_dirty_check_until_checkout(self):
        revision = 'a' * 40
        source = {
            'key': 'git-source',
            'provider': 'git',
            'repo_url': 'https://example.invalid/benchmark.git',
            'revision': revision,
            'target_dir': 'benchmark',
            'availability': 'metadata_only',
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / source['target_dir']
            commands = []

            def run_command(command, capture_output=False):
                commands.append(list(command))
                if command[1] == 'clone':
                    (target / '.git').mkdir(parents=True)
                elif 'checkout' in command:
                    (target / 'README.md').write_text('benchmark', encoding='utf-8')
                stdout = revision if 'rev-parse' in command else ''
                return mock.Mock(returncode=0, stdout=stdout)

            with mock.patch.object(downloader, '_run_command', side_effect=run_command):
                result = downloader.download_git_source(source, root, refresh=False)

            self.assertEqual(result['status'], 'metadata_only')
            self.assertEqual(result['file_count'], 1)
            self.assertFalse(any('status' in command for command in commands))

    def test_required_file_size_and_hash_are_verified(self):
        content = b'known benchmark shard'
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / 'data' / 'shard.parquet'
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            source = {
                'required_files': [{
                    'path': 'data/shard.parquet',
                    'size': len(content),
                    'sha256': hashlib.sha256(content).hexdigest(),
                }],
                'minimum_files': 1,
            }

            stats = downloader._verify_required_data(root, source)

            self.assertEqual(stats['file_count'], 1)
            path.write_bytes(b'corrupt')
            with self.assertRaises(downloader.DownloadError):
                downloader._verify_required_data(root, source)

    def test_marker_is_invalidated_when_payload_disappears(self):
        source = {
            'key': 'source',
            'provider': 'huggingface',
            'repo_id': 'owner/repo',
            'revision': '01234567',
            'availability': 'public',
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            payload = target / 'payload.bin'
            payload.write_bytes(b'data')
            file_count, total_bytes = downloader._tree_stats(target)
            downloader._write_marker(target, source, file_count, total_bytes)
            self.assertIsNotNone(downloader._read_marker(target, source))

            payload.unlink()

            self.assertIsNone(downloader._read_marker(target, source))

    def test_tree_stats_prunes_only_top_level_cache_and_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / '.cache').mkdir()
            (root / '.cache' / 'ignored.bin').write_bytes(b'cache')
            (root / '.git').mkdir()
            (root / '.git' / 'ignored.bin').write_bytes(b'git')
            (root / 'payload' / '.cache').mkdir(parents=True)
            (root / 'payload' / '.cache' / 'kept.bin').write_bytes(b'nested')
            (root / 'payload' / 'data.bin').write_bytes(b'data')
            (root / downloader.MARKER_NAME).write_text('{}', encoding='utf-8')

            self.assertEqual(
                downloader._tree_stats(root),
                (2, len(b'nested') + len(b'data')),
            )

    def test_known_incomplete_benchmarks_keep_explicit_status(self):
        restricted = {
            'name': 'EgoPlan2',
            'category': 'sequential_adaptive_reasoning',
            'availability': 'restricted',
            'source_keys': ['metadata', 'media'],
        }
        results = {
            'metadata': {'status': 'metadata_only'},
            'media': {'status': 'restricted'},
        }

        result = downloader.benchmark_result(restricted, results, dry_run=False)

        self.assertEqual(result['status'], 'restricted')

    def test_dry_run_does_not_create_target_or_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'new-target'
            args = argparse.Namespace(
                manifest=downloader.DEFAULT_MANIFEST,
                target_root=target,
                report=None,
                benchmark=['BLINK'],
                category=[],
                provider=[],
                jobs=2,
                timeout=1,
                retries=0,
                hf_command=None,
                refresh=False,
                no_link_reuse=False,
                require_complete=False,
                dry_run=True,
                list=False,
            )

            report, exit_code = downloader.run(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(report['benchmark_status_counts'], {'planned': 1})
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
