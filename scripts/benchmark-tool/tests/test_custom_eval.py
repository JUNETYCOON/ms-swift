import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from custom_eval import (FREEFORM_SIMILARITY_THRESHOLD, EvalConfig, _processor_compat_kwargs, _report_progress,
                         description_scores,
                         grounded_description_scores,
                         grounded_description_text, grounding_scores, normalize_vqa_answer, parse_prediction_boxes,
                         prepare_sample, run_custom_eval, semantic_answer_similarity, vqa_scores)
from benchmark_metrics import _score_custom_dataset
from cli import _build_parser
from rescore_custom_predictions import rescore


class ProgressReportingTest(unittest.TestCase):

    def test_reports_when_a_batch_crosses_an_interval_boundary(self):
        stream = io.StringIO()
        with patch('sys.stderr', stream):
            _report_progress(3320, 100, previous_processed=3288)
            _report_progress(3352, 100, previous_processed=3320)
        self.assertEqual(stream.getvalue(), '[custom-eval] processed=3320\n')


class ProcessorCompatibilityTest(unittest.TestCase):

    def test_overrides_legacy_extra_special_tokens_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / 'tokenizer_config.json').write_text(
                json.dumps({'extra_special_tokens': ['<|image_pad|>']}), encoding='utf-8')

            self.assertEqual(_processor_compat_kwargs(source), {'extra_special_tokens': {}})

    def test_leaves_mapping_format_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / 'tokenizer_config.json').write_text(
                json.dumps({'extra_special_tokens': {'image_token': '<|image_pad|>'}}), encoding='utf-8')

            self.assertEqual(_processor_compat_kwargs(source), {})


class VQAMetricTest(unittest.TestCase):

    def test_normalization_and_single_reference_accuracy(self):
        self.assertEqual(normalize_vqa_answer('The TWO, cats!'), '2 cats')
        scores = vqa_scores('</think> Two.', ['2'])
        self.assertEqual(scores['exact_match'], 0.0)
        self.assertEqual(scores['vqa_accuracy'], 1.0)
        self.assertEqual(scores['token_f1'], 1.0)

    def test_multiple_reference_soft_accuracy(self):
        scores = vqa_scores('cat', ['cat', 'cat', 'dog', 'bird'])
        self.assertAlmostEqual(scores['vqa_accuracy'], 2 / 3)

    def test_yes_no_uses_first_explicit_prediction_token_and_cleans_reference_markup(self):
        scores = vqa_scores(
            'After checking the video: yes, the robot moves.',
            ['<think>Long annotation reasoning.</think><answer>yes</answer>'],
        )
        self.assertEqual(scores['answer_type'], 'yes_no')
        self.assertEqual(scores['answer_accuracy'], 1.0)
        self.assertEqual(scores['semantic_similarity'], 1.0)
        self.assertEqual(scores['vqa_accuracy'], 0.0)

        first_token_wins = vqa_scores('I considered yes, but my final choice is no.', ['no'])
        self.assertEqual(first_token_wins['answer_accuracy'], 0.0)

    def test_freeform_similarity_accepts_action_paraphrase_and_rejects_conflicts(self):
        self.assertGreaterEqual(
            semantic_answer_similarity('The robot grabs the cup.', ['pick up the cup']),
            FREEFORM_SIMILARITY_THRESHOLD)
        self.assertEqual(vqa_scores('The robot grabs the cup.', ['pick up the cup'])['answer_accuracy'], 1.0)
        self.assertEqual(
            vqa_scores('The robot closes the drawer.', ['The robot opens the drawer.'])['answer_accuracy'], 0.0)
        self.assertEqual(vqa_scores('Do not open the drawer.', ['open the drawer'])['answer_accuracy'], 0.0)
        self.assertEqual(vqa_scores("Don't open the drawer.", ['open the drawer'])['answer_accuracy'], 0.0)
        self.assertEqual(vqa_scores('move the cup left', ['move the cup right'])['answer_accuracy'], 0.0)

    def test_generic_custom_scorer_exposes_type_aware_metrics(self):
        scores = _score_custom_dataset([
            {'response': 'Yes, it does.', 'labels': '<answer>yes</answer>'},
            {'response': 'The robot grabs the cup.', 'labels': 'pick up the cup'},
        ], ['answer_accuracy', 'semantic_similarity', 'yes_no_accuracy', 'freeform_similarity'])
        self.assertEqual(scores['answer_accuracy'], 1.0)
        self.assertEqual(scores['yes_no_accuracy'], 1.0)
        self.assertGreaterEqual(scores['semantic_similarity'], FREEFORM_SIMILARITY_THRESHOLD)
        self.assertGreaterEqual(scores['freeform_similarity'], FREEFORM_SIMILARITY_THRESHOLD)
        self.assertLessEqual(scores['semantic_similarity'], 1.0)
        self.assertLessEqual(scores['freeform_similarity'], 1.0)

    def test_score_custom_help_lists_type_aware_metrics(self):
        stream = io.StringIO()
        with patch('sys.stdout', stream), self.assertRaises(SystemExit) as exit_context:
            _build_parser().parse_args(['score-custom', '--help'])

        self.assertEqual(exit_context.exception.code, 0)
        help_text = stream.getvalue()
        for metric in (
                'answer_accuracy', 'semantic_similarity', 'yes_no_accuracy', 'freeform_accuracy',
                'freeform_similarity'):
            self.assertIn(metric, help_text)


class DescriptionMetricTest(unittest.TestCase):

    def test_exact_description_gets_full_scores(self):
        scores = description_scores('A dog runs through grass.', ['A dog runs through grass.'])
        self.assertEqual(scores, {'token_f1': 1.0, 'rouge_l': 1.0, 'bleu_4': 1.0, 'chrf': 1.0})

    def test_chrf_and_rouge_handle_unsegmented_chinese_variation(self):
        scores = description_scores('一只黑猫坐在桌上', ['一只黑色的猫坐在桌子上'])
        self.assertGreater(scores['chrf'], 0.25)
        self.assertGreater(scores['rouge_l'], 0.7)
        self.assertLess(scores['bleu_4'], 1.0)

    def test_multiple_references_are_supported(self):
        scores = description_scores('A child is riding a bicycle.', [
            'A person is cycling on the road.',
            'A child is riding a bicycle.',
        ])
        self.assertEqual(scores['rouge_l'], 1.0)
        self.assertEqual(scores['chrf'], 1.0)

    def test_empty_prediction_and_reference_are_handled_consistently(self):
        scores = description_scores('', [''])
        self.assertEqual(scores, {'token_f1': 1.0, 'rouge_l': 1.0, 'bleu_4': 1.0, 'chrf': 1.0})


class GroundingMetricTest(unittest.TestCase):

    def test_parse_common_qwen_and_json_boxes(self):
        self.assertEqual(
            parse_prediction_boxes('<|box_start|>(100,200),(500,800)<|box_end|>'), [(100.0, 200.0, 500.0, 800.0)])
        self.assertEqual(
            parse_prediction_boxes('```json\n{"bbox_2d": [10, 20, 30, 40]}\n```'), [(10.0, 20.0, 30.0, 40.0)])

    def test_grounding_converts_real_gt_and_scores_parse_failures_as_zero(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new('RGB', (200, 100)).save(root / 'image.jpg')
            record = {
                'images': ['image.jpg'],
                'objects': {
                    'bbox': [[20, 20, 100, 80]],
                    'bbox_type': 'real',
                    'image_id': [0],
                },
            }
            scores = grounding_scores(record, '(100,200),(500,800)', 'norm1000', 0.5, root)
            self.assertAlmostEqual(scores['mean_iou'], 1.0)
            self.assertEqual(scores['iou_accuracy'], 1.0)
            self.assertTrue(scores['complete'])

            failed = grounding_scores(record, 'I cannot locate it.', 'norm1000', 0.5, root)
            self.assertEqual(failed['mean_iou'], 0.0)
            self.assertEqual(failed['iou_accuracy'], 0.0)
            self.assertFalse(failed['parse_success'])

    def test_grounding_uses_reference_box_when_objects_are_absent(self):
        record = {
            'messages': [
                {'role': 'user', 'content': '<image>Locate the described region.'},
                {'role': 'assistant', 'content': '```json\n[{"bbox_2d": [48, 302, 171, 481]}]\n```'},
            ],
            'images': ['unused.jpg'],
        }

        scores = grounding_scores(record, '{"bbox_2d": [48, 302, 171, 481]}', 'norm1000', 0.5, Path('.'))

        self.assertEqual(scores['gt_boxes'], [(48.0, 302.0, 171.0, 481.0)])
        self.assertEqual(scores['iou_accuracy'], 1.0)
        self.assertTrue(scores['complete'])

    def test_grounded_description_ignores_qwen_markup_and_coordinates(self):
        prediction = ('<|object_ref_start|>red cup<|object_ref_end|>'
                      '<|box_start|>(100,200),(500,800)<|box_end|>')

        self.assertEqual(grounded_description_text(prediction), 'red cup')
        self.assertEqual(
            grounded_description_scores(prediction, ['red cup<bbox>']),
            {'token_f1': 1.0, 'rouge_l': 1.0, 'bleu_4': 1.0, 'chrf': 1.0},
        )

        missing_text = grounded_description_scores(
            '<|box_start|>(100,200),(500,800)<|box_end|>',
            ['red cup<bbox>'],
        )
        self.assertEqual(missing_text, {'token_f1': 0.0, 'rouge_l': 0.0, 'bleu_4': 0.0, 'chrf': 0.0})

    def test_multiple_boxes_are_matched_independently_of_output_order(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new('RGB', (200, 100)).save(root / 'image.jpg')
            record = {
                'images': ['image.jpg'],
                'objects': {
                    'bbox': [[0, 0, 20, 20], [180, 80, 200, 100]],
                    'bbox_type': 'real',
                    'image_id': [0, 0],
                },
            }
            prediction = '(900,800),(1000,1000) and (0,0),(100,200)'
            scores = grounding_scores(record, prediction, 'norm1000', 0.5, root)
            self.assertAlmostEqual(scores['mean_iou'], 1.0)
            self.assertTrue(scores['complete'])

    def test_rescore_saved_predictions_without_model_inference(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / 'image.jpg'
            Image.new('RGB', (200, 100)).save(image)
            prediction_file = root / 'source.jsonl'
            prediction_file.write_text(json.dumps({
                'sample_id': 'one',
                'messages': [
                    {'role': 'user', 'content': '<image>Locate the red cup.'},
                    {'role': 'assistant', 'content': 'red cup<bbox>'},
                ],
                'images': [str(image)],
                'objects': {
                    'bbox': [[20, 20, 100, 80]],
                    'bbox_type': 'real',
                    'image_id': [0],
                },
                'labels': ['red cup<bbox>'],
                'response': 'red cup <|box_start|>(100,200),(500,800)<|box_end|>',
            }) + '\n', encoding='utf-8')
            output = root / 'rescored'

            rows = rescore(
                prediction_file, output, 'model', 'dataset', 'grounded', overwrite=True)

            self.assertEqual(rows[0]['task'], 'grounded')
            self.assertEqual(rows[0]['chrf'], 1.0)
            self.assertEqual(rows[0]['mean_iou'], 1.0)
            report = json.loads((output / 'scores.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['processed_samples'], 1)
            self.assertEqual(json.loads((output / 'predictions.jsonl').read_text())['task'], 'grounded')

    def test_rescore_vqa_reports_yes_no_and_freeform_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_file = root / 'source.jsonl'
            prediction_file.write_text(
                json.dumps({
                    'sample_id': 'binary',
                    'labels': ['<think>reason</think><answer>yes</answer>'],
                    'response': 'Yes, the robot does.',
                }) + '\n' + json.dumps({
                    'sample_id': 'freeform',
                    'labels': ['pick up the cup'],
                    'response': 'The robot grabs the cup.',
                }) + '\n',
                encoding='utf-8')
            output = root / 'rescored'

            rows = rescore(prediction_file, output, 'model', 'robovqa', 'vqa', overwrite=True)

            self.assertEqual(rows[0]['answer_accuracy'], 1.0)
            self.assertEqual(rows[0]['yes_no_samples'], 1)
            self.assertEqual(rows[0]['yes_no_accuracy'], 1.0)
            self.assertEqual(rows[0]['freeform_samples'], 1)
            self.assertEqual(rows[0]['freeform_accuracy'], 1.0)
            results = (output / 'results.csv').read_text(encoding='utf-8')
            self.assertIn('answer_type,answer_accuracy,semantic_similarity', results)
            report = json.loads((output / 'scores.json').read_text(encoding='utf-8'))
            self.assertEqual(report['freeform_similarity_threshold'], FREEFORM_SIMILARITY_THRESHOLD)


class DatasetPreparationTest(unittest.TestCase):

    def test_cli_accepts_model_type_with_underscore(self):
        args = _build_parser().parse_args([
            'eval-custom', '--val-dataset', 'val.jsonl', '--model-weights', 'model', '--model_type', 'qwen3_vl',
            '--output-dir', 'output', '--video-backend', 'decord', '--allow-empty-video-fallback',
            '--group-image-batches', '--batched-image-grids', '20x32,32x20',
            '--image-grid-fallback', '20x32',
            '--max-batched-image-grid-area', '768',
            '--group-video-batches', '--min-video-frames', '32'
        ])
        self.assertEqual(args.model_type, 'qwen3_vl')
        self.assertEqual(args.video_backend, 'decord')
        self.assertTrue(args.allow_empty_video_fallback)
        self.assertTrue(args.group_image_batches)
        self.assertEqual(args.batched_image_grids, '20x32,32x20')
        self.assertEqual(args.image_grid_fallback, '20x32')
        self.assertEqual(args.max_batched_image_grid_area, 768)
        self.assertTrue(args.group_video_batches)
        self.assertEqual(args.min_video_frames, 32)

    def test_model_loader_sets_left_padding(self):
        fake_processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(padding_side='right'),
            image_processor=types.SimpleNamespace(size={
                'shortest_edge': 65536,
                'longest_edge': 16777216,
            }),
            video_processor=types.SimpleNamespace(
                min_frames=4, max_frames=768, fetch_videos=unittest.mock.MagicMock()))
        fake_hf_config = types.SimpleNamespace(
            model_type='qwen3_vl',
            text_config=types.SimpleNamespace(
                rope_scaling=None,
                rope_parameters={
                    'mrope_interleaved': True,
                    'mrope_section': [24, 20, 20],
                    'rope_theta': 5000000,
                    'rope_type': 'default',
                },
            ),
        )
        fake_model = unittest.mock.MagicMock()
        fake_model.device = 'cpu'
        fake_model.to.return_value = fake_model
        fake_model.config = types.SimpleNamespace(
            use_cache=False, text_config=types.SimpleNamespace(use_cache=False))
        fake_model.generation_config = types.SimpleNamespace(use_cache=False)
        fake_auto_config = unittest.mock.MagicMock()
        fake_auto_config.from_pretrained.return_value = fake_hf_config
        fake_auto_processor = unittest.mock.MagicMock()
        fake_auto_processor.from_pretrained.return_value = fake_processor
        fake_model_class = unittest.mock.MagicMock()
        fake_model_class.from_pretrained.return_value = fake_model
        fake_transformers = types.ModuleType('transformers')
        fake_transformers.AutoConfig = fake_auto_config
        fake_transformers.AutoModelForImageTextToText = fake_model_class
        fake_transformers.AutoProcessor = fake_auto_processor
        fake_transformers.Qwen3VLForConditionalGeneration = fake_model_class
        fake_video_utils = types.ModuleType('transformers.video_utils')
        fake_video_utils.load_video = unittest.mock.MagicMock(return_value=('frames', 'metadata'))
        fake_torch = types.ModuleType('torch')

        config = EvalConfig(Path('val.jsonl'), Path('model'), Path('output'), 'model', 'val',
                            model_type='qwen3_vl', device='cpu', max_image_pixels=262144, max_video_frames=32,
                            video_backend='decord')
        with patch.dict(sys.modules, {
                'torch': fake_torch,
                'transformers': fake_transformers,
                'transformers.video_utils': fake_video_utils,
        }):
            from custom_eval import _load_model
            _load_model(config)
            first_sampler = object()
            second_sampler = object()
            fetched = fake_processor.video_processor.fetch_videos('clip.mp4', sample_indices_fn=first_sampler)
            fetched_again = fake_processor.video_processor.fetch_videos('clip.mp4', sample_indices_fn=second_sampler)

        self.assertEqual(fake_processor.tokenizer.padding_side, 'left')
        self.assertEqual(fake_processor.image_processor.size['longest_edge'], 262144)
        self.assertEqual(fake_processor.video_processor.max_frames, 32)
        self.assertTrue(fake_model.config.use_cache)
        self.assertTrue(fake_model.config.text_config.use_cache)
        self.assertTrue(fake_model.generation_config.use_cache)
        self.assertEqual(fake_model_class.from_pretrained.call_args.kwargs['attn_implementation'], 'sdpa')
        self.assertIs(fake_model_class.from_pretrained.call_args.kwargs['config'], fake_hf_config)
        self.assertEqual(fake_hf_config.text_config.rope_scaling, {
            'mrope_interleaved': True,
            'mrope_section': [24, 20, 20],
            'rope_type': 'default',
        })
        self.assertEqual(fetched, ('frames', 'metadata'))
        self.assertEqual(fetched_again, ('frames', 'metadata'))
        fake_video_utils.load_video.assert_called_once_with(
            'clip.mp4', backend='decord', sample_indices_fn=first_sampler)

    def test_video_loader_pads_single_frame_and_updates_indices(self):
        import numpy as np

        from custom_eval import _configure_video_backend

        metadata = types.SimpleNamespace(frames_indices=np.array([7]))
        frames = np.ones((1, 8, 8, 3), dtype=np.uint8)
        fake_video_utils = types.ModuleType('transformers.video_utils')
        fake_video_utils.VideoMetadata = unittest.mock.MagicMock()
        fake_video_utils.load_video = unittest.mock.MagicMock(return_value=(frames, metadata))
        processor = types.SimpleNamespace(
            video_processor=types.SimpleNamespace(fetch_videos=unittest.mock.MagicMock()))

        with patch.dict(sys.modules, {'transformers.video_utils': fake_video_utils}):
            _configure_video_backend(processor, 'decord')
            padded, padded_metadata = processor.video_processor.fetch_videos('clip.mp4')

        self.assertEqual(padded.shape, (2, 8, 8, 3))
        self.assertEqual(padded_metadata.frames_indices, [7, 7])
        fake_video_utils.load_video.assert_called_once_with(
            'clip.mp4', backend='decord', sample_indices_fn=None)

    def test_short_video_letterbox_preserves_aspect_ratio(self):
        import numpy as np

        from custom_eval import _ensure_minimum_video_frames

        frames = np.full((1, 4, 8, 3), 255, dtype=np.uint8)
        metadata = types.SimpleNamespace(frames_indices=np.array([3]))
        padded, padded_metadata = _ensure_minimum_video_frames(
            (frames, metadata), canonical_size=(8, 12))

        self.assertEqual(padded.shape, (2, 8, 12, 3))
        self.assertEqual(padded_metadata.frames_indices, [3, 3])
        self.assertTrue((padded[:, 1:7] == 255).all())
        self.assertTrue((padded[:, :1] == 0).all())
        self.assertTrue((padded[:, 7:] == 0).all())

    def test_tiny_video_grid_signature_uses_canonical_short_video_shape(self):
        from custom_eval import _generation_batch_minimum, _video_grid_signature

        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / 'empty.mp4'
            video.write_bytes(b'not a video')
            processor = types.SimpleNamespace(video_processor=types.SimpleNamespace(
                temporal_patch_size=2,
                patch_size=16,
                merge_size=2,
                size={'shortest_edge': 4096, 'longest_edge': 25165824},
                fps=2,
                min_frames=32,
                max_frames=32,
            ))

            signature = _video_grid_signature(str(video), processor, canonicalize_single_frame=True)

        self.assertEqual(signature, (16, 40, 72))
        config = EvalConfig(
            Path('val.jsonl'), Path('model'), Path('output'), 'model', 'val',
            min_video_batch_size=3)
        self.assertEqual(_generation_batch_minimum(('video', *signature), config), 3)
        self.assertEqual(_generation_batch_minimum(('video', 16, 34, 60), config), 3)
        self.assertEqual(_generation_batch_minimum('default', config), 1)

    def test_image_grid_signature_matches_qwen_smart_resize(self):
        from PIL import Image

        from custom_eval import _image_grid_signature

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'image.jpg'
            Image.new('RGB', (640, 480)).save(image_path)
            processor = types.SimpleNamespace(image_processor=types.SimpleNamespace(
                patch_size=16,
                merge_size=2,
                size={'shortest_edge': 65536, 'longest_edge': 524288},
            ))

            signature = _image_grid_signature(str(image_path), processor)

        self.assertEqual(signature, (30, 40))

    def test_unsupported_image_grid_uses_letterboxed_fallback(self):
        from PIL import Image

        from custom_eval import _configure_image_cache, _image_grid_signature

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'image.jpg'
            Image.new('RGB', (416, 288), color=(255, 255, 255)).save(image_path)
            source = Image.open(image_path).copy()
            fetch_images = unittest.mock.MagicMock(return_value=source)
            image_processor = types.SimpleNamespace(
                patch_size=16,
                merge_size=2,
                size={'shortest_edge': 65536, 'longest_edge': 524288},
                fetch_images=fetch_images,
            )
            processor = types.SimpleNamespace(image_processor=image_processor)
            allowed = ((20, 32), (24, 32))

            signature = _image_grid_signature(
                str(image_path), processor,
                batched_image_grids=allowed, image_grid_fallback=(24, 32))
            _configure_image_cache(
                processor, batched_image_grids=allowed, image_grid_fallback=(24, 32))
            transformed = processor.image_processor.fetch_images(str(image_path))

        self.assertEqual(signature, (24, 32))
        self.assertEqual(transformed.size, (512, 384))
        self.assertEqual(transformed.getpixel((256, 192)), (255, 255, 255))
        self.assertEqual(transformed.getpixel((256, 0)), (0, 0, 0))

    def test_large_image_grids_use_singleton_generation_batches(self):
        from custom_eval import _generation_batch_minimum, _generation_batch_size

        config = EvalConfig(
            Path('val.jsonl'), Path('model'), Path('output'), 'model', 'val',
            batch_size=8, group_image_batches=True,
            batched_image_grids=((20, 32), (24, 32), (26, 32)),
            image_grid_fallback=(24, 32),
            max_batched_image_grid_area=768)

        self.assertEqual(_generation_batch_size(('image', 24, 32), config), 8)
        self.assertEqual(_generation_batch_size(('image', 26, 32), config), 1)
        self.assertEqual(_generation_batch_size(('image', 22, 32), config), 1)
        self.assertEqual(_generation_batch_size('default', config), 8)
        self.assertEqual(_generation_batch_minimum(('image', 24, 32), config), 8)
        self.assertEqual(_generation_batch_minimum(('image', 26, 32), config), 1)

    def test_parses_image_grid_allowlist(self):
        from cli import _parse_image_grid, _parse_image_grids

        self.assertEqual(_parse_image_grids('20x32, 32X20,20x32'), ((20, 32), (32, 20)))
        self.assertEqual(_parse_image_grid('24x32'), (24, 32))
        with self.assertRaisesRegex(ValueError, 'HEIGHTxWIDTH'):
            _parse_image_grids('20-by-32')

    def test_empty_video_fallback_requires_explicit_opt_in(self):
        import numpy as np

        from custom_eval import _configure_video_backend

        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / 'empty.mp4'
            video.write_bytes(b'not a video')
            fake_video_utils = types.ModuleType('transformers.video_utils')
            fake_video_utils.VideoMetadata = types.SimpleNamespace
            fake_video_utils.load_video = unittest.mock.MagicMock(side_effect=RuntimeError('no stream'))
            processor = types.SimpleNamespace(
                video_processor=types.SimpleNamespace(fetch_videos=unittest.mock.MagicMock()))

            with patch.dict(sys.modules, {'transformers.video_utils': fake_video_utils}):
                _configure_video_backend(processor, 'decord', allow_empty_video_fallback=True)
                frames, metadata = processor.video_processor.fetch_videos(str(video))

        self.assertIsInstance(frames, np.ndarray)
        self.assertEqual(frames.shape, (2, 1080, 1920, 3))
        self.assertEqual(metadata.frames_indices, [0, 1])

    def test_prepares_video_placeholders(self):
        record = {
            'messages': [{
                'role': 'user',
                'content': '<video>Does the robot move?'
            }, {
                'role': 'assistant',
                'content': '<answer>yes</answer>'
            }],
            'videos': ['clip.mp4'],
        }
        config = EvalConfig(Path('/data/val.jsonl'), Path('.'), Path('out'), 'model', 'video-val', task='vqa')

        sample = prepare_sample(record, 1, config)

        self.assertEqual(sample.messages[0]['content'][0], {
            'type': 'video',
            'video': '/data/clip.mp4',
        })
        self.assertEqual(sample.references, ['<answer>yes</answer>'])

    def test_multimodal_messages_normalize_system_text_content(self):
        record = {
            'messages': [{
                'role': 'system',
                'content': 'You are a helpful assistant.'
            }, {
                'role': 'user',
                'content': '<image>What is shown?'
            }, {
                'role': 'assistant',
                'content': 'A cup.'
            }],
            'images': ['image.jpg'],
        }
        config = EvalConfig(Path('/data/val.jsonl'), Path('.'), Path('out'), 'model', 'image-val')

        sample = prepare_sample(record, 1, config)

        self.assertEqual(sample.messages[0]['content'], [{
            'type': 'text',
            'text': 'You are a helpful assistant.',
        }])

    def test_single_sample_retries_flash_shape_error_with_safe_attention(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is not installed')
        from custom_eval import _generate

        class FakeInputs(dict):

            def to(self, _device):
                return self

        processor = unittest.mock.MagicMock()
        processor.apply_chat_template.return_value = FakeInputs(input_ids=torch.tensor([[1, 2]]))
        processor.batch_decode.return_value = ['answer']
        model = unittest.mock.MagicMock()
        model._benchmark_force_safe_sdpa = False
        model.generate.side_effect = [
            RuntimeError('q must have shape (batch_size, seqlen_q, num_heads, head_size_og)'),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2, 4]]),
        ]

        result = _generate([[{'role': 'user', 'content': 'question'}]], model, processor, 'cpu', 8)
        second_result = _generate([[{'role': 'user', 'content': 'next question'}]], model, processor, 'cpu', 8)

        self.assertEqual(result, ['answer'])
        self.assertEqual(second_result, ['answer'])
        self.assertTrue(model._benchmark_force_safe_sdpa)
        self.assertEqual(model.generate.call_count, 3)

    def test_batch_retries_flash_shape_error_without_splitting(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is not installed')
        from custom_eval import _generate

        class FakeInputs(dict):

            def to(self, _device):
                return self

        processor = unittest.mock.MagicMock()
        processor.apply_chat_template.return_value = FakeInputs(input_ids=torch.tensor([[1, 2], [1, 2]]))
        processor.batch_decode.return_value = ['first', 'second']
        model = unittest.mock.MagicMock()
        model._benchmark_force_safe_sdpa = False
        model.generate.side_effect = [
            RuntimeError('q must have shape (batch_size, seqlen_q, num_heads, head_size_og)'),
            torch.tensor([[1, 2, 3], [1, 2, 4]]),
        ]

        result = _generate([
            [{'role': 'user', 'content': 'first question'}],
            [{'role': 'user', 'content': 'second question'}],
        ], model, processor, 'cpu', 8)

        self.assertEqual(result, ['first', 'second'])
        self.assertTrue(model._benchmark_force_safe_sdpa)
        self.assertEqual(model.generate.call_count, 2)
        self.assertTrue(processor.apply_chat_template.call_args.kwargs['padding'])
        self.assertNotIn('processor_kwargs', processor.apply_chat_template.call_args.kwargs)

    def test_image_loader_caches_consecutive_paths(self):
        from custom_eval import _configure_image_cache

        processor = unittest.mock.MagicMock()
        original_fetch = unittest.mock.MagicMock(side_effect=lambda value: f'image:{value}')
        processor.image_processor.fetch_images = original_fetch

        _configure_image_cache(processor)
        result = processor.image_processor.fetch_images(['first.jpg', 'first.jpg', 'second.jpg', 'second.jpg'])

        self.assertEqual(result, ['image:first.jpg', 'image:first.jpg', 'image:second.jpg', 'image:second.jpg'])
        self.assertEqual(original_fetch.call_args_list, [
            unittest.mock.call('first.jpg'),
            unittest.mock.call('second.jpg'),
        ])

    def test_auto_detects_grounding_and_replaces_swift_placeholders(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / 'image.jpg'
            Image.new('RGB', (200, 100)).save(image)
            dataset = root / 'val.jsonl'
            record = {
                'messages': [{
                    'role': 'user',
                    'content': '<image>Locate <ref-object>.'
                }, {
                    'role': 'assistant',
                    'content': '<bbox>'
                }],
                'images': ['image.jpg'],
                'objects': {
                    'ref': ['red cup'],
                    'bbox': [[20, 20, 100, 80]],
                    'bbox_type': 'real',
                    'image_id': [0],
                },
            }
            dataset.write_text(json.dumps(record) + '\n', encoding='utf-8')
            config = EvalConfig(dataset, root, root / 'out', 'model', 'val')
            sample = prepare_sample(record, 1, config)

            self.assertEqual(sample.task, 'grounding')
            self.assertEqual(sample.references, ['<bbox>'])
            self.assertEqual(sample.messages[0]['content'][0]['type'], 'image')
            self.assertIn('<|object_ref_start|>red cup<|object_ref_end|>', sample.question)
            self.assertNotIn('assistant', [message['role'] for message in sample.messages])

    def test_auto_detects_grounding_from_explicit_box_reference(self):
        record = {
            'messages': [{
                'role': 'user',
                'content': '<image>Return the coordinates of the red cup.'
            }, {
                'role': 'assistant',
                'content': '{"bbox_2d": [100, 200, 500, 800]}'
            }],
            'images': ['image.jpg'],
            'objects': {
                'bbox': [[100, 200, 500, 800]],
                'bbox_type': 'norm1000',
                'image_id': [0],
            },
        }
        config = EvalConfig(Path('/data/val.jsonl'), Path('.'), Path('out'), 'model', 'grounding-val')

        sample = prepare_sample(record, 1, config)

        self.assertEqual(sample.task, 'grounding')

    def test_auto_detects_phrase_and_box_output_as_grounded(self):
        record = {
            'messages': [{
                'role': 'user',
                'content': '<image>Describe the image with a grounded region.'
            }, {
                'role': 'assistant',
                'content': '<ref-object><bbox>'
            }],
            'images': ['image.jpg'],
            'objects': {
                'ref': ['red cup'],
                'bbox': [[100, 200, 500, 800]],
                'bbox_type': 'norm1000',
                'image_id': [0],
            },
        }
        config = EvalConfig(Path('/data/val.jsonl'), Path('.'), Path('out'), 'model', 'grounded-val')

        sample = prepare_sample(record, 1, config)

        self.assertEqual(sample.task, 'grounded')
        self.assertEqual(sample.references, ['red cup<bbox>'])

    def test_input_box_with_text_reference_remains_description(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new('RGB', (200, 100)).save(root / 'image.jpg')
            record = {
                'messages': [{
                    'role': 'user',
                    'content': '<image>Describe the region <bbox>.'
                }, {
                    'role': 'assistant',
                    'content': 'red cup'
                }],
                'images': ['image.jpg'],
                'objects': {
                    'bbox': [[20, 20, 100, 80]],
                    'bbox_type': 'real',
                    'image_id': [0],
                },
            }
            config = EvalConfig(root / 'val.jsonl', root, root / 'out', 'model', 'region-description')

            sample = prepare_sample(record, 1, config)

            self.assertEqual(sample.task, 'description')

    def test_auto_detects_description_and_expands_reference_object(self):
        record = {
            'messages': [{
                'role': 'user',
                'content': 'Describe the highlighted region.'
            }, {
                'role': 'assistant',
                'content': '<ref-object>'
            }],
            'objects': {'ref': ['red cup']},
        }
        config = EvalConfig(Path('val.jsonl'), Path('.'), Path('out'), 'model', 'val')
        sample = prepare_sample(record, 1, config)

        self.assertEqual(sample.task, 'description')
        self.assertEqual(sample.references, ['red cup'])

    def test_cli_accepts_explicit_description_task(self):
        args = _build_parser().parse_args([
            'eval-custom', '--val-dataset', 'val.jsonl', '--model-weights', 'model', '--output-dir', 'output',
            '--task', 'description'
        ])
        self.assertEqual(args.task, 'description')

    def test_end_to_end_writes_predictions_results_and_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            dataset = root / 'val.jsonl'
            dataset.write_text(
                json.dumps(
                    {'messages': [{
                        'role': 'user',
                        'content': 'How many?'
                    }, {
                        'role': 'assistant',
                        'content': 'two'
                    }]}) + '\n',
                encoding='utf-8')
            output = root / 'output'
            config = EvalConfig(dataset, model, output, 'test-model', 'test-val')

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', return_value=['2']):
                rows = run_custom_eval(config)

            self.assertEqual(rows[0]['vqa_accuracy'], 1.0)
            self.assertTrue((output / 'predictions.jsonl').is_file())
            self.assertIn('vqa_accuracy', (output / 'results.csv').read_text(encoding='utf-8'))
            self.assertIn('test-model,test-val,vqa,1', (output / 'scores.csv').read_text(encoding='utf-8'))
            report = json.loads((output / 'scores.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['processed_samples'], 1)
            self.assertEqual(report['scores'][0]['vqa_accuracy'], 1.0)

    def test_runtime_failure_writes_partial_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            dataset = root / 'val.jsonl'
            records = [
                {'messages': [{'role': 'user', 'content': 'How many?'}, {'role': 'assistant', 'content': 'two'}]},
                {'messages': [{'role': 'user', 'content': 'What color?'}, {'role': 'assistant', 'content': 'red'}]},
            ]
            dataset.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            output = root / 'output'
            config = EvalConfig(dataset, model, output, 'test-model', 'test-val')

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', side_effect=[['2'], RuntimeError('backend failed')]):
                with self.assertRaisesRegex(RuntimeError, 'backend failed'):
                    run_custom_eval(config)

            report = json.loads((output / 'scores.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(report['processed_samples'], 1)
            self.assertEqual(report['scores'][0]['vqa_accuracy'], 1.0)

    def test_resume_restores_scores_and_runs_only_unfinished_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            dataset = root / 'val.jsonl'
            records = [
                {'messages': [{'role': 'user', 'content': 'How many?'}, {'role': 'assistant', 'content': 'two'}]},
                {'messages': [{'role': 'user', 'content': 'What color?'}, {'role': 'assistant', 'content': 'red'}]},
            ]
            dataset.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            output = root / 'output'
            first = EvalConfig(dataset, model, output, 'test-model', 'test-val')

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', side_effect=[['2'], RuntimeError('interrupted')]):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run_custom_eval(first)

            results_path = output / 'results.csv'
            with results_path.open('r', encoding='utf-8', newline='') as stream:
                legacy_rows = list(csv.DictReader(stream))
            legacy_columns = [
                column for column in legacy_rows[0]
                if column not in {'answer_type', 'answer_accuracy', 'semantic_similarity'}
            ]
            with results_path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=legacy_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(legacy_rows)

            resumed = EvalConfig(dataset, model, output, 'test-model', 'test-val', resume=True)
            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', return_value=['red']) as generate:
                rows = run_custom_eval(resumed)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(rows[0]['num_samples'], 2)
            self.assertEqual(rows[0]['vqa_accuracy'], 1.0)
            with (output / 'predictions.jsonl').open() as stream:
                self.assertEqual(sum(1 for _ in stream), 2)
            with results_path.open('r', encoding='utf-8', newline='') as stream:
                result_reader = csv.DictReader(stream)
                self.assertIn('answer_type', result_reader.fieldnames)
                self.assertEqual(len(list(result_reader)), 2)
            report = json.loads((output / 'scores.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['processed_samples'], 2)

    def test_retry_errors_removes_failed_rows_and_regenerates_only_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            dataset = root / 'val.jsonl'
            records = [
                {'messages': [{'role': 'user', 'content': 'How many?'}, {'role': 'assistant', 'content': 'two'}]},
                {'messages': [{'role': 'user', 'content': 'What color?'}, {'role': 'assistant', 'content': 'red'}]},
            ]
            dataset.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            output = root / 'output'
            first = EvalConfig(
                dataset, model, output, 'test-model', 'test-val', continue_on_error=True)

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', side_effect=[RuntimeError('oom'), ['red']]):
                rows = run_custom_eval(first)

            self.assertEqual(rows[0]['failed_samples'], 1)
            retry = EvalConfig(
                dataset, model, output, 'test-model', 'test-val', resume=True, retry_errors=True)
            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', return_value=['2']) as generate:
                rows = run_custom_eval(retry)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(rows[0]['num_samples'], 2)
            self.assertEqual(rows[0]['failed_samples'], 0)
            self.assertEqual(rows[0]['vqa_accuracy'], 1.0)
            with (output / 'predictions.jsonl').open() as stream:
                predictions = [json.loads(line) for line in stream]
            self.assertEqual(len(predictions), 2)
            self.assertNotIn('eval_error', predictions[-1])

    def test_end_to_end_writes_description_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            dataset = root / 'val.jsonl'
            dataset.write_text(
                json.dumps({
                    'messages': [{
                        'role': 'user',
                        'content': 'Describe the image.'
                    }, {
                        'role': 'assistant',
                        'content': 'A red car is parked by the road.'
                    }]
                }) + '\n',
                encoding='utf-8')
            output = root / 'output'
            config = EvalConfig(dataset, model, output, 'test-model', 'description-val')

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', return_value=['A red car is parked by the road.']):
                rows = run_custom_eval(config)

            self.assertEqual(rows[0]['task'], 'description')
            self.assertEqual(rows[0]['chrf'], 1.0)
            result = (output / 'results.csv').read_text(encoding='utf-8')
            self.assertIn('rouge_l', result)
            self.assertIn('bleu_4', result)
            self.assertIn('chrf', result)

    def test_end_to_end_grounding_writes_iou_scores(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            Image.new('RGB', (200, 100)).save(root / 'image.jpg')
            dataset = root / 'val.jsonl'
            dataset.write_text(
                json.dumps({
                    'messages': [{
                        'role': 'user',
                        'content': '<image>Locate <ref-object>.'
                    }, {
                        'role': 'assistant',
                        'content': '<bbox>'
                    }],
                    'images': ['image.jpg'],
                    'objects': {
                        'ref': ['red cup'],
                        'bbox': [[20, 20, 100, 80]],
                        'bbox_type': 'real',
                        'image_id': [0],
                    },
                }) + '\n',
                encoding='utf-8')
            output = root / 'output'
            config = EvalConfig(dataset, model, output, 'test-model', 'grounding-val')

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate',
                          return_value=['<|box_start|>(100,200),(500,800)<|box_end|>']):
                rows = run_custom_eval(config)

            self.assertEqual(rows[0]['task'], 'grounding')
            self.assertEqual(rows[0]['mean_iou'], 1.0)
            self.assertEqual(rows[0]['iou_accuracy'], 1.0)
            result = (output / 'results.csv').read_text(encoding='utf-8')
            self.assertIn('100.0, 200.0, 500.0, 800.0', result)

    def test_end_to_end_grounded_writes_description_and_iou_scores(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is not installed')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'model'
            model.mkdir()
            Image.new('RGB', (200, 100)).save(root / 'image.jpg')
            dataset = root / 'val.jsonl'
            dataset.write_text(
                json.dumps({
                    'messages': [{
                        'role': 'user',
                        'content': '<image>Describe the image with a grounded region.'
                    }, {
                        'role': 'assistant',
                        'content': '<ref-object><bbox>'
                    }],
                    'images': ['image.jpg'],
                    'objects': {
                        'ref': ['red cup'],
                        'bbox': [[20, 20, 100, 80]],
                        'bbox_type': 'real',
                        'image_id': [0],
                    },
                }) + '\n',
                encoding='utf-8')
            output = root / 'output'
            config = EvalConfig(dataset, model, output, 'test-model', 'grounded-val')
            prediction = ('<|object_ref_start|>red cup<|object_ref_end|>'
                          '<|box_start|>(100,200),(500,800)<|box_end|>')

            with patch('custom_eval._load_model', return_value=(object(), object(), 'cpu')), \
                    patch('custom_eval._generate', return_value=[prediction]):
                rows = run_custom_eval(config)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['task'], 'grounded')
            self.assertEqual(rows[0]['token_f1'], 1.0)
            self.assertEqual(rows[0]['rouge_l'], 1.0)
            self.assertEqual(rows[0]['bleu_4'], 1.0)
            self.assertEqual(rows[0]['chrf'], 1.0)
            self.assertEqual(rows[0]['mean_iou'], 1.0)
            self.assertEqual(rows[0]['iou_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
