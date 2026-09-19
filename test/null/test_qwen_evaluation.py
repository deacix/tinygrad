import contextlib, io, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test.external.external_qwen_vl import (
  ROOT, answer_matches, checkpoint_phase, compare_reproduction, controlled_suite, digest, image_url, parse_args, quality_summary,
  rss_record, suite_manifest,
)

class TestQwenEvaluation(unittest.TestCase):
  def test_import_does_not_load_reference(self):
    subprocess.run([sys.executable, "-c", "import sys; import test.external.external_qwen_vl; import test.unit.test_qwen_vision; "
                    "assert not any(x in sys.modules for x in ('transformers', 'torch', 'torchvision', 'PIL', 'jinja2'))"], check=True)

  def test_suite_reproducible_and_bounded(self):
    from tinygrad.llm.multimodal import ImageLimits, preprocess_image
    images, cases = controlled_suite()
    again_images, again_cases = controlled_suite()
    manifest = suite_manifest(images, cases)
    self.assertEqual(manifest, suite_manifest(again_images, again_cases))
    self.assertEqual(len(cases), len({c['id'] for c in cases}))
    scored = [c for c in cases if c['scored']]
    self.assertEqual(len(scored), 20)
    self.assertEqual({g:sum(c['category'] == g for c in scored) for g in ('color','shape','ocr','chart','ordering')},
                     dict.fromkeys(('color','shape','ocr','chart','ordering'), 4))
    self.assertEqual(len([c for c in cases if not c['scored']]), 4)
    self.assertEqual([c for c in cases if c['id'] == 'color_yellow'][0]['images'], ['yellow'])
    grids = {}
    for name, image in images.items():
      with self.subTest(image=name):
        self.assertEqual(image_url(image), image_url(again_images[name]))
        pixels, grids[name] = preprocess_image(image_url(image), ImageLimits())
        self.assertEqual(pixels.shape, (grids[name][1]*grids[name][2],1536))
        self.assertTrue(np.isfinite(pixels).all())
    self.assertEqual(grids['yellow'], (1,32,32))
    for case in cases:
      self.assertTrue(case['expected'])
      self.assertLessEqual(len(case['images']), 2)
      self.assertLessEqual(sum(grids[k][1]*grids[k][2]//4 for k in case['images']), 1024)
    by_id = {c['id']:c for c in cases}
    for a, b in (('order_red_blue','order_blue_red'), ('order_circle_triangle','order_triangle_circle')):
      self.assertEqual(by_id[a]['images'], by_id[b]['images'][::-1])
      self.assertEqual(by_id[a]['question'], by_id[b]['question'])
      self.assertNotEqual(by_id[a]['expected'], by_id[b]['expected'])
    for a,b in manifest['ablation_pairs']:
      self.assertEqual(by_id[a]['question'], by_id[b]['question'])
      self.assertNotEqual(by_id[a]['images'], by_id[b]['images'])
      self.assertNotEqual(by_id[a]['expected'], by_id[b]['expected'])

  def test_predeclared_answers_not_substrings(self):
    for answer in (' RED. ', 'Red!', 'ＲＥＤ', '"red"'):
      self.assertTrue(answer_matches(answer, ['red']))
    for answer in ('blue and red', 'red or blue', 'not red', 'red blue', 'red, blue', 'reddish', 'the color is red', '<think>red</think>red'):
      self.assertFalse(answer_matches(answer, ['red']), answer)
    self.assertFalse(answer_matches('14', ['4']))
    self.assertFalse(answer_matches('STOP CAT', ['STOP']))
    self.assertTrue(answer_matches('grey', ['gray','grey']))

  def test_quality_threshold_order_and_control_gates(self):
    _, cases = controlled_suite()
    results = [{'id':c['id'], 'answer':c['expected'][0], 'stop_reason':'eos'} for c in cases]
    self.assertTrue(quality_summary(cases, results)['passed'])
    results[0]['answer'] = results[1]['answer'] = 'wrong'
    self.assertEqual(quality_summary(cases, results)['visual_correct'], 18)
    self.assertFalse(quality_summary(cases, results)['passed'])
    self.assertFalse(quality_summary(cases, results)['groups_passed']['ablation_pairs'])
    # Two misses outside the required gates remain allowed, but not three.
    results = [{'id':c['id'], 'answer':c['expected'][0], 'stop_reason':'eos'} for c in cases]
    for r in results:
      if r['id'] in ('shape_square', 'ocr_cat'): r['answer'] = 'wrong'
    self.assertTrue(quality_summary(cases, results)['passed'])
    next(r for r in results if r['id'] == 'color_green')['answer'] = 'wrong'
    self.assertFalse(quality_summary(cases, results)['passed'])
    for failing in ('order_red_blue', 'text_math', 'color_red', 'color_blue', 'ablation_blank', 'ablation_removed'):
      results = [{'id':c['id'], 'answer':c['expected'][0], 'stop_reason':'eos'} for c in cases]
      next(r for r in results if r['id'] == failing)['answer'] = 'wrong'
      self.assertFalse(quality_summary(cases, results)['passed'])
    # A truncated answer that happens to equal the label is not a complete successful response.
    next(r for r in results if r['id'] == 'order_red_blue')['stop_reason'] = 'output_limit'
    self.assertFalse(quality_summary(cases, results)['groups_passed']['ordering'])

  def test_ablation_identical_answers_fail_each_pair(self):
    images, cases = controlled_suite()
    by_id = {c['id']:c for c in cases}
    for a,b in suite_manifest(images, cases)['ablation_pairs']:
      for same in (by_id[a]['expected'][0], by_id[b]['expected'][0], 'wrong'):
        with self.subTest(pair=(a,b), answer=same):
          results = [{'id':c['id'], 'answer':same if c['id'] in (a,b) else c['expected'][0], 'stop_reason':'eos'} for c in cases]
          summary = quality_summary(cases, results)
          self.assertGreaterEqual(summary['visual_correct'], 18)
          self.assertFalse(summary['passed'])
          self.assertFalse(next(p['passed'] for p in summary['ablation_pairs'] if p['endpoints'] == [a,b]))
      # Exact labels do not excuse incomplete generations at either endpoint.
      for endpoint in (a,b):
        results = [{'id':c['id'], 'answer':c['expected'][0], 'stop_reason':'output_limit' if c['id'] == endpoint else 'eos'} for c in cases]
        self.assertFalse(quality_summary(cases, results)['groups_passed']['ablation_pairs'])

  def test_cli_modes_and_resource_bounds(self):
    self.assertEqual(parse_args(['--phase','processor-compare']).phase, 'processor-compare')
    self.assertTrue(parse_args(['--phase','fusion','--synthetic']).synthetic)
    self.assertTrue(parse_args(['--phase','text','--synthetic']).synthetic)
    self.assertTrue(parse_args(['--phase','vision','--synthetic']).synthetic)
    for argv in (['--phase','text'], ['--phase','vision'], ['--verify','--synthetic'], ['--phase','processor-compare','--synthetic'],
                 ['--phase','e2e'], ['--phase','benchmark'], ['--verify','--warm-runs','4'],
                 ['--verify','--max-output-tokens','257'], ['--verify','--chunk-size','0'],
                 ['--verify','--max-context','8192'], ['--verify','--report','unused.json']):
      with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
      gguf = Path(tmp)/'model.gguf'
      gguf.touch()
      for phase in ('e2e','benchmark'):
        args = parse_args(['--phase',phase,'--model-dir',tmp,'--gguf',str(gguf),'--image-suite','controlled'])
        self.assertEqual(args.gguf, gguf)
        self.assertEqual(args.warm_runs, 2)

  def test_vision_cli_dispatch(self):
    from test.external.external_qwen_vl import main
    args = parse_args(['--phase','vision','--synthetic'])
    payload = {'phase':'vision', 'synthetic':True, 'metrics':{'full_tower':{'max_abs':0.0}}}
    stdout = io.StringIO()
    with patch('test.external.external_qwen_vl.parse_args', return_value=args), \
         patch('test.external.external_qwen_vl.verify') as verify, \
         patch('test.external.external_qwen_vl.vision_phase', return_value=payload) as vision, \
         patch('test.external.external_qwen_vl.fusion_phase', side_effect=AssertionError('wrong phase')), contextlib.redirect_stdout(stdout):
      main()
    verify.assert_called_once_with(ROOT)
    vision.assert_called_once()
    manifest, arrays = vision.call_args.args
    self.assertEqual(manifest['configs']['vision']['depth'], 2)
    self.assertIn('vision.weights.patch_embed.proj.weight', arrays)
    self.assertEqual(json.loads(stdout.getvalue()), payload)

  def test_untrusted_bundle_rejected_before_model_or_template(self):
    with tempfile.TemporaryDirectory() as tmp:
      gguf = Path(tmp)/'model.gguf'
      gguf.touch()
      args = parse_args(['--phase','e2e','--model-dir',tmp,'--gguf',str(gguf)])
      with patch('tinygrad.llm.model.Transformer.from_gguf') as load, patch('jinja2.Environment.from_string') as compile_template:
        with self.assertRaisesRegex(ValueError, 'missing or incorrect bundle file'): checkpoint_phase(args)
        load.assert_not_called()
        compile_template.assert_not_called()

  def test_checkpoint_compiles_verified_template_bytes(self):
    from unittest.mock import MagicMock
    with tempfile.TemporaryDirectory() as tmp:
      gguf = Path(tmp)/'model.gguf'
      gguf.touch()
      # Simulate a file replaced after validation; only the returned trusted snapshot may be compiled.
      (Path(tmp)/'chat_template.jinja').write_text('untrusted replacement')
      args = parse_args(['--phase','e2e','--model-dir',tmp,'--gguf',str(gguf)])
      model, tokenizer = MagicMock(), MagicMock()
      model.token_embd.weight.device, tokenizer.bos_id = 'CPU', None
      with contextlib.ExitStack() as stack:
        validate = stack.enter_context(patch('tinygrad.llm.vision.validate_vision_bundle',
                                            return_value={'chat_template.jinja':b'trusted template'}))
        stack.enter_context(patch('tinygrad.llm.vision.validate_vision_metadata'))
        stack.enter_context(patch('tinygrad.llm.model.Transformer.from_gguf', return_value=(model, {})))
        stack.enter_context(patch('tinygrad.llm.vision.load_vision'))
        stack.enter_context(patch('tinygrad.Device'))
        stack.enter_context(patch('tinygrad.nn.state.get_parameters', return_value=[]))
        stack.enter_context(patch('tinygrad.llm.cli.SimpleTokenizer.from_gguf_kv', return_value=tokenizer))
        compile_template = stack.enter_context(patch('jinja2.Environment.from_string', side_effect=ValueError('stop after compilation')))
        with self.assertRaisesRegex(ValueError, 'stop after compilation'): checkpoint_phase(args)
        validate.assert_called_once_with(args.model_dir, args.gguf)
        compile_template.assert_called_once_with('trusted template')

  def test_json_report_and_quality_exit(self):
    from test.external.external_qwen_vl import main
    # Serialization/exit plumbing only; does not exercise or claim checkpoint execution.
    for success in (False, True):
      with tempfile.TemporaryDirectory() as tmp:
        gguf, report = Path(tmp)/'model.gguf', Path(tmp)/'report.json'
        gguf.touch()
        args = parse_args(['--phase','e2e','--model-dir',tmp,'--gguf',str(gguf),'--report',str(report)])
        payload = {'phase':'e2e', 'quality_gate_passed':success}
        def fake_report(_args):
          print('diagnostic is not JSON')
          return payload
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('test.external.external_qwen_vl.parse_args', return_value=args), \
             patch('test.external.external_qwen_vl.checkpoint_phase', side_effect=fake_report), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
          if success: main()
          else:
            with self.assertRaises(SystemExit) as error: main()
            self.assertEqual(error.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue()), payload)
        self.assertEqual(report.read_text(), stdout.getvalue())
        self.assertEqual(stderr.getvalue(), 'diagnostic is not JSON\n')

  def test_rss_platform_units(self):
    self.assertEqual(rss_record(123, 'Linux')['bytes'], 123*1024)
    self.assertEqual(rss_record(123, 'Linux')['raw_unit'], 'KiB')
    self.assertEqual(rss_record(123, 'Darwin')['bytes'], 123)
    self.assertEqual(rss_record(123, 'Darwin')['raw_unit'], 'bytes')
    self.assertIsNone(rss_record(123, 'Other')['bytes'])

  def test_reproduction_identity_excludes_only_execution_provenance(self):
    from copy import deepcopy
    manifest = json.loads((ROOT/'manifest.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
      expected, actual = Path(tmp)/'expected', Path(tmp)/'actual'
      expected.mkdir()
      actual.mkdir()
      # Comparator plumbing uses small stand-ins; --verify validates real archive contents separately.
      for root in (expected, actual):
        (root/'reference.npz').write_bytes(b'archive bytes')
        (root/'manifest.json').write_text(json.dumps(manifest))
      self.assertTrue(compare_reproduction(expected, actual)['execution_provenance']['matching'])
      regenerated = deepcopy(manifest)
      regenerated['backend'].update(machine='different-architecture', python='different-python', zlib='different-zlib')
      (actual/'manifest.json').write_text(json.dumps(regenerated))
      result = compare_reproduction(expected, actual)
      self.assertFalse(result['execution_provenance']['matching'])
      self.assertEqual(result['execution_provenance']['recorded']['machine'], manifest['backend']['machine'])
      self.assertEqual(result['execution_provenance']['regenerated']['machine'], 'different-architecture')
      self.assertEqual(result['numerical_array_hashes'], 'exact')
      self.assertEqual(result['archive'], 'byte-for-byte')
      # Provenance differences must never hide numerical, archive or other identity drift.
      for field in ('arrays', 'archive', 'backend', 'configs', 'versions', 'generator'):
        with self.subTest(field=field):
          changed = deepcopy(regenerated)
          changed[field]['unexpected'] = 'changed identity'
          (actual/'manifest.json').write_text(json.dumps(changed))
          with self.assertRaisesRegex(ValueError, 'numerical array hashes/schema' if field == 'arrays' else 'manifest identity'):
            compare_reproduction(expected, actual)
      (actual/'manifest.json').write_text(json.dumps(regenerated))
      (actual/'reference.npz').write_bytes(b'different archive bytes')
      with self.assertRaisesRegex(ValueError, 'array hashes match; check archive serialization/zlib'): compare_reproduction(expected, actual)

  def test_reproduction_cli_reports_identity_and_provenance(self):
    from test.external.external_qwen_vl import main
    args = parse_args(['--check-reproducible'])
    result = {'archive':'byte-for-byte', 'execution_provenance':{'matching':False}}
    stdout = io.StringIO()
    with patch('test.external.external_qwen_vl.parse_args', return_value=args), \
         patch('test.external.external_qwen_vl.verify'), patch('test.external.external_qwen_vl.generate'), \
         patch('test.external.external_qwen_vl.compare_reproduction', return_value=result) as compare, contextlib.redirect_stdout(stdout):
      main()
    compare.assert_called_once()
    message, payload = stdout.getvalue().split('\n', 1)
    self.assertIn('manifest identity excluding execution provenance', message)
    self.assertNotIn('byte-for-byte reproducible (archive and manifest)', message)
    self.assertEqual(json.loads(payload), result)

  def test_fixture_archive_immutable(self):
    manifest = json.loads((ROOT/'manifest.json').read_text())
    self.assertEqual(len(manifest['arrays']), 356)
    self.assertEqual(digest((ROOT/'reference.npz').read_bytes()), '90d713e72d6592edd26f4b0e3f1ba21ca40c77b275f2973f6e401e68f2f4d097')
    self.assertLessEqual(sum((ROOT/name).stat().st_size for name in ('manifest.json','reference.npz')), 2*1024*1024)

if __name__ == '__main__': unittest.main()
