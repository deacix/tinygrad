import contextlib, io, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test.external.external_qwen_vl import (
  ROOT, answer_matches, checkpoint_phase, controlled_suite, digest, image_url, parse_args, quality_summary, rss_record, suite_manifest,
)

class TestQwenEvaluation(unittest.TestCase):
  def test_import_does_not_load_reference(self):
    subprocess.run([sys.executable, "-c", "import sys; import test.external.external_qwen_vl; "
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
    self.assertTrue(quality_summary(cases, results)['passed'])
    results[2]['answer'] = 'wrong'
    self.assertFalse(quality_summary(cases, results)['passed'])
    for failing in ('order_red_blue', 'text_math', 'ablation_blank'):
      results = [{'id':c['id'], 'answer':c['expected'][0], 'stop_reason':'eos'} for c in cases]
      next(r for r in results if r['id'] == failing)['answer'] = 'wrong'
      self.assertFalse(quality_summary(cases, results)['passed'])
    # A truncated answer that happens to equal the label is not a complete successful response.
    next(r for r in results if r['id'] == 'order_red_blue')['stop_reason'] = 'output_limit'
    self.assertFalse(quality_summary(cases, results)['groups_passed']['ordering'])

  def test_cli_modes_and_resource_bounds(self):
    self.assertEqual(parse_args(['--phase','processor-compare']).phase, 'processor-compare')
    self.assertTrue(parse_args(['--phase','fusion','--synthetic']).synthetic)
    self.assertTrue(parse_args(['--phase','text','--synthetic']).synthetic)
    for argv in (['--phase','text'], ['--verify','--synthetic'], ['--phase','processor-compare','--synthetic'],
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

  def test_untrusted_bundle_rejected_before_model_or_template(self):
    with tempfile.TemporaryDirectory() as tmp:
      gguf = Path(tmp)/'model.gguf'
      gguf.touch()
      args = parse_args(['--phase','e2e','--model-dir',tmp,'--gguf',str(gguf)])
      with patch('tinygrad.llm.model.Transformer.from_gguf') as load, patch('jinja2.Environment.from_string') as compile_template:
        with self.assertRaisesRegex(ValueError, 'missing or incorrect bundle file'): checkpoint_phase(args)
        load.assert_not_called()
        compile_template.assert_not_called()

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

  def test_fixture_archive_immutable(self):
    manifest = json.loads((ROOT/'manifest.json').read_text())
    self.assertEqual(len(manifest['arrays']), 356)
    self.assertEqual(digest((ROOT/'reference.npz').read_bytes()), '90d713e72d6592edd26f4b0e3f1ba21ca40c77b275f2973f6e401e68f2f4d097')
    self.assertLessEqual(sum((ROOT/name).stat().st_size for name in ('manifest.json','reference.npz')), 2*1024*1024)

if __name__ == '__main__': unittest.main()
