"""Regression tests for public launcher contracts, without model downloads."""

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import ast

from xtuner.configs.loader import load_config


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    """Verify forwarding, legacy aliases and fail-fast sequential training."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        stub = self.work / 'xtuner'
        stub.write_text('''#!/usr/bin/env python3
import copy
import json
import os
from pathlib import Path
import sys
with open(os.environ['CALLS'], 'a', encoding='utf-8') as handle:
    handle.write(json.dumps(dict(args=sys.argv[1:], env=dict(os.environ))) + '\\n')
sys.exit(int(os.environ.get('FAKE_EXIT', '0')))
''')
        stub.chmod(0o755)
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith(('LOCPATH_', 'SLIDECHAT_')):
                del self.env[key]
        self.env.update(
            PATH=f'{self.work}:{self.env["PATH"]}',
            CALLS=str(self.work / 'calls.jsonl'),
            WORK_DIR=str(self.work / 'train'),
            STAGE1_WORK_DIR=str(self.work / 'stage1'),
            STAGE2_WORK_DIR=str(self.work / 'stage2'),
            SLIDECHAT_FEATURE_ROOT='/fixtures/features',
            SLIDECHAT_RESAMPLER_CKPT='/fixtures/mae.pth',
            SLIDECHAT_STAGE1_REPORT_JSON='/fixtures/report.json',
            SLIDECHAT_STAGE1_DATA_PATH='/fixtures/stage1.json',
            SLIDECHAT_STAGE2_DATA_PATH='/fixtures/stage2.json',
            SLIDECHAT_STAGE2_PRETRAINED_PTH='/fixtures/stage1.pth')
        self.env.pop('DEEPSPEED_CFG', None)

    def launch(self, name: str, *args: str) -> subprocess.CompletedProcess:
        """Invoke a real shell launcher against the isolated command stub."""
        return subprocess.run(
            ['bash', str(ROOT / 'scripts' / name), *args], env=self.env,
            capture_output=True, text=True, check=False)

    def calls(self) -> list[dict]:
        """Read invocations recorded by the stub."""
        return [json.loads(line) for line in
                (self.work / 'calls.jsonl').read_text().splitlines()]

    def test_training_aliases_and_extra_arguments(self) -> None:
        for name, expected in (
                ('train_resampler_pretrain.sh', 'LOCPATH_STAGE1_REPORT_JSON'),
                ('train_stage1.sh', 'LOCPATH_STAGE1_DATA_PATH'),
                ('train_stage2.sh', 'LOCPATH_STAGE2_PRETRAINED_PTH')):
            with self.subTest(launcher=name):
                result = self.launch(name, '--seed', '2025')
                self.assertEqual(result.returncode, 0, result.stderr)
                call = self.calls()[-1]
                self.assertEqual(call['args'][-2:], ['--seed', '2025'])
                self.assertEqual(call['env']['LOCPATH_FEATURE_ROOT'],
                                 '/fixtures/features')
                self.assertTrue(call['env'][expected].startswith('/fixtures/'))

    def test_failed_stage1_stops_stage2(self) -> None:
        self.env['FAKE_EXIT'] = '37'
        result = self.launch('train_stage1_stage2.sh')
        self.assertEqual(result.returncode, 37)
        self.assertEqual(len(self.calls()), 1)

    def test_missing_required_path_fails_before_launch(self) -> None:
        del self.env['SLIDECHAT_FEATURE_ROOT']
        result = self.launch('train_stage1.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('LOCPATH_FEATURE_ROOT', result.stderr)
        self.assertFalse((self.work / 'calls.jsonl').exists())

    def test_sequential_checkpoint_handoff(self) -> None:
        folder = self.work / 'stage1'
        folder.mkdir()
        checkpoint = folder / 'iter_2.pth'
        checkpoint.touch()
        result = self.launch('train_stage1_stage2.sh', '--seed', '2025')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]['env']['LOCPATH_STAGE2_PRETRAINED_PTH'],
                         str(checkpoint))
        self.assertEqual(calls[1]['args'][-2:], ['--seed', '2025'])


class ConfigTest(unittest.TestCase):
    """Load each released config with the same loader as the entry point."""

    def test_configs_and_visual_budget(self) -> None:
        paths = list((ROOT / 'xtuner/configs/vis_adapter').glob('stage_*.py'))
        paths += list((ROOT / 'xtuner/configs/mae_pretraining').glob('*.py'))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(config=path.name):
                n_latents = 128 if path.stem.endswith('_l128') else 256
                self.assertIn(f'n_latents={n_latents},', path.read_text())
                cfg = load_config(str(path))
                ast.parse(cfg.pretty_text)
                copy.deepcopy(cfg)
                self.assertFalse(cfg.optim_wrapper.loss_scale.enabled)
                self.assertEqual(cfg.train_dataloader.dataset.sampler_cfg.max_tokens,
                                 61440)


class CheckpointTest(unittest.TestCase):
    """Preserve frozen visual weights across ZeRO export and reload."""

    def test_frozen_weights_are_exported_and_loaded(self) -> None:
        import torch
        from xtuner.engine._strategy.deepspeed import DeepSpeedStrategy
        from xtuner.model.utils import guess_load_checkpoint

        model = torch.nn.Linear(4, 3)
        model.requires_grad_(False)
        owner = types.SimpleNamespace(
            config=dict(zero_optimization=dict(stage=2)),
            model=types.SimpleNamespace(module=model))
        with tempfile.TemporaryDirectory() as temp:
            DeepSpeedStrategy._save_portable_weights(owner, temp)
            restored = guess_load_checkpoint(temp)
            self.assertTrue(torch.equal(restored['weight'], model.weight))
            self.assertTrue(torch.equal(restored['bias'], model.bias))


class EvaluationTest(unittest.TestCase):
    """Score known closed-ended predictions with the public evaluator."""

    def test_closed_accuracy_without_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            rows = [dict(file='diagnosis_choice_test.json', id=str(index),
                         ref_answer='A', output=prediction)
                    for index, prediction in enumerate(('A', 'B'))]
            source = folder / 'predictions.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = subprocess.run(
                [os.sys.executable,
                 str(ROOT / 'data_engine/evaluate_model_output_by_task2.py'),
                 '--input', str(source), '--out-dir', str(folder / 'eval'),
                 '--dry-run'], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((folder / 'eval/summary.json').read_text())
            score = summary['by_file']['diagnosis_choice_test.json']['Close']
            self.assertEqual(score['Acc'], 0.5)
            self.assertEqual(score['counts']['items'], 2)


if __name__ == '__main__':
    unittest.main()
