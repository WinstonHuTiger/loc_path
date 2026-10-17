"""Exercise MAE, both training stages, checkpoint transfer and inference on CUDA.

Uses generated features and a tiny random Qwen2 model by default. Supply
--llm-path to exercise a downloaded Qwen2.5 model with the same test data.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from xtuner.model.utils import guess_load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / 'xtuner/configs/vis_adapter/stage_1_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl.py'
STAGE2 = ROOT / 'xtuner/configs/vis_adapter/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass.py'
MAE = ROOT / 'xtuner/configs/mae_pretraining/mae_feature_pretrain_longnet_ctx_encoder.py'


def write_json(path: Path, data: object) -> None:
    """Write a fixture or machine-readable result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def make_tiny_llm(path: Path) -> None:
    """Save a small, local eight-layer Qwen and tokenizer without downloads."""
    tokens = ['[UNK]', '[PAD]', '<|endoftext|>', '<|im_start|>', '<|im_end|>']
    tokens += ['A', 'B', 'Describe', 'the', 'slide', 'tissue', 'normal', 'tumor']
    tokens += [f'token_{index}' for index in range(len(tokens), 128)]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
        eos_token='<|im_end|>', bos_token='<|endoftext|>',
        additional_special_tokens=tokens[2:5])
    tokenizer.save_pretrained(path)
    config = Qwen2Config(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, pad_token_id=1, eos_token_id=4,
        bos_token_id=2, tie_word_embeddings=False)
    Qwen2ForCausalLM(config).save_pretrained(path)


def make_data(work: Path) -> None:
    """Generate unequal slide lengths, grid coordinates and LLaVA-style JSON."""
    rng = np.random.default_rng(2025)
    records = []
    for index, length in enumerate((512, 1152)):
        case = f'smoke_{index}'
        folder = work / 'features/smoke_224x224_b20_t15'
        npy = folder / 'npy_files'
        npy.mkdir(parents=True, exist_ok=True)
        features = rng.normal(size=(length, 512)).astype(np.float16)
        coords = np.stack((np.arange(length) % 32,
                           np.arange(length) // 32), axis=1).astype(np.float32)
        np.save(npy / f'{case}_features.npy', features)
        np.save(npy / f'{case}_coords.npy', coords)
        # Inference's default CONCH loader reads H5, so exercise both formats.
        h5_dir = folder / 'h5_files'
        h5_dir.mkdir(exist_ok=True)
        with h5py.File(h5_dir / f'{case}.h5', 'w') as handle:
            handle['features'] = features
            handle['coords'] = coords
        records.append(dict(
            id=case, image=[f'./SMOKE/{case}.h5'],
            conversations=[
                dict(from_='human', value='<image>\nDescribe the slide.'),
                dict(from_='gpt', value='A')]))
    for record in records:
        record['conversations'] = [
            {'from': turn['from_'], 'value': turn['value']}
            for turn in record['conversations']]
    write_json(work / 'train.json', records)
    write_json(work / 'benchmark/diagnosis_choice_test.json', records)


def make_config(base: Path, path: Path, steps: int, stage: int) -> None:
    """Inherit released configs, overriding only the test's run length and I/O."""
    common = f"""
# Smoke overrides: original config above, small run and generated data below.
train_cfg = dict(type=TrainLoop, max_iters={steps})
train_dataloader.update(
    batch_size=2, num_workers=0, persistent_workers=False)
train_dataloader['dataset']['sampler_cfg'] = dict(max_tokens=1024)
train_dataloader['collate_fn']['max_tokens'] = 1024
optim_wrapper['accumulative_counts'] = 1
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1),
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=False,
         begin=1, end={steps})]
default_hooks['logger']['interval'] = 1
default_hooks['checkpoint']['interval'] = {steps}
randomness = dict(seed=2025, deterministic=False)
"""
    if stage:
        common += """train_dataloader['prefetch_factor'] = None
train_dataloader['dataset']['map_num_proc'] = 1
"""
    path.write_text(base.read_text() + common, encoding='utf-8')


def run(command: list[str], env: dict[str, str], log: Path,
        expect_failure: bool = False) -> None:
    """Run a public entry point; preserve output and enforce its exit status."""
    print(f'Running {log.stem}: {" ".join(command)}', flush=True)
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False, timeout=1800)
    if (result.returncode == 0) == expect_failure:
        tail = '\n'.join(log.read_text().splitlines()[-50:])
        raise RuntimeError(f'{log.name}: exit={result.returncode}\n{tail}')


def check_training(work: Path, stage: str, steps: int) -> dict:
    """Require a final checkpoint and finite losses for every iteration."""
    checkpoint = work / stage / f'iter_{steps}.pth'
    if not checkpoint.exists():
        raise RuntimeError(f'Missing checkpoint: {checkpoint}')
    scalars = list((work / stage).glob('*/vis_data/scalars.json'))
    losses = []
    for path in scalars:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'loss' in row:
                losses.append(float(row['loss']))
    if len(losses) != steps or not np.isfinite(losses).all():
        raise RuntimeError(f'{stage}: expected {steps} finite losses, got {losses}')
    return dict(checkpoint=str(checkpoint), losses=losses)


def main() -> None:
    """Run the GPU integration test and write a result only after all checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--llm-path', help='Use Qwen2.5 weights instead of tiny Qwen.')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--deepspeed', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required; this test exercises NF4 and FlashAttention.')
    if not 1 <= args.nproc <= torch.cuda.device_count():
        parser.error('--nproc must fit the visible CUDA device count.')
    if args.steps < 2:
        parser.error('--steps must be at least 2 for warmup + cosine scheduling.')
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('Use a new or empty --work-dir to avoid stale checkpoints.')
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2025)
    make_data(work)
    llm_path = args.llm_path or str(work / 'tiny_qwen')
    if not args.llm_path:
        make_tiny_llm(Path(llm_path))
    env = os.environ.copy()
    env.update(
        LOCPATH_LLM_PATH=llm_path,
        LOCPATH_FEATURE_ROOT=str(work / 'features'),
        LOCPATH_STAGE1_REPORT_JSON=str(work / 'train.json'),
        LOCPATH_STAGE1_DATA_PATH=str(work / 'train.json'),
        LOCPATH_STAGE2_DATA_PATH=str(work / 'train.json'),
        LOCPATH_RESAMPLER_CKPT=str(work / 'mae' / f'iter_{args.steps}.pth'),
        LOCPATH_STAGE2_PRETRAINED_PTH=str(work / 'stage1' / f'iter_{args.steps}.pth'),
        NPROC_PER_NODE=str(args.nproc), NNODES='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
    env.pop('LOCPATH_UNWANTED_PREFIX_CSV', None)
    env.pop('SLIDECHAT_UNWANTED_PREFIX_CSV', None)
    env.pop('DEEPSPEED_CFG', None)
    if args.deepspeed:
        env['DEEPSPEED_CFG'] = str(
            ROOT / 'xtuner/configs/deepspeed/deepspeed_zero2.json')
    # A CLI failure must not appear successful to the sequential launcher.
    run(['xtuner', 'train', str(work / 'missing.py'), '--work-dir',
         str(work / 'expected_failure')], env, work / 'failure_exit.log', True)
    results = {}
    for index, (stage, base, launcher) in enumerate((
            ('mae', MAE, 'train_resampler_pretrain.sh'),
            ('stage1', STAGE1, 'train_stage1.sh'),
            ('stage2', STAGE2, 'train_stage2.sh'))):
        config = work / f'{stage}.py'
        make_config(base, config, args.steps, index)
        env.update(CONFIG=str(config), WORK_DIR=str(work / stage))
        run(['bash', str(ROOT / 'scripts' / launcher)], env,
            work / f'{stage}.log')
        results[stage] = check_training(work, stage, args.steps)
    before = guess_load_checkpoint(results['stage1']['checkpoint'])
    after = guess_load_checkpoint(results['stage2']['checkpoint'])
    if len([key for key in after if key.startswith('resampler.')]) < 30:
        raise RuntimeError('Checkpoint is missing frozen resampler weights.')
    if not any('lora_' in key for key in after):
        raise RuntimeError('Stage 2 checkpoint contains no LoRA weights.')
    if not any(not torch.equal(before[key], after[key])
               for key in before if key.startswith('projector.')):
        raise RuntimeError('Stage 2 did not update the projector.')
    # Prove deployment does not require the historical training checkpoints.
    env['LOCPATH_RESAMPLER_CKPT'] = str(work / 'not_shipped_mae.pth')
    env['LOCPATH_STAGE2_PRETRAINED_PTH'] = str(work / 'not_shipped_stage1.pth')
    env.update(
        CHECKPOINT=results['stage2']['checkpoint'],
        DATA_DIR=str(work / 'benchmark'),
        FEATURE_ROOT=str(work / 'features'),
        OUT_CSV=str(work / 'predictions.csv'),
        OUT_JSONL=str(work / 'predictions.jsonl'))
    run(['bash', str(ROOT / 'scripts/test_vis_adapter.sh'),
         '--sample_num', '1024', '--max_new_tokens', '8', '--router-top-m', '96'],
        env, work / 'inference.log')
    rows = [json.loads(line) for line in
            (work / 'predictions.jsonl').read_text().splitlines() if line]
    if len(rows) != 2 or len({row['id'] for row in rows}) != 2:
        raise RuntimeError(f'Expected 2 unique predictions, got {len(rows)}')
    run(['bash', str(ROOT / 'scripts/evaluate_predictions.sh'),
         '--input', env['OUT_JSONL'], '--out-dir', str(work / 'evaluation'),
         '--dry-run'], env, work / 'evaluation.log')
    summary = json.loads((work / 'evaluation/summary.json').read_text())
    if summary['totals']['total_items'] != 2:
        raise RuntimeError('Evaluation did not keep both predictions.')
    results.update(llm=llm_path, nproc=args.nproc, deepspeed=args.deepspeed,
                   predictions=len(rows), evaluation=summary)
    write_json(work / 'smoke_result.json', results)
    print(f'PASS: {work / "smoke_result.json"}', flush=True)


if __name__ == '__main__':
    main()
