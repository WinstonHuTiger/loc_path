# Testing

## Unit tests

From the repository root, after `uv sync --locked`:

```bash
uv pip check
uv run --locked xtuner train --help
uv run --locked python -m unittest discover -s tests -v
```

The unit tests cover config loading, the visual sampling budgets and the launchers. They also check that a failed Stage 1 stops Stage 2. No model weights are downloaded.

## Smoke test on GPU

To run the whole pipeline on one GPU with synthetic data:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python tools/smoke_test.py \
  --work-dir work_dirs/smoke_tiny
```

The script first generates synthetic inputs:
- two feature arrays of different lengths (NPY and H5) with grid coordinates;
- training and test JSON;
- a tiny random 8-layer Qwen model.

It then runs two iterations each of MAE pretraining, Stage 1 and Stage 2, using the released resampler, NF4 loading, bf16, FlashAttention and LoRA. Compared with real training, only the LLM size, the visual input cap (1,024 tiles), the number of workers, gradient accumulation and run length are smaller. The script checks that the losses are finite, that checkpoints are saved, and that Stage 2 updates the parameters and LoRA weights. It then generates predictions from the Stage 2 checkpoint and runs the closed-ended evaluation. `smoke_result.json` is written only if every check passes. Use a new, empty `--work-dir` for each run.

With the real Qwen2.5-7B base model (`--llm-path` also accepts a local model directory):

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python tools/smoke_test.py \
  --work-dir work_dirs/smoke_qwen7b \
  --llm-path Qwen/Qwen2.5-7B-Instruct
```

On two GPUs with DeepSpeed ZeRO-2:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python tools/smoke_test.py \
  --work-dir work_dirs/smoke_zero2 --nproc 2 --deepspeed
```

These tests only check that the code runs end to end. They do not reproduce the paper's numbers, full-length training or full 61,440-tile batches.
