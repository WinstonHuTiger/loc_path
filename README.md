# LoC-Path: Learning to Compress for Pathology Multimodal Large Language Models

> This repository contains the official training, inference and evaluation code for **LoC-Path**.

## Overview

A whole slide image (WSI) is gigapixel-scale. After tiling and encoding with a tile-level foundation model, one slide becomes thousands to tens of thousands of visual tokens. Most of these tokens are redundant, and the evidence a question actually needs is sparse. Most slide-level pathology MLLMs (e.g. WSI-LLaVA, SlideChat) follow a LLaVA-style design: they turn the slide into a long visual prefix and concatenate it with the text inside the LLM. This makes training and deployment expensive.

**LoC-Path compresses the slide before fusing it with the LLM:**

- **Sparse Token Merger (STM)** merges neighboring tile tokens.
- **MAE-pretrained resampler** compresses the slide into a fixed set of latents (L = 256). It uses sparse Top-K cross-attention and a LongNet (dilated attention) context encoder, and is pretrained with a masked-autoencoder objective on tile features.
- **Token Importance Scorer (TIS)** selects the Top-M latents (M = 96) that matter for the current question.
- **Cross-Attention Routing Adapters (CARA)** inject the selected latents into a few LLM decoder layers (layers 1, 3, 5, 7) through gated cross-attention. The visual tokens never enter the LLM's self-attention sequence.

Compared with prior slide-level MLLMs, LoC-Path lowers inference TFLOPs by **97%** and inference memory by **39%** while remaining competitive on WSI-Bench, Slide-Bench and Path-Bench.

## Why this codebase

### An efficient data pipeline for long WSI sequences

In our experiments, the main training bottleneck was not GPU compute but moving tile features from disk to the GPU. The pipeline is built around three ideas:

1. **Memory-mapped `.npy` features to cut read I/O.** Tile features and coordinates are stored as separate `.npy` files. Features may be stored in `float16` or `float32`; training uses mixed precision. The datasets open them with `np.load(..., mmap_mode='r')` (`use_memmap=True`). Only the rows a sample actually uses are read from disk. When a slide has more tiles than the token budget, the sampler takes evenly spaced indices (`linspace`) from the memory map instead of loading the whole array. Compared with the common `.pt` / `.h5` loaders, this keeps the data loader from stalling on multi-GPU training.
2. **Batching variable-length slides.** Slides have very different numbers of tiles. The collate functions (`collate_feature_tokens`, `slidechat_feature_collate_fn`) pad each batch to its longest slide (capped at 61,440 tokens) and return a padding mask (`vision_pad_mask`) that every visual module respects. Text is padded separately with its own attention mask.
3. **Per-GPU batch size larger than 1.** Conventional slide-level pathology loaders process one slide per step. Because of (1) and (2), LoC-Path trains with a per-GPU batch of 2 or more slides and uses gradient accumulation for large global batches. This cuts the number of data-loading rounds.

Together, these changes brought Stage-2 training from more than a day down to under 20 hours on our machines, and gave roughly 2–3× wall-clock speedups in the other stages.

### Every stage fits in 24 GB per GPU

Peak memory per GPU, measured on NVIDIA RTX 6000 Ada:

| Stage | What is trained | Per-GPU batch × grad. accum. | Peak memory / GPU |
|---|---|---|---|
| MAE resampler pretraining | resampler + light MAE decoder | 2 × 4 | **9.2 GB** |
| Stage 1 (alignment) | projector + TIS + CARA | 2 × 64 | **12.4 GB** |
| Stage 2 (instruction tuning) | + LoRA on the LLM, last resampler layer | 2 × 16 | **16.8 GB** |

Here is why each stage stays small.

**MAE resampler pretraining (no LLM).**
- The model is small: 2 resampler layers with width 512, 256 latents, and a 2-layer cross-attention decoder that reconstructs tile features, not pixels.
- The context encoder uses LongNet dilated attention (segments 1024/2048/4096). Its cost grows roughly linearly with the number of tiles, instead of quadratically with full self-attention over 60k tokens.
- Each latent attends only to its Top-K tiles (K annealed from (128, 64) to (64, 32)).
- 70% of tile tokens are masked. The decoder only queries the masked positions against the 256 latents.
- Training runs in bf16 mixed precision; the feature loader accepts fp16 and fp32 NPY arrays.

**Stage 1: alignment.**
- **The LLM is frozen and loaded in 4-bit NF4.** Qwen2.5-7B-Instruct is quantized with bitsandbytes, so its weights take a few GB. It has no gradients and no optimizer state.
- **The resampler is frozen.** Only the projector, TIS and CARA are trained, so the optimizer state is tiny.
- **Visual tokens are not part of the LLM sequence.** The LLM attends over text only (≤ 2,280 tokens). Vision enters through cross-attention in 4 of the decoder layers, against at most M routed latents. A LLaVA-style model would instead append thousands of visual tokens to every self-attention layer.
- **The slide is compressed before it reaches the LLM:** STM merges neighboring tiles, then the resampler compresses them to 256 latents, then TIS routes the Top-M of those.

**Stage 2: instruction tuning.**
- **QLoRA-style tuning.** LoRA adapters (r = 128, α = 64) are added to the linear layers of the same frozen 4-bit LLM, so only the adapters carry gradients and optimizer state.
- **Small trainable visual part.** Only the last resampler layer, the latent queries and the feature input projection are unfrozen, together with the projector, TIS and CARA.
- **FlashAttention-2** is used for the LLM self-attention.
- **DeepSpeed ZeRO-2** shards gradients and optimizer states across GPUs.
- **The same compression path** (STM → 256 latents → Top-M) keeps the visual part of the activation memory fixed, whatever the slide size.

## Installation

The code was developed and tested on Linux with CUDA GPUs, Python 3.11, PyTorch 2.4.1 (CUDA 12.1) and flash-attn 2.8.3. Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`.

```bash
git clone https://github.com/WinstonHuTiger/loc_path.git
cd loc_path
uv sync --locked        # creates .venv and installs exactly uv.lock
# uv sync --locked --extra wandb # optional: Weights & Biases logging
```

Use a CUDA-capable NVIDIA GPU, an NVIDIA driver compatible with the CUDA 12.1
PyTorch wheel, and a C++ toolchain. For a source build of FlashAttention, install
a CUDA toolkit compatible with that PyTorch build and set `CUDA_HOME` if needed.
The validation machine used an RTX 6000 Ada; other devices were not tested.
`uv` selects Python 3.11 from `requires-python`; Python 3.12+ is not supported by
this lockfile. `setuptools` is also a runtime dependency of Triton 3.0.

`flash-attn` is built against the locked PyTorch during `uv sync`. It uses a matching prebuilt wheel when one exists; otherwise it compiles from source, which needs a CUDA toolkit (`nvcc`) and can take a while.

You can run every command below through `uv run ...` without activating `.venv`. The `xtuner` command comes from this repository.

METEOR (used in report-generation evaluation) needs NLTK's WordNet data, which is downloaded once:

```bash
uv run python -m nltk.downloader wordnet omw-1.4
```

## Data

Prepare WSI features with either of the following pipelines, then use the
resulting CONCH-v1 embeddings as input to LoC-Path:

| Pipeline | Processing flow | Settings used in the parent repository |
|---|---|---|
| [DSMIL](https://github.com/binli123/dsmil-wsi) | WSI → JPEG tiles → CONCH-v1 feature extraction | 224 × 224 tiles at 20×, no overlap, edge-response threshold 15 |
| [TRIDENT](https://github.com/mahmoodlab/TRIDENT) | WSI → tissue segmentation → patch coordinates → CONCH-v1 feature extraction | HEST segmentation, 224 × 224 tiles at 20×, overlap 0 |

The parent repository's DSMIL route uses its modified
`dsmil-wsi/deepzoom_tiler_new.py` for tiling and
`CLAM/extract_features_fp_new.py` for encoding. TRIDENT provides an integrated
segmentation/patching/encoding workflow. These routes select tissue differently;
keep the chosen preprocessing protocol consistent across train and test.
Readers can prepare their own data with these tools; preprocessing environments
and model weights are separate from this training package.

Use **CONCH v1** with the contrastive projection and output normalization off to
obtain `[N, 512]` tile features. Save H5 datasets named `features` and `coords`;
for training, convert them to separate `float16` or `float32` `*_features.npy` and `float32`
`*_coords.npy` files. Normalize coordinates to zero-anchored tile-grid indices
in `(x, y)` order and preserve feature/coordinate row alignment. The DSMIL/CLAM
route emits scaled tile-grid coordinates, while TRIDENT emits level-0 pixel
coordinates, so account for their coordinate scales during conversion.

See [the prepared-data format](docs/data_format.md) for the directory layout,
report/VQA JSON schema and benchmark task-file naming rules.

Prepared **WSI-Bench and BCNB** annotations and features are distributed in
[WinstonHu/LoC-Path-Data](https://huggingface.co/datasets/WinstonHu/LoC-Path-Data).
The WSI-Bench task partitions are inherited from **WSI-LLaVA/WSI-Bench** through
the parent repository's data preparation. This release retains those partitions
and filters out annotations with unavailable features. The dataset card records
the published counts and source terms.

The data repository contains **processed annotation JSON, pre-extracted CONCH-v1
features and coordinates**. Raw WSI images, image patches and original pathology
reports are not included. Existing NPY arrays retain their stored precision;
model computation follows the training configuration. Download the dataset with
the Hugging Face client installed by this package:

```bash
export LOCPATH_DATA_ROOT=$PWD/data/LoC-Path-Data
uv run --locked python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="WinstonHu/LoC-Path-Data",
    repo_type="dataset",
    revision="v1.0",
    local_dir=os.environ["LOCPATH_DATA_ROOT"],
)
PY

export LOCPATH_FEATURE_ROOT="$LOCPATH_DATA_ROOT/features/conch_v1"
export LOCPATH_STAGE1_REPORT_JSON="$LOCPATH_DATA_ROOT/annotations/train/stage1_reports.json"
export LOCPATH_STAGE1_DATA_PATH="$LOCPATH_STAGE1_REPORT_JSON"
export LOCPATH_STAGE2_DATA_PATH="$LOCPATH_DATA_ROOT/annotations/train/stage2_instructions.json"
```

These JSON files already exclude unavailable training features, so no exclusion
CSV is needed for the downloaded release. Check `metadata/dataset_summary.json`
for the feature download size and `metadata/feature_manifest.json` for per-file
SHA256 hashes. The dataset's source-data and CONCH terms apply separately from
this code package's license.

Every launcher reads its paths from environment variables:

| Variable | Used by | Meaning |
|---|---|---|
| `LOCPATH_FEATURE_ROOT` | all stages | root of the pre-extracted tile features (`.npy`) |
| `LOCPATH_UNWANTED_PREFIX_CSV` | all stages (optional) | CSV of slide prefixes to skip |
| `LOCPATH_STAGE1_REPORT_JSON` | MAE pretraining | report JSON that defines the pretraining slide set |
| `LOCPATH_RESAMPLER_CKPT` | Stage 1, Stage 2 | MAE-pretrained resampler checkpoint |
| `LOCPATH_STAGE1_DATA_PATH` | Stage 1 | Stage-1 (report) training JSON |
| `LOCPATH_STAGE2_DATA_PATH` | Stage 2 | Stage-2 (VQA) training JSON |
| `LOCPATH_STAGE2_PRETRAINED_PTH` | Stage 2 | Stage-1 checkpoint to start from |

## Training

Training has three stages. Each stage has a launcher in `scripts/` and a config in `xtuner/configs/`. Multi-GPU runs are controlled by these variables:

```bash
export NPROC_PER_NODE=4          # GPUs on this node
export NNODES=1 NODE_RANK=0 ADDR=127.0.0.1 PORT=29500
export DEEPSPEED_CFG=$PWD/xtuner/configs/deepspeed/deepspeed_zero2.json   # ZeRO-2, as in the paper
```

Single-GPU training also works: set `NPROC_PER_NODE=1` and unset
`DEEPSPEED_CFG` for the plain PyTorch route. Select physical devices with
`CUDA_VISIBLE_DEVICES`; local ranks are assigned within that visible set.

All three stage launchers forward extra training arguments. For example:

```bash
uv run --locked bash scripts/train_stage1.sh --seed 2025
```

Training starts a new run by default. To resume deliberately, pass
`--resume /path/to/checkpoint` or `--resume auto` (reads `WORK_DIR/last_checkpoint`).
Use a new `WORK_DIR` for a fresh experiment. Checkpoints may be regular `.pth`
files or DeepSpeed checkpoint directories ending in `.pth`; keep the entire
DeepSpeed directory for resuming training. For ZeRO-1/2, this release also writes
`locpath_model.pth` inside each checkpoint directory. That file contains the
LoRA and complete visual state, including frozen visual parameters, and can be
used alone for stage handoff or inference. The loader prefers it when given the
directory. Older ZeRO checkpoints without this export may omit frozen visual
weights. ZeRO-3 portable export is not supported by this release.
A failed training process returns a nonzero exit status,
so sequential training stops if Stage 1 fails.

Effective global batch = `NPROC_PER_NODE × NNODES × per-GPU batch × accumulative counts`. If you use a different number of GPUs, change `ACCUMULATIVE_COUNTS` in the config to keep the same global batch.

### Stage 0: MAE resampler pretraining

The resampler learns to compress tile features into 256 latents by reconstructing masked tiles. It uses a 70% mask ratio, runs for 100k iterations with AdamW (lr 1e-4, 3% warmup, cosine schedule), and anneals Top-K from (128, 64) to (64, 32). A coverage term keeps every tile within some latent's attention, and a feature-diversity term keeps the latents from collapsing onto each other.

```bash
export LOCPATH_FEATURE_ROOT=/path/to/features/conch_v1
export LOCPATH_STAGE1_REPORT_JSON=/path/to/report.json
export WORK_DIR=$PWD/work_dirs/mae_resampler

uv run bash scripts/train_resampler_pretrain.sh
```

Config: `xtuner/configs/mae_pretraining/mae_feature_pretrain_longnet_ctx_encoder.py`

### Stage 1: alignment on reports

The LLM and resampler are frozen. The projector, TIS and CARA are trained on WSI-Bench report data for 3 epochs (lr 1e-4). TIS learns from CARA's cross-attention through self-distillation.

```bash
export LOCPATH_FEATURE_ROOT=/path/to/features/conch_v1
export LOCPATH_RESAMPLER_CKPT=/path/to/mae_resampler.pth
export LOCPATH_STAGE1_DATA_PATH=/path/to/stage1_dataset.json
export WORK_DIR=$PWD/work_dirs/stage1

uv run bash scripts/train_stage1.sh
```

Config: `xtuner/configs/vis_adapter/stage_1_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl.py`

### Stage 2: instruction tuning on VQA

This stage adds LoRA to the LLM and unfreezes the last resampler layer, the latent queries and the input projection. It trains on all WSI-Bench VQA pairs (lr 2e-5) with an extra pairwise ranking loss for TIS. `RouterAnnealHook` anneals the routing budget and the TIS temperatures, with milestones at 7k and 14k iterations; the budget ends at M = 96.

```bash
export LOCPATH_FEATURE_ROOT=/path/to/features/conch_v1
export LOCPATH_RESAMPLER_CKPT=/path/to/mae_resampler.pth
export LOCPATH_STAGE2_DATA_PATH=/path/to/stage2_dataset.json
export LOCPATH_STAGE2_PRETRAINED_PTH=/path/to/stage1_checkpoint.pth
export WORK_DIR=$PWD/work_dirs/stage2

uv run bash scripts/train_stage2.sh
```

Config: `xtuner/configs/vis_adapter/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass.py`

To run Stage 1 and then Stage 2 automatically (Stage 2 starts from the newest Stage-1 checkpoint):

```bash
export STAGE1_WORK_DIR=$PWD/work_dirs/stage1
export STAGE2_WORK_DIR=$PWD/work_dirs/stage2
uv run bash scripts/train_stage1_stage2.sh
```

## Inference

`scripts/test_vis_adapter.sh` runs distributed generation over every `*_test.json` file in a folder and writes one prediction per question.

```bash
export CHECKPOINT=/path/to/stage2_checkpoint.pth
export DATA_DIR=/path/to/benchmark_json_dir      # folder with *_test.json
export FEATURE_ROOT=/path/to/features/conch_v1
export OUT_CSV=$PWD/results/predictions.csv
export OUT_JSONL=$PWD/results/predictions.jsonl
export NPROC_PER_NODE=4

uv run bash scripts/test_vis_adapter.sh --sample_num 61440 --max_new_tokens 512
```

For the downloaded data, use the NPY inference backend:

```bash
export DATA_DIR="$LOCPATH_DATA_ROOT/annotations/test/wsi_bench"
export FEATURE_ROOT="$LOCPATH_FEATURE_ROOT"
uv run --locked bash scripts/test_vis_adapter.sh \
  --image_feature_suffix .npy --sample_num 61440 --router-top-m 96
```

To evaluate BCNB, set `DATA_DIR` to
`$LOCPATH_DATA_ROOT/annotations/test/external` and use the same command. BCNB
records point to the included numeric CSV features relative to `FEATURE_ROOT`.

For deployment, provide `CHECKPOINT`, `DATA_DIR` and `FEATURE_ROOT` as above.
The full Stage-2 checkpoint contains the visual weights; historical MAE and
Stage-1 checkpoint paths are not required during inference. The Qwen base model
is still required, because training checkpoints omit its frozen weights. To
use an already downloaded base model, set `LOCPATH_LLM_PATH=/path/to/Qwen2.5-7B-Instruct`
for both training and inference. Otherwise Hugging Face downloads the default
`Qwen/Qwen2.5-7B-Instruct` when first needed.

The router schedule is not stored in the checkpoint; reproduce the final paper
routing budget explicitly with `--router-top-m 96` during inference.

Useful options: `--sample_num` (maximum tiles per slide), `--sample_strategy {linspace,random}`, `--router-top-m` (override the routing budget M) and `--decoding` (greedy by default). Set `CONFIG=...` to run inference with another config.

## Evaluation

`scripts/evaluate_predictions.sh` scores the prediction JSONL by task (morphological analysis, diagnosis, treatment planning, report generation, plus Slide-Bench/BCNB and Path-Bench):

- **Closed-ended questions** (multiple choice / true-false) are scored by accuracy. No LLM is needed.
- **Open-ended questions** are scored with **WSI-Precision** and **WSI-Relevance**, the LLM-as-judge metrics from WSI-LLaVA. The script uses their claim-extraction and scoring prompts unchanged.
- **Report generation** additionally reports BLEU-1/2, ROUGE-L and METEOR.

The paper uses **GPT-4o** as the judge, with the API's default sampling settings:

```bash
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=your_api_key

uv run bash scripts/evaluate_predictions.sh \
  --input "$OUT_JSONL" \
  --out-dir "$PWD/results/eval" \
  --provider openai --model gpt-4o \
  --max-workers 4
```

Any OpenAI-compatible server (for example, a local Qwen3 served with vLLM) also works:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

uv run bash scripts/evaluate_predictions.sh \
  --input "$OUT_JSONL" \
  --out-dir "$PWD/results/eval_qwen3" \
  --provider openai-compatible --model <served-model-name> \
  --max-workers 4
```

To compute only the report-generation metrics without any LLM calls, add `--report-metric`. To compute the report metrics and the LLM-judge scores for report rows only, add `--report-metric-llm`. Outputs are `results.jsonl` (per item) and `summary.json` (per task and per file) in `--out-dir`. Other options: `--dry-run`, `--limit N`, `--sample N` and `--gt-claims-cache-dir DIR`, which caches the extracted ground-truth claims across runs.

## Test the installation and training pipeline

Run these commands from the repository root, after `uv sync --locked`:

```bash
uv pip check
uv run --locked xtuner train --help
uv run --locked python -m unittest discover -s tests -v
```

The unit tests load and serialize all three configs, check visual sampling
budgets, exercise launcher argument forwarding and path aliases, and ensure
Stage-1 failure stops Stage 2. They do not download model weights.

Then run the CUDA integration test on one free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python tools/smoke_test.py \
  --work-dir work_dirs/smoke_tiny
```

This generates two unequal-length feature arrays (NPY and H5), grid coordinates,
training/test JSON and a tiny random eight-layer Qwen model/tokenizer. It runs
two iterations each of **MAE → Stage 1 → Stage 2**, with the released 512-wide,
256-latent LongNet resampler, NF4 loading, bf16, FlashAttention and Stage-2 LoRA.
Only the LLM size, visual input cap (1,024), workers, gradient accumulation and
run length are reduced. It checks finite losses, checkpoint creation, Stage-2
parameter updates and LoRA weights, then generates two predictions from the
Stage-2 checkpoint and runs closed-ended evaluation without API calls.
`smoke_result.json` is written only after all checks pass. Logs and checkpoints
are kept under the selected work directory. Always use a **new/empty** directory
for each run.

To exercise the actual Qwen2.5-7B base model instead of the tiny random model:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python tools/smoke_test.py \
  --work-dir work_dirs/smoke_qwen7b \
  --llm-path Qwen/Qwen2.5-7B-Instruct
```

A local Hugging Face model directory is also accepted by `--llm-path`.
To verify the two-GPU DeepSpeed ZeRO-2 route:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked python tools/smoke_test.py \
  --work-dir work_dirs/smoke_zero2 --nproc 2 --deepspeed
```

These are execution tests on generated data, not accuracy measurements or full
training reproductions. They do not validate the paper's results, long-run
convergence, full 61,440-tile batches, raw-WSI extraction or external LLM judges.

## Repository layout

```
scripts/                     launchers for pretraining, Stage 1/2, inference, evaluation
tools/smoke_test.py           self-contained CUDA integration test
tests/                       config and launcher regression tests
docs/data_format.md          prepared JSON / feature contract
data_engine/                 LLM-as-judge and report-metric evaluation
xtuner/configs/              MAE pretraining, Stage 1, Stage 2 and DeepSpeed configs
xtuner/model/                LoC-Path model (llava_vis_adapter.py), MAE pretraining,
                             resampler / TIS / CARA modules, vendored LongNet (torchscale)
xtuner/dataset/              memory-mapped feature datasets and variable-length collate
xtuner/engine/hooks/         Top-K / router annealing, evaluation and logging hooks
xtuner/tools/                training entry point and distributed inference
```

## Acknowledgements

This project builds on the following works. We thank the authors for releasing their code and data.

- [XTuner](https://github.com/InternLM/xtuner): the training framework this codebase is built on.
- [SlideChat](https://github.com/uni-medical/SlideChat): slide-level pathology MLLM and the Slide-Bench benchmark. Our slide-feature training setup builds on it.
- [WSI-LLaVA](https://github.com/XinhengLyu/WSI-LLaVA): the WSI-Bench dataset and the WSI-Precision / WSI-Relevance evaluation protocol.
- [MAE](https://github.com/facebookresearch/mae): the masked-autoencoder pretraining that our resampler pretraining follows.

The LongNet implementation under `xtuner/model/torchscale/` is adapted from [TorchScale](https://github.com/microsoft/torchscale) (MIT License).

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attributions.

## Citation

```bibtex
@misc{hu2026locpathlearningcompresspathology,
      title={LoC-Path: Learning to Compress for Pathology Multimodal Large Language Models},
      author={Qingqiao Hu and Weimin Lyu and Meilong Xu and Kehan Qi and Xiaoling Hu and Saumya Gupta and Jiawei Zhou and Chao Chen},
      year={2026},
      eprint={2512.05391},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.05391},
}
```
