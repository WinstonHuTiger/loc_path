#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/xtuner/configs/vis_adapter/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass.py}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the model checkpoint.}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the folder containing *_test.json files.}"
FEATURE_ROOT="${FEATURE_ROOT:?Set FEATURE_ROOT to the slide feature root.}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results/test_predictions.csv}"
OUT_JSONL="${OUT_JSONL:-$REPO_ROOT/results/test_predictions.jsonl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun \
  --nproc_per_node "$NPROC_PER_NODE" \
  "$REPO_ROOT/xtuner/tools/test_vis_adapter_distributed.py" \
  "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --feature_root "$FEATURE_ROOT" \
  --out_csv "$OUT_CSV" \
  --out_jsonl "$OUT_JSONL" \
  --launcher pytorch \
  "$@"
