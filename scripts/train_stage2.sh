#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/xtuner/configs/vis_adapter/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass.py}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass}"

: "${LOCPATH_RESAMPLER_CKPT:=${SLIDECHAT_RESAMPLER_CKPT:-}}"
: "${LOCPATH_FEATURE_ROOT:=${SLIDECHAT_FEATURE_ROOT:-}}"
: "${LOCPATH_STAGE2_DATA_PATH:=${SLIDECHAT_STAGE2_DATA_PATH:-}}"
: "${LOCPATH_STAGE2_PRETRAINED_PTH:=${SLIDECHAT_STAGE2_PRETRAINED_PTH:-}}"
: "${LOCPATH_RESAMPLER_CKPT:?Set LOCPATH_RESAMPLER_CKPT to the resampler checkpoint.}"
: "${LOCPATH_FEATURE_ROOT:?Set LOCPATH_FEATURE_ROOT to the slide feature root.}"
: "${LOCPATH_STAGE2_DATA_PATH:?Set LOCPATH_STAGE2_DATA_PATH to the stage-2 training json.}"
: "${LOCPATH_STAGE2_PRETRAINED_PTH:?Set LOCPATH_STAGE2_PRETRAINED_PTH to the stage-1 checkpoint.}"

export LOCPATH_RESAMPLER_CKPT LOCPATH_FEATURE_ROOT LOCPATH_STAGE2_DATA_PATH LOCPATH_STAGE2_PRETRAINED_PTH

ARGS=("$CONFIG" --work-dir "$WORK_DIR" --local_rank 0)
if [[ -n "${DEEPSPEED_CFG:-}" ]]; then
  ARGS+=(--deepspeed "$DEEPSPEED_CFG")
fi

xtuner train "${ARGS[@]}" "$@"
