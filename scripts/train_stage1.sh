#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/xtuner/configs/vis_adapter/stage_1_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl.py}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs/stage_1_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl}"

: "${LOCPATH_RESAMPLER_CKPT:=${SLIDECHAT_RESAMPLER_CKPT:-}}"
: "${LOCPATH_FEATURE_ROOT:=${SLIDECHAT_FEATURE_ROOT:-}}"
: "${LOCPATH_STAGE1_DATA_PATH:=${SLIDECHAT_STAGE1_DATA_PATH:-}}"
: "${LOCPATH_RESAMPLER_CKPT:?Set LOCPATH_RESAMPLER_CKPT to the resampler checkpoint.}"
: "${LOCPATH_FEATURE_ROOT:?Set LOCPATH_FEATURE_ROOT to the slide feature root.}"
: "${LOCPATH_STAGE1_DATA_PATH:?Set LOCPATH_STAGE1_DATA_PATH to the stage-1 training json.}"

export LOCPATH_RESAMPLER_CKPT LOCPATH_FEATURE_ROOT LOCPATH_STAGE1_DATA_PATH

ARGS=("$CONFIG" --work-dir "$WORK_DIR" --local_rank 0)
if [[ -n "${DEEPSPEED_CFG:-}" ]]; then
  ARGS+=(--deepspeed "$DEEPSPEED_CFG")
fi

xtuner train "${ARGS[@]}" "$@"
