#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_WORK_DIR="${STAGE1_WORK_DIR:-$REPO_ROOT/work_dirs/stage_1_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl}"
STAGE2_WORK_DIR="${STAGE2_WORK_DIR:-$REPO_ROOT/work_dirs/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass}"

: "${LOCPATH_RESAMPLER_CKPT:=${SLIDECHAT_RESAMPLER_CKPT:-}}"
: "${LOCPATH_FEATURE_ROOT:=${SLIDECHAT_FEATURE_ROOT:-}}"
: "${LOCPATH_STAGE1_DATA_PATH:=${SLIDECHAT_STAGE1_DATA_PATH:-}}"
: "${LOCPATH_STAGE2_DATA_PATH:=${SLIDECHAT_STAGE2_DATA_PATH:-}}"
: "${LOCPATH_RESAMPLER_CKPT:?Set LOCPATH_RESAMPLER_CKPT to the resampler checkpoint.}"
: "${LOCPATH_FEATURE_ROOT:?Set LOCPATH_FEATURE_ROOT to the slide feature root.}"
: "${LOCPATH_STAGE1_DATA_PATH:?Set LOCPATH_STAGE1_DATA_PATH to the stage-1 training json.}"
: "${LOCPATH_STAGE2_DATA_PATH:?Set LOCPATH_STAGE2_DATA_PATH to the stage-2 training json.}"

export LOCPATH_RESAMPLER_CKPT LOCPATH_FEATURE_ROOT
export LOCPATH_STAGE1_DATA_PATH LOCPATH_STAGE2_DATA_PATH

find_latest_pth() {
  local search_dir="$1"
  local latest_file=""
  local latest_mtime=0
  local candidate

  for candidate in "$search_dir"/*.pth; do
    [[ -e "$candidate" ]] || continue
    local mtime
    mtime=$(stat -c %Y "$candidate")
    if [[ -z "$latest_file" || $mtime -gt $latest_mtime ]]; then
      latest_mtime=$mtime
      latest_file="$candidate"
    fi
  done

  printf '%s' "$latest_file"
}

WORK_DIR="$STAGE1_WORK_DIR" \
CONFIG="$REPO_ROOT/xtuner/configs/vis_adapter/stage_1_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl.py" \
DEEPSPEED_CFG="${DEEPSPEED_CFG:-}" \
bash "$REPO_ROOT/scripts/train_stage1.sh" "$@"

STAGE1_CKPT="$(find_latest_pth "$STAGE1_WORK_DIR")"
if [[ -z "$STAGE1_CKPT" ]]; then
  echo "Failed to locate a stage-1 checkpoint in $STAGE1_WORK_DIR" >&2
  exit 1
fi

LOCPATH_STAGE2_PRETRAINED_PTH="$STAGE1_CKPT" \
WORK_DIR="$STAGE2_WORK_DIR" \
CONFIG="$REPO_ROOT/xtuner/configs/vis_adapter/stage_2_vis_adapter_topk_anneal_longnet_ctx_language_add_router_kl_mass.py" \
DEEPSPEED_CFG="${DEEPSPEED_CFG:-}" \
bash "$REPO_ROOT/scripts/train_stage2.sh" "$@"
