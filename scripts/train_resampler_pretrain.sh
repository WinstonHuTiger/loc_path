#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/xtuner/configs/mae_pretraining/mae_feature_pretrain_longnet_ctx_encoder.py}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs/mae_feature_pretrain_longnet_ctx_encoder}"

: "${LOCPATH_FEATURE_ROOT:=${SLIDECHAT_FEATURE_ROOT:-}}"
: "${LOCPATH_STAGE1_REPORT_JSON:=${SLIDECHAT_STAGE1_REPORT_JSON:-}}"
: "${LOCPATH_FEATURE_ROOT:?Set LOCPATH_FEATURE_ROOT to the slide feature root.}"
: "${LOCPATH_STAGE1_REPORT_JSON:?Set LOCPATH_STAGE1_REPORT_JSON to the stage-1 report json.}"

export LOCPATH_FEATURE_ROOT LOCPATH_STAGE1_REPORT_JSON

ARGS=("$CONFIG" --work-dir "$WORK_DIR" --local_rank 0)
if [[ -n "${DEEPSPEED_CFG:-}" ]]; then
  ARGS+=(--deepspeed "$DEEPSPEED_CFG")
fi

xtuner train "${ARGS[@]}" "$@"
