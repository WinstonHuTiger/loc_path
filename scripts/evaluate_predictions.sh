#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$REPO_ROOT/data_engine/evaluate_model_output_by_task2.py" "$@"
