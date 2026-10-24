#!/usr/bin/env bash
set -euo pipefail

# Usage: bash command.sh
# Override MODEL_NAME, TRAIN_IMAGE_DIR, EVAL_IMAGE_DIR, or OUTPUT_DIR as needed.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export OUTPUT_DIR="${OUTPUT_DIR:-training/FlagMRL_B3_Qwen2_2B_cls}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoint-epoch-0}"

echo "Starting training..."
bash scripts/train/b3_qwen2_2b_flagmrl_cls.sh

echo "Training completed. Evaluating ${CHECKPOINT_PATH}..."
bash scripts/eval/b3_qwen2_2b_flagmrl_cls.sh
