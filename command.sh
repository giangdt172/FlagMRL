#!/usr/bin/env bash
set -euo pipefail

# Usage: bash command.sh
# Override MODEL_NAME, TRAIN_IMAGE_DIR, or EVAL_IMAGE_DIR as needed.
# Per-task paths: CLS_OUTPUT_DIR / VQA_OUTPUT_DIR and
# CLS_ENCODE_OUTPUT_PATH / VQA_ENCODE_OUTPUT_PATH.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

for task in cls vqa; do
    if [[ "$task" == cls ]]; then
        task_output_dir="${CLS_OUTPUT_DIR:-training/FlagMRL_B3_Qwen2_2B_cls}"
        task_encode_output_path="${CLS_ENCODE_OUTPUT_PATH:-evaluation/FlagMRL_B3_Qwen2_2B_cls/checkpoint-epoch-0}"
    else
        task_output_dir="${VQA_OUTPUT_DIR:-training/FlagMRL_B3_Qwen2_2B_vqa}"
        task_encode_output_path="${VQA_ENCODE_OUTPUT_PATH:-evaluation/FlagMRL_B3_Qwen2_2B_vqa/checkpoint-epoch-0}"
    fi
    task_checkpoint_path="${task_output_dir}/checkpoint-epoch-0"

    echo "Starting ${task} training..."
    OUTPUT_DIR="$task_output_dir" \
        bash "scripts/train/b3_qwen2_2b_flagmrl_${task}.sh"

    echo "${task} training completed. Evaluating ${task_checkpoint_path}..."
    CHECKPOINT_PATH="$task_checkpoint_path" \
        ENCODE_OUTPUT_PATH="$task_encode_output_path" \
        bash "scripts/eval/b3_qwen2_2b_flagmrl_${task}.sh"
done

echo "CLS and VQA training/evaluation completed."
