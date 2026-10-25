#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mkdir -p evaluation
eval_run_dir="$(mktemp -d evaluation/cls-run.XXXXXXXXXX)"

# Qwen3VL: train + eval CLS (1 epoch -> checkpoint-epoch-0)
qwen3_model="${QWEN3_MODEL_NAME:-Qwen/Qwen3-VL-Embedding-2B}"
qwen3_output="${QWEN3_OUTPUT_DIR:-training/FlagMRL_Qwen3_2B_cls}"
MODEL_NAME="$qwen3_model" OUTPUT_DIR="$qwen3_output" \
    bash scripts/train/qwen3vl_flagmrl_cls.sh
MODEL_NAME="$qwen3_model" CHECKPOINT_PATH="$qwen3_output/checkpoint-epoch-0" \
ENCODE_OUTPUT_PATH="$eval_run_dir/qwen3vl" \
    bash scripts/eval/qwen3vl_flagmrl_cls.sh

# Qwen2VL: train + eval CLS
qwen2_model="${QWEN2_MODEL_NAME:-Qwen/Qwen2-VL-2B}"
qwen2_output="${QWEN2_OUTPUT_DIR:-training/FlagMRL_B3_Qwen2_2B_cls}"
MODEL_NAME="$qwen2_model" OUTPUT_DIR="$qwen2_output" \
    bash scripts/train/b3_qwen2_2b_flagmrl_cls.sh
MODEL_NAME="$qwen2_model" CHECKPOINT_PATH="$qwen2_output/checkpoint-epoch-0" \
ENCODE_OUTPUT_PATH="$eval_run_dir/qwen2vl" \
    bash scripts/eval/b3_qwen2_2b_flagmrl_cls.sh
