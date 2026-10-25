#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval/b3_qwen2_2b_flagmrl_cls.sh --checkpoint_path /path/to/checkpoint-epoch-0 --image_dir /data/MMEB-eval
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec "${PYTHON:-python}" eval_mmeb.py \
    --model_name "${MODEL_NAME:-raghavlite/B3_Qwen2_2B}" \
    --model_backbone qwen2_vl \
    --checkpoint_path "${CHECKPOINT_PATH:-training/FlagMRL_B3_Qwen2_2B_cls/checkpoint-epoch-0}" \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos --normalize True \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
    --dataset_split test \
    --image_dir "${EVAL_IMAGE_DIR:-data/MMEB-eval}" \
    --encode_output_path "${ENCODE_OUTPUT_PATH:-evaluation/FlagMRL_B3_Qwen2_2B_cls/checkpoint-epoch-0}" \
    --per_device_eval_batch_size 8 \
    --bf16 --image_resolution "${IMAGE_RESOLUTION:-mid}" --tgt_prefix_mod \
    --wandb True \
    --run_name FlagMRL_B3_Qwen2_2B_cls \
    "$@"
