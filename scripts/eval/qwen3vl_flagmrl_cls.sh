#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec "${PYTHON:-python}" eval_mmeb.py \
    --model_name "${MODEL_NAME:-Qwen/Qwen3-VL-Embedding-2B}" \
    --model_backbone qwen3_vl \
    --checkpoint_path "${CHECKPOINT_PATH:-training/FlagMRL_Qwen3_2B_cls/checkpoint-epoch-1}" \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos --normalize True \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 \
    --dataset_split test --image_dir "${EVAL_IMAGE_DIR:-data/MMEB-eval}" \
    --encode_output_path "${ENCODE_OUTPUT_PATH:-evaluation/FlagMRL_Qwen3_2B_cls/checkpoint-epoch-1}" \
    --per_device_eval_batch_size 16 \
    --nested_dims 64 128 256 512 768 1024 2048 \
    --bf16 --image_resolution "${IMAGE_RESOLUTION:-mid}" --tgt_prefix_mod \
    --wandb True --run_name FlagMRL_Qwen3_2B_cls \
    "$@"
