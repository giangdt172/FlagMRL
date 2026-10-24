#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train/fastvlm_flagmrl_cls.sh --image_dir /data/MMEB-train
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

torchrun \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_ddp_one_model.py \
    --model_name "${MODEL_NAME:-apple/FastVLM-0.5B}" \
    --model_backbone llava_qwen2 \
    --lora --lora_r 64 --lora_alpha 64 \
    --pooling eos --normalize True --temperature 0.02 \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 \
    --dataset_split original \
    --image_dir "${TRAIN_IMAGE_DIR:-data/MMEB-train}" \
    --output_dir "${OUTPUT_DIR:-training/FlagMRL_FastVLM_cls_3}" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --kd_loss_type "${KD_LOSS_TYPE:-flag_mrl}" \
    --nested_dims 64 128 256 512 768 896 \
    --flag_init random --flag_lr 0.1 --flag_eps 1e-4 \
    --flag_retraction svd --flag_update_every 1 \
    --flag_lambda_neg 1.0 --flag_use_neg_mask False --flag_far_mode uniform \
    --flag_lambda_cross "${FLAG_LAMBDA_CROSS:-0.0}" \
    --optimizer_name adamw --learning_rate 5e-5 \
    --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --num_train_epochs 1 --weight_decay 0.01 \
    --bf16 --image_resolution "${IMAGE_RESOLUTION:-448}" \
    --save_strategy epoch --logging_steps 1 --seed 42 \
    --wandb True \
    --run_name FlagMRL_FastVLM_cls_3 \
    "$@"
