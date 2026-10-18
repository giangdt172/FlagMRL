#!/usr/bin/env bash
set -euo pipefail

# Usage: NPROC_PER_NODE=2 bash scripts/train/fastvlm_flagmrl_cls.sh --image_dir /data/MMEB-train
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec "${PYTHON:-python}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NPROC_PER_NODE:-1}" \
    train_ddp_one_model.py \
    --model_name "${MODEL_NAME:-apple/FastVLM-0.5B}" \
    --model_backbone llava_qwen2 \
    --lora --lora_r 64 --lora_alpha 128 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --pooling eos --normalize True --temperature 0.02 \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 \
    --dataset_split original \
    --image_dir "${TRAIN_IMAGE_DIR:-data/MMEB-train}" \
    --output_dir "${OUTPUT_DIR:-training/FlagMRL_FastVLM_cls}" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --kd_loss_type flag_mrl \
    --nested_dims 64 128 256 512 768 896 \
    --flag_init random --flag_lr 0.1 --flag_eps 1e-4 \
    --flag_retraction svd --flag_update_every 1 \
    --flag_use_neg_mask True --flag_far_mode uniform \
    --optimizer_name adamw --learning_rate 5e-5 \
    --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --num_train_epochs 1 --weight_decay 0.01 \
    --bf16 --image_resolution mid \
    --save_strategy epoch --logging_steps 1 --seed 42 \
    --report_to none \
    "$@"
