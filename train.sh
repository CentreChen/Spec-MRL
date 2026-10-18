#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Usage: MODEL_PATH=... TRAIN_DATA=... bash train.sh [SpecMRL|rawMRL]
method="${1:-SpecMRL}"
case "$method" in
    SpecMRL|rawMRL) ;;
    *) echo "Method must be SpecMRL or rawMRL"; exit 1 ;;
esac
: "${MODEL_PATH:?Set MODEL_PATH to the pretrained model directory}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the training data path}"
export PYTHONDONTWRITEBYTECODE=1

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" -m train.run \
    --method "$method" \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --output_dir "./trained_models/$method" \
    --gc_chunk_size 64 \
    --ddp_find_unused_parameters False \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --bf16 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --save_only_model True \
    --per_device_train_batch_size 96 \
    --sentence_pooling_method mean \
    --temperature 0.01 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --train_group_size 8 \
    --report_to tensorboard \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers True
