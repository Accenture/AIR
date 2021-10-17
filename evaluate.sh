#!/bin/bash
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo 'Usage: bash '$(basename $0)' [EVAL_MODE="sar-apd"] [BBA_MODE="enclose"] [MODEL="dauntless-sweep-2_resnet152_pascal.h5"] [DATASET="data/datasets/heridal_keras_retinanet_voc"]'
  exit 0
fi

export WANDB_MODE="disabled"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

EVAL_MODE="sar-apd"
BBA_MODE="enclose"
MODEL="dauntless-sweep-2_resnet152_pascal.h5"
DATASET="${PWD}/data/datasets/heridal_keras_retinanet_voc"

EVAL_MODE="${1:-$EVAL_MODE}"
BBA_MODE="${2:-$BBA_MODE}"
MODEL="${3:-$MODEL}"
DATASET="${4:-$DATASET}"

python3 keras_retinanet/keras_retinanet/bin/evaluate.py \
    --backbone resnet152 \
    --image_min_side 1525 \
    --image_max_side 2025 \
    --score_threshold 0.05 \
    --max_detections 100000 \
    --iou_threshold 0.5 \
    --nms_threshold 0.25 \
    --config $PWD/keras_retinanet/config.ini \
    --anchor_scale 0.965 \
    --save_path $PWD/data/predictions/${MODEL/.h5/}-$BBA_MODE-$EVAL_MODE-eval \
    --convert_model true \
    --image_tiling_dim 4 \
    --nms_mode $BBA_MODE \
    --eval_mode $EVAL_MODE \
    --set_name test \
    --profile \
    --model $PWD/models/$MODEL \
    pascal \
    $DATASET