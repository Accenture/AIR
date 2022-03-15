#!/bin/bash
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo 'Usage: bash '$(basename $0)' [BBA_MODE="enclose"] [MODEL="dauntless-sweep-2_resnet152_pascal.h5"] [IMAGE_FOLDER="data/images/test"]'
  exit 0
fi

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

BBA_MODE="enclose"
MODEL="dauntless-sweep-2_resnet152_pascal.h5"
IMAGE_FOLDER="${PWD}/data/images/test"

BBA_MODE="${1:-$BBA_MODE}"
MODEL="${2:-$MODEL}"
DATASET="${3:-$DATASET}"

python3 keras_retinanet/keras_retinanet/bin/infer.py \
    --gpu 0 \
    --backbone resnet152 \
    --image_min_side 1525 \
    --image_max_side 2025 \
    --score_threshold 0.05 \
    --max_detections 100000 \
    --nms_threshold 0.25 \
    --config $PWD/keras_retinanet/config.ini \
    --anchor_scale 0.965 \
    --convert_model true \
    --image_tiling_dim 4 \
    --nms_mode $BBA_MODE \
    --model $PWD/models/$MODEL \
    $IMAGE_FOLDER \
    $PWD/data/predictions/${MODEL/.h5/}-$BBA_MODE-inference