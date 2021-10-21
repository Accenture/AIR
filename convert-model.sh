#!/bin/bash
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo 'Usage: bash '$(basename $0)' [BBA_MODE="mob"] [MODEL="dauntless-sweep-2_resnet152_pascal.h5"]'
  exit 0
fi

BBA_MODE="mob"
TRAINING_MODEL="dauntless-sweep-2_resnet152_pascal.h5"

BBA_MODE="${1:-$BBA_MODE}"
TRAINING_MODEL="${2:-$TRAINING_MODEL}"

MAX_DETS="1000"
NMS_THRES="0.25"
SCORE_THRES="0.05"
NO_NMS=""

if [[ "$BBA_MODE" == "mob" ]]; then
  MAX_DETS="100000"
  NO_NMS="--no-nms"
fi

python3 keras_retinanet/keras_retinanet/bin/convert_model.py $NO_NMS \
    --backbone resnet152 \
    --score-threshold $SCORE_THRES \
    --max-detections $MAX_DETS \
    --nms-threshold $NMS_THRES \
    --anchor-scale 0.965 \
    --config $PWD/keras_retinanet/config.ini \
    --anchor-scale 0.965 \
    $PWD/models/$TRAINING_MODEL \
    $PWD/models/${TRAINING_MODEL/.h5/}-$BBA_MODE-inference.h5