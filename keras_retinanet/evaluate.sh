#!/bin/bash
python3 keras_retinanet/bin/evaluate.py \
    --score-threshold 0.15 \
    --iou-threshold 0.05 \
    --max-detections 20 \
    --config config.ini \
    --convert-model \
    --save-path ../../data/images/$1 \
    pascal \
    ../../data/datasets/heridal_keras_retinanet_voc \
    snapshots/$1.h5
    

