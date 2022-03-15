#!/bin/bash
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo 'Usage: bash '$(basename $0)' [DATASET="data/datasets/heridal_keras_retinanet_voc"] [EPOCHS=3] [BACKBONE="resnet152"]'
  exit 0
fi

export WANDB_MODE="disabled"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DATASET="${PWD}/data/datasets/heridal_keras_retinanet_voc"
EPOCHS=3
BACKBONE="resnet50"

DATASET="${1:-$DATASET}"
EPOCHS="${2:-$EPOCHS}"
BACKBONE="${3:-$BACKBONE}"
# use 5564 for real training (show all images per epoch), although it can take long time
STEPS=3

python3 keras_retinanet/keras_retinanet/bin/train.py \
    --gpu 0 \
    --random_transform=true \
    --anchor_scale=0.965 \
    --config=$PWD/keras_retinanet/config.ini \
    --early_stop_patience=10 \
    --image_max_side=1000 \
    --image_min_side=750 \
    --lr=0.00002196 \
    --seed=26203 \
    --reduce_lr_factor=0.33 \
    --reduce_lr_patience=4 \
    --no_resize=true \
    --compute_val_loss \
    --steps=$STEPS \
    --epochs=$EPOCHS \
    --backbone=$BACKBONE \
    --group=retinanet-train-model-selection \
    --tags=model-selection \
    --snapshot_interval=2 \
    pascal \
    $DATASET
