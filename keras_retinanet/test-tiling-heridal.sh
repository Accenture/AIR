#!/bin/bash
# you might wanna specify WANDB env variables here
# export WANDB_MODE="dryrun"

# fixes some errors (maybe)
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

wandb on
wandb agent --count 1 hypersphere/masters-thesis/0qwss4eh
# python3 $WANDB_DIR/keras_retinanet/bin/evaluate.py "$@"
