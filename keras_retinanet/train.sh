#!/bin/bash
# you might wanna specify WANDB env variables here
# export WANDB_MODE="dryrun"

# fixes some errors (maybe)
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

wandb on
# wandb agent --count 1 hypersphere/masters-thesis/07i915n0 # resnet152 noise tiled
# wandb agent --count 1 hypersphere/masters-thesis/oorn45y8 # effnetb4 signal tiled
# wandb agent --count 1 hypersphere/masters-thesis/k049oyjm # seresnet152 signal tiled
# wandb agent --count 1 hypersphere/masters-thesis/a9ih2l95 # seresnext101 signal tiled
# wandb agent --count 1 hypersphere/masters-thesis/mnq0aaen # resnet152 signal tiled
# hypersphere/masters-thesis/4bv7vqtt # resnet152 signal
# hypersphere/masters-thesis/wdfdztdm # resnet101 signal v3
# hypersphere/masters-thesis/zj2gcqap # resnet101 signal v2
# hypersphere/masters-thesis/cvfqrmz5 # resnet101 noise
# hypersphere/masters-thesis/ye2tgw50 # resnet101 signal
# hypersphere/masters-thesis/8jqncdkx # resnet50 noise
# hypersphere/masters-thesis/0mtdcpum # resnet50 signal
python3 $WANDB_DIR/keras_retinanet/bin/train.py "$@"
