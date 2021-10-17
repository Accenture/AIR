#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --time=0-01:00:00
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tmp=30G
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=mob-tiling-perf
#SBATCH --output=/scratch/cs/sar-uav-cv/masters-thesis/computer_vision/keras-retinanet/slurm_out/%j-%x.out
#SBATCH --partition=gpushort

export SINGULARITY_CACHEDIR=/scratch/cs/sar-uav-cv/.singularity

TRAIN_RUN_NAME="dauntless-sweep-2"
BACKBONE="resnet152"
DATASET="heridal_keras_retinanet_voc"

MODEL="${TRAIN_RUN_NAME}_${BACKBONE}_pascal.h5"
#"celestial-dawn-559_resnet152_pascal.h5"
#"neat-sweep-4_resnet50_pascal_40.h5"

PROJECT_ROOT="/scratch/cs/sar-uav-cv/masters-thesis"
TMP_ROOT="/tmp/${SLURM_JOB_ID}"

mkdir -p $TMP_ROOT/data/datasets/
cp $PROJECT_ROOT/data/datasets/$DATASET.tar $TMP_ROOT/data/datasets/

trap "rm -rf ${TMP_ROOT}; echo 'quit' | nvidia-cuda-mps-control; exit" TERM EXIT

tar -Uxf $TMP_ROOT/data/datasets/$DATASET.tar -C $TMP_ROOT/data/datasets/

module load nvidia-tensorflow/20.02-tf1-py3
# module load anaconda/2020-03-tf1

## Start the MPS server
CUDA_MPS_LOG_DIRECTORY=nvidia-mps srun --gres=gpu:1 nvidia-cuda-mps-control -d&

srun singularity exec --nv -B $PROJECT_ROOT keras-retinanet-gpu.simg /bin/bash $PROJECT_ROOT/computer_vision/keras-retinanet/test-tiling-heridal.sh \
    --gpu 0 \
    --group score-thres-sweep \
    --tags score-thres,sar-eval \
    --backbone infer \
    --score_threshold 0.4 \
    --max_detections 100000 \
    --iou_threshold 0.5 \
    --nms_threshold 0.25 \
    --top_k 10 \
    --seed 26203 \
    --anchor_scale 0.965 \
    --save_path $PROJECT_ROOT/data/images/$MODEL-eval \
    --convert_model true \
    --image_tiling true \
    --image_min_side 1525 \
    --image_max_side 2025 \
    --nms_mode enclose \
    --eval_mode sar-apd \
    --config config.ini \
    --set_name train \
    --model $PROJECT_ROOT/computer_vision/keras-retinanet/snapshots/$MODEL \
    pascal \
    $TMP_ROOT/data/datasets/$DATASET
    
