#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#asd SBATCH --array=0-3
#SBATCH --ntasks=1
#asd SBATCH --ntasks-per-node=1
#SBATCH --tmp=30G
#asd SBATCH --exclusive=user
#asd SBATCH --test-only
#asd SBATCH --mem-per-gpu=64G
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#asd SBATCH --cpus-per-gpu=4
#SBATCH --job-name=train-profiling
#SBATCH --output=/scratch/cs/sar-uav-cv/masters-thesis/computer_vision/keras-retinanet/slurm_out/%A-%a-%j-%x.out
#SBATCH --partition=gpu

export SINGULARITY_CACHEDIR=/scratch/cs/sar-uav-cv/.singularity

PROJECT_ROOT="/scratch/cs/sar-uav-cv/masters-thesis"
DATASET="heridal_keras_retinanet_voc_tiled"

# if [[ -z "${SLURM_ARRAY_JOB_ID}" ]]; then
#     TMP_ROOT="/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# else 
TMP_ROOT="/tmp/${SLURM_JOB_ID}"
# fi

mkdir -p $TMP_ROOT/data/datasets/
cp $PROJECT_ROOT/data/datasets/$DATASET.tar $TMP_ROOT/data/datasets/

trap "rm -rf ${TMP_ROOT}; echo 'quit' | nvidia-cuda-mps-control; exit" TERM EXIT
# trap "rm -rf ${TMP_ROOT}; exit" TERM EXIT

tar -Uxf $TMP_ROOT/data/datasets/$DATASET.tar -C $TMP_ROOT/data/datasets/

module load nvidia-tensorflow/20.02-tf1-py3
# module load anaconda/2020-03-tf1

## Start the MPS server
CUDA_MPS_LOG_DIRECTORY=nvidia-mps srun --gres=gpu:1 nvidia-cuda-mps-control -d&

# --snapshot=$PROJECT_ROOT/computer_vision/keras-retinanet/snapshots/stanford-drones_resnet152_csv.h5 \

srun singularity exec --nv -B $PROJECT_ROOT keras-retinanet-gpu.simg \
    /bin/bash $PROJECT_ROOT/computer_vision/keras-retinanet/train.sh \
    --random_transform=true \
    --gpu=array \
    --compute_val_loss \
    --steps=5564 \
    --epochs=50 \
    --backbone=resnet152 \
    --group=retinanet-train-profiling \
    --tags=profiling \
    --snapshot_interval=1 \
    --config=config_optimized.ini \
    --no_resize=true \
    --image_min_side=1525 \
    --image_max_side=2025 \
    --early_stop_patience=15 \
    --reduce_lr_patience=4 \
    --anchor_scale=0.965 \
    --reduce_lr_factor=0.33 \
    --seed=222 \
    --lr=0.00002196 \
    --initial_epoch=0 \
    --loss_alpha_factor=0.25 \
    --foreground_overlap_threshold=0.5 \
    pascal \
    $TMP_ROOT/data/datasets/$DATASET
