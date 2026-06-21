#!/bin/bash
#SBATCH --job-name=jepa_extract
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/extract_%A_%a.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/extract_%A_%a.err
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-127
#SBATCH --exclude=gpu-b11-1,gpu-b11-2,gpu-b11-3

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2
mkdir -p logs embeddings/shards

export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=4

python -u extract_jepa_clips.py \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --study_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_all.txt \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/fl/mode1/internal/round_240.pt \
    --output_dir embeddings/fl/mode1/internal/round_240_shards \
    --num_clips 2 \
    --batch_size 16 \
    --n_tasks 128
