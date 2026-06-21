#!/bin/bash
#SBATCH --job-name=merge_shards
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/merge_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/merge_%j.err
#SBATCH --mem=256G
#SBATCH --time=2:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2

python -u merge_shards.py \
    --shard_dir embeddings/fl/mode1/internal/round_240_shards_outside \
    --output embeddings/fl/mode1/internal/round_240_outside.npz
