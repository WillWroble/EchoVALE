#!/bin/bash
#SBATCH --job-name=v3_merge
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/merge_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/merge_%j.err
#SBATCH --mem=256G
#SBATCH --time=2:00:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3
python -u merge_shards.py \
    --shard_dir embeddings/clip_encoder_outside_shards \
    --output embeddings/clip_encoder_4x1024_outside.npz
