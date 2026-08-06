#!/bin/bash
#SBATCH --job-name=efm_chd_probe
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/efm_chd_probe_%j.out
#SBATCH --error=logs/efm_chd_probe_%j.err
#SBATCH --mem=256G
#SBATCH --gres=gpu:large:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --partition=bch-compute
#SBATCH --exclude=gpu-b07-0,gpu-b07-1,gpu-b07-2,gpu-b07-3,gpu-b08-0,gpu-b09-0,gpu-b11-0,gpu-b11-1,gpu-b11-2,gpu-b11-3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini

mkdir -p logs

python -u train_echofocusmini_chd.py \
    --embeddings ../video_pretraining_v3/embeddings/clip_encoder_v3_best_4x1024.npz \
    --train_manifest ../manifests/platon_train.txt \
    --val_manifest ../manifests/platon_val.txt \
    --output_dir results/ce/v3_best \
    --min_pos 5 \
    --n_videos 48
