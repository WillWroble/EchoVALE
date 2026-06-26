#!/bin/bash
#SBATCH --job-name=efm_chd_probe
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/efm_chd_probe_%j.out
#SBATCH --error=logs/efm_chd_probe_%j.err
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --partition=bch-gpu-pe


source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini

mkdir -p logs

python -u train_echofocusmini_chd.py \
    --embeddings ../video_pretraining_v2/embeddings/fl/mode3/internal/round_240.npz \
    --train_manifest ../manifests/platon_train.txt \
    --val_manifest ../manifests/platon_val.txt \
    --output_dir results/fl_jepa_mode1_v3 \
    --min_pos 5 \
    --n_videos 48
