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
    --embeddings ../video_pretraining_v2/embeddings/jepa_clips_4x768_fixed_plus_outside.npz \
    --train_manifest ../manifests/study_train_50.txt \
    --val_manifest ../manifests/study_outside.txt \
    --output_dir results/chd_jepa_v4_external \
    --min_pos 1 \
    --n_videos 48
