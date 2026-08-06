#!/bin/bash
#SBATCH --job-name=chd_fl
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/chd_fl_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/chd_fl_%j.err
#SBATCH --mem=256G
#SBATCH --gres=gpu:large:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu-26-0

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini
mkdir -p logs

python -u train_chd_fl.py \
    --embeddings ../video_pretraining_v3/embeddings/clip_encoder_v3_best_4x1024.npz \
                 ../video_pretraining_v3/embeddings/clip_encoder_v3_best_4x1024_outside.npz \
    --labels ../Echo_Labels_SG_Fyler_112025.csv \
    --val_manifests ../manifests/platon_val.txt ../manifests/study_outside_us_val.txt \
    --train_manifests ../manifests/platon_train.txt \
    --local_steps 100 \
    --batch_sizes 64 \
    --rounds 100 \
    --output_dir results/chd_internal_v3_best
