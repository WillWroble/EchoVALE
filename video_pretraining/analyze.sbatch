#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/analyze_%j.out
#SBATCH --mem=256G
#SBATCH --time=06:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining

python -u analyze.py \
    --input ../video_pretraining_v2/embeddings/jepa_clips_4x1024_MIMIC.npz \
    --output_fig results/mean_pool/analysis.png \
    --n_studies 50
