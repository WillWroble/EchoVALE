#!/bin/bash
#SBATCH --job-name=ltv2_encode
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=bch-compute
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/encode_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/encode_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2
#--checkpoint results/v19/best.pt \
python -u encode.py \
    --h5_path /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data/study_findings.h5 \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --output results/v0/line_embeddings.npz
