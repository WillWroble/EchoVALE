#!/bin/bash
#SBATCH --job-name=ltv2_cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=bch-compute
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/cluster_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/cluster_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/cluster.py \
    --npz /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v21/line_embeddings.npz \
    --h5_path /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data/study_findings.h5 \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/val_50_modern_nofetal.txt \
    --method kmeans --k 10000 \
    --output_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v21/clusters
