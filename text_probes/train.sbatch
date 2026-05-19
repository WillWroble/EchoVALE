#!/bin/bash
#SBATCH --job-name=text_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=bch-compute
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_probes/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_probes/logs/train_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_probes
mkdir -p logs

python -u train.py \
    --field study_findings \
    --line_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v0/line_embeddings.npz \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --targets_csv /lab-share/Cardio-Mayourian-e2/Public/Echo_Labels/echo_measurements_090425.csv \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/train_50_modern_nofetal.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/val_50_modern_nofetal.txt \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --output_dir results/v0/study_findings/v2/

