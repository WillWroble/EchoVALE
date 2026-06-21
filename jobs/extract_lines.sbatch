#!/bin/bash
#SBATCH --job-name=extract_lines
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=bch-compute-pe
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_lines_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_lines_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip

python -u code/extract_lines.py \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/study_outside.txt \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --output_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_labels_outside
