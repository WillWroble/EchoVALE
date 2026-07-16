#!/bin/bash
#SBATCH --job-name=ltv2_gen_lines
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/generate_lines_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/generate_lines_%j.err
##SBATCH --partition=bch-compute-pe
##SBATCH --partition=bch-compute
#SBATCH --exclude=gpu-b07-0,gpu-b07-1,gpu-b07-2,gpu-b07-3,gpu-b08-0,gpu-b09-0,gpu-b11-0,gpu-b11-1,gpu-b11-2,gpu-b11-3,gpu-21-0,gpu-21-1,gpu-21-2
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:large:1 --cpus-per-task=8 --mem=512G --time=7-00:00:00
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2
mkdir -p logs

#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
python -u generate_lines.py \
    --checkpoint results/v54/best.pt \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x1024_MIMIC.npz \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --pool_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/train_50_modern_nofetal.txt \
    --eval_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val_modern_1500.txt \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --output results/v54/line_heatmaps_v3.json \
    --threshold 0.95 "$@"
