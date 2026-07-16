#!/bin/bash
#SBATCH --job-name=ltv2_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:large:1
##SBATCH --gres=gpu:xlarge:1
#SBATCH --time=7-00:00:00
##SBATCH --partition=bch-gpu-pe
##SBATCH --partition=bch-compute
#SBATCH --exclude=gpu-b07-0,gpu-b07-1,gpu-b07-2,gpu-b07-3,gpu-b08-0,gpu-b09-0,gpu-b11-0,gpu-b11-1,gpu-b11-2,gpu-b11-3
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/train_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/random_clips_4x768.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
python -u train.py \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x1024_MIMIC.npz \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --fields study_findings summary history \
    --epochs 50 \
    --output_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v57
    #--checkpoint results/v53/best.pt \
    #--lr 1e-4 \
    #--warmup_frac 0
