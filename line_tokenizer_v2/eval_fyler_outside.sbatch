#!/bin/bash
#SBATCH --job-name=eval_fyler
#SBATCH --gres=gpu:1
#SBATCH --mem=350G
#SBATCH --time=7-00:00:00
#SBATCH --partition=bch-compute
##SBATCH --partition=bch-compute-pe
##SBATCH --partition=bch-gpu
#SBATCH --exclude=gpu-b07-0,gpu-b07-1,gpu-b07-2,gpu-b07-3,gpu-b08-0,gpu-b09-0,gpu-b11-0,gpu-b11-1,gpu-b11-2,gpu-b11-3
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/eval_fyler_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/eval_fyler_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_outside_fixed.npz \
python -u eval_fyler.py \
    --checkpoint results/v60/best.pt \
    --fyler_lines /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_lines.csv \
    --fyler_labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/fyler_labels_v2.csv \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/embeddings/clip_encoder_4x1024_outside.npz \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/study_outside.txt \
    --output_dir results/v60/eval_fyler_lines_v2_outside
