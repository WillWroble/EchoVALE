#!/bin/bash
#SBATCH --job-name=eval_line_aurocs
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --time=24:00:00
##SBATCH --partition=bch-compute
#SBATCH --partition=bch-compute-pe
##SBATCH --partition=bch-gpu
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/eval_fyler_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/eval_fyler_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
#--video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_outside_fixed.npz \
python -u eval_line_aurocs.py \
    --checkpoint results/v35b/best.pt \
    --fyler_lines /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_labels/line_vocab.csv \
    --fyler_labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_labels/line_labels.csv \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x1024_platon.npz \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt \
    --output_dir results/v35b/eval_lines
