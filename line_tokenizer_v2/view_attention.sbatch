#!/bin/bash
#SBATCH --job-name=view_attn
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --partition=bch-compute-pe
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/view_attn_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/view_attn_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

python -u view_attention_heads.py \
    --checkpoint results/v35b/best.pt \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x1024_platon.npz \
    --view_labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/view_labels.csv \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt
