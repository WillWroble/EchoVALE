#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=bch-compute
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/train_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

python -u train.py \
    --contrast \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings_v2 \
    --train_manifest manifests/train_50_nofetal.txt \
    --val_manifest manifests/val_50_nofetal.txt \
    --output_dir results/full_contrast_jepa_v1
