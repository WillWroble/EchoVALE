#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=bch-compute
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/embed_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/embed_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

python -u embed.py \
    --checkpoint results/full_contrast_jepa_v1/latest.pt \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings_v2 \
    --train_manifest ../manifests/study_train_50.txt \
    --val_manifest ../manifests/study_outside.txt \
    --output results/full_contrast_jepa_v1/eval_external/embeddings.npz \
    --echofocus_checkpoint results/full_contrast_jepa_v1/echofocus_latest.pt \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed_plus_outside.npz
