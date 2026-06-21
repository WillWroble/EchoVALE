#!/bin/bash
#SBATCH --job-name=eval_ext_chd
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/eval_ext_chd_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/eval_ext_chd_%j.err
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini

python -u eval_external_chd.py \
    --checkpoint results/768_jepa_chd_platon/best.pt \
    --embeddings ../video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz \
    --train_manifest ../manifests/platon_train.txt \
    --output_dir results/768_jepa_chd_platon/external
