#!/bin/bash
#SBATCH --job-name=umap_views
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/umap_views_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/umap_views_%j.err
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3
export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
mkdir -p logs
python -u umap_views.py \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_mimic_bch_cooldown/latest.pt \
    --v3_checkpoint results/v3/best.pt \
    --output_dir results/v3/umaps
