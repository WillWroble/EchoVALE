#!/bin/bash
#SBATCH --job-name=v3_extract
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/extract_%A_%a.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/extract_%A_%a.err
#SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:large:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=37-40,57-127%32
#SBATCH --exclude=gpu-b11-3,gpu-10-3,gpu-20-1,gpu-b07-0,gpu-26-0,gpu-26-1,gpu-26-5,gpu-21-1,gpu-21-2
##SBATCH --exclude=gpu-b07-0,gpu-b07-1,gpu-b07-2,gpu-b07-3,gpu-b08-0,gpu-b09-0,gpu-b11-0,gpu-b11-1,gpu-b11-2,gpu-b11-3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3
mkdir -p logs embeddings
export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=4
python -u extract_clip_encoder.py \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --study_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_all.txt \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_mimic_bch_cooldown/latest.pt \
    --v3_checkpoint results/v4/best.pt \
    --output_dir embeddings/clip_encoder_v4_best_shards \
    --num_clips 4 \
    --batch_size 8 \
    --n_tasks 128
