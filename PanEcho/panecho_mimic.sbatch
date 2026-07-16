#!/bin/bash
#SBATCH --array=1-198%16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=panecho_mimic
#SBATCH --output=logs/panecho_%A_%a.out
#SBATCH --error=logs/panecho_%A_%a.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/PanEcho
python -u process_batch.py --batch_num $SLURM_ARRAY_TASK_ID --out_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Embeddings/mimic_iv
