#!/bin/bash
#SBATCH --job-name=merge_hdf5
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/merge_hdf5_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/merge_hdf5_%j.err
#SBATCH --mem=256G
#SBATCH --time=4:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public

python -u Echo_Clip/video_pretraining_v2/merge_shards_hdf5.py \
    --shard_dir Echo_Clip/video_pretraining_v2/embeddings/shards_platon_1024d \
    --hdf5_output_dir Echo_JEPA_Embeddings/1024d
