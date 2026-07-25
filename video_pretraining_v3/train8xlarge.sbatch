#!/bin/bash
#SBATCH --job-name=vp3_train8
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/train8_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/train8_%j.err
#SBATCH --partition=bch-compute-pe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:large:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=750G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=gpu-b07-0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3

export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=8 train.py \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_mimic_bch_cooldown/latest.pt \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --output_dir results/v3 \
    --batch_size 64 \
    --K 4 \
    --N_neg 128 \
    --epochs 300 \
    --ipe 300 \
    --val_ipe 100 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --num_workers 8 \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt
    #--resume results/v3/latest.pt
