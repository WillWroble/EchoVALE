#!/bin/bash
#SBATCH --job-name=vp3_train
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:xlarge:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3

export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=4 train.py \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_mimic_bch_cooldown/latest.pt \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --output_dir results/v1 \
    --batch_size 128 \
    --K 2 \
    --N_neg 128 \
    --epochs 300 \
    --ipe 300 \
    --val_ipe 100 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --num_workers 8 \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt
    #--line_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v50/best.pt
    #--resume results/v1/latest.pt
