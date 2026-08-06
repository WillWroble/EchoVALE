#!/bin/bash
#SBATCH --job-name=ltv3_train
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v3/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v3/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:large:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
##SBATCH --exclude=gpu-10-3,gpu-20-1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v3
export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/JEPA_probes:$PYTHONPATH
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
torchrun --standalone --nproc_per_node=4 train.py \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitl_mimic_bch_cooldown/latest.pt \
    --data_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled/Echo_Internal_30k \
    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_train.txt \
    --val_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/platon_val.txt \
    --fields study_findings summary \
    --K 2 \
    --M 16 \
    --clip_dropout 0.8 \
    --max_clips 32 \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --output_dir results/v2
    #--encoder_warmstart /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v3/results/v3/day2.pt
    #--resume results/v1/latest.pt
