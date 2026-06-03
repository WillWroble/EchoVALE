#!/bin/bash
#SBATCH --job-name=concat_npz
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --partition=bch-compute
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/concat_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/concat_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python -u -c "
import numpy as np
a = np.load('/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed.npz')
b = np.load('/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_outside_fixed.npz')
np.savez(
    '/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/embeddings/jepa_clips_4x768_fixed_plus_outside.npz',
    embeddings=np.concatenate([a['embeddings'], b['embeddings']]),
    study_ids=np.concatenate([a['study_ids'], b['study_ids']]),
)
print(f'train: {len(a[\"study_ids\"])}, outside: {len(b[\"study_ids\"])}, combined: {len(a[\"study_ids\"])+len(b[\"study_ids\"])}')
"
