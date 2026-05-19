#!/bin/bash
#SBATCH --job-name=gen_random_embs
#SBATCH --partition=bch-compute
#SBATCH --mem=350G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/gen_random_embs_%j.out

python -u code/generate_random_clip_embeddings.py
