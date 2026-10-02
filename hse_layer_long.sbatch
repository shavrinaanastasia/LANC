#!/bin/bash
#SBATCH --job-name=lanc_layer_long
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/layer_long_%A_%a.log
# 10k-word neutral dialogues with pca_32 after layer L = 1..30 (8 cards x seed 1103); lock-file claims.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY hse_layer_sweep.py long --threads $SLURM_CPUS_PER_TASK
