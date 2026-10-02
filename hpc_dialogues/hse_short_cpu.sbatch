#!/bin/bash
#SBATCH --job-name=lanc_short
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/short_%A_%a.log
# Short 10-turn dialogues for frame-specific CBOW: worker k of 22 takes every 22nd unit (seed-major order).
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY hse_short_dialogues.py work --worker $SLURM_ARRAY_TASK_ID --workers 22 --threads $SLURM_CPUS_PER_TASK --device cpu
