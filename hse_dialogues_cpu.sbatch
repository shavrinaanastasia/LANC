#!/bin/bash
#SBATCH --job-name=lanc_dlg_cpu
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --array=0-215%MAXPAR
#SBATCH --output=logs/cpu_%A_%a.log
# One array task = one dialogue unit (worker k of 216). Already finished units are skipped.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY hse_dialogues.py work --worker $SLURM_ARRAY_TASK_ID --workers 216 --threads $SLURM_CPUS_PER_TASK --device cpu
