#!/bin/bash
#SBATCH --job-name=lanc_smoke
#SBATCH --account=proj_1897
#SBATCH --partition=PART
#SBATCH --cpus-per-task=4
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --output=logs/smoke_%j.log
# Generates ONE real unit (worker W of 216) to measure speed; the record is valid and is kept.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY hse_dialogues.py work --worker ${W:-0} --workers 216 --threads $SLURM_CPUS_PER_TASK ${DEV:+--device $DEV}
