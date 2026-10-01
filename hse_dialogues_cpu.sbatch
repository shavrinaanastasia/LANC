#!/bin/bash
#SBATCH --job-name=lanc_dlg_cpu
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=4-00:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/cpu_%A_%a.log
# CPU workers share the run with GPU workers: each one claims the next free unit (lock file),
# starting from the END of the unit list (GPU workers start from the beginning), until none is left.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY hse_dialogues.py work --workers 1 --claims --order desc --device cpu --threads $SLURM_CPUS_PER_TASK
