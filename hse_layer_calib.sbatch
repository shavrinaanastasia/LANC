#!/bin/bash
#SBATCH --job-name=lanc_layer_calib
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/layer_calib_%j.log
# Per-layer PCA bases (30 layers) from one identity calibration run, then the immutable run directory.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY hse_layer_sweep.py calibrate --threads $SLURM_CPUS_PER_TASK && $PY hse_layer_sweep.py prepare
