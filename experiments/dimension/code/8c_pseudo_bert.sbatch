#!/bin/bash
#SBATCH --job-name=pbert
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=0-08:00:00
#SBATCH --array=0-7
#SBATCH --output=pseudo_logs/pbert_%A_%a.log
# One task = one (m, walk): small BERT from scratch on the pseudo-texts -> mean contextual vectors -> PCA / full -> 4 estimators.
M=(2 4 3 6)
W=(dist eps)
m=${M[$((SLURM_ARRAY_TASK_ID / 2))]}
w=${W[$((SLURM_ARRAY_TASK_ID % 2))]}
cd ~/wap_run
module purge; module load Python/Google_Colab_GPU_2025
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
/opt/software/python/envs/google_colab_gpu_2025/bin/python pseudo_bert.py --m $m --walk $w --threads $SLURM_CPUS_PER_TASK --epochs 30 --max-train-min 240
