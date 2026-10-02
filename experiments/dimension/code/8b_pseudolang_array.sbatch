#!/bin/bash
#SBATCH --job-name=pseudo
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=pseudo_logs/pseudo_%A_%a.log
# One task = one (object dimension m, walk): pseudolanguage texts -> SVD/CBOW -> 4 estimators.
M=(2 4 3 6)
W=(dist eps)
m=${M[$((SLURM_ARRAY_TASK_ID / 2))]}
w=${W[$((SLURM_ARRAY_TASK_ID % 2))]}
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python pseudolang.py --dims-obj $m --walks $w --emb-dims 2 5 10 15 --max-points 10000 --ladders 2 --out pseudo_results
