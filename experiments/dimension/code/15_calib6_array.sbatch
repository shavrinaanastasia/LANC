#!/bin/bash
#SBATCH --job-name=calib6
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --array=0-23
#SBATCH --exclude=cn-040
#SBATCH --output=calib6_logs/calib6_%A_%a.log
# E15: cross-swaps torus vs pseudolanguage generator -- calib6.py --list.
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python calib6.py --task $SLURM_ARRAY_TASK_ID
