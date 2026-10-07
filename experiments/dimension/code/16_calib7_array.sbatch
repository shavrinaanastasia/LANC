#!/bin/bash
#SBATCH --job-name=calib7
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=0-12:00:00
#SBATCH --array=0-12
#SBATCH --exclude=cn-040
#SBATCH --output=calib7_logs/calib7_%A_%a.log
# E16: hidden state beyond the current word (n-gram gain, Markov surrogate) -- calib7.py --list.
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python calib7.py --task $SLURM_ARRAY_TASK_ID
