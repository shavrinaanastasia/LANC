#!/bin/bash
#SBATCH --job-name=calib2
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --array=0-28
#SBATCH --exclude=cn-040
#SBATCH --output=calib2_logs/calib2_%A_%a.log
# E10 (Gromov 2026-10-05): tasks 0-20 = part A (higher d up to 100, T12/S12, fractals, multifractals, FNN/Cao);
# tasks 21-28 = part B (vocabulary 1000..20000 and 100..2000 trajectories). See calib2.py --list.
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python calib2.py --task $SLURM_ARRAY_TASK_ID
