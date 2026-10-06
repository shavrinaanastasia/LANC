#!/bin/bash
#SBATCH --job-name=calib5
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --array=0-20
#SBATCH --exclude=cn-040
#SBATCH --output=calib5_logs/calib5_%A_%a.log
# E14: CBOW noise removal (bootstrap runs, averaging, frequency filter, scale above noise) -- calib5.py --list.
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python calib5.py --task $SLURM_ARRAY_TASK_ID
