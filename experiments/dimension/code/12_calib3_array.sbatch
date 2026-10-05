#!/bin/bash
#SBATCH --job-name=calib3
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --array=0-29
#SBATCH --exclude=cn-040
#SBATCH --output=calib3_logs/calib3_%A_%a.log
# E12: robustness of the CBOW over-estimation factor inside the Brownian group (calib3.py --list).
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python calib3.py --task $SLURM_ARRAY_TASK_ID
