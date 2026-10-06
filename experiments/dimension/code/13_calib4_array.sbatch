#!/bin/bash
#SBATCH --job-name=calib4
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --exclude=cn-040
#SBATCH --output=calib4_logs/calib4_%A_%a.log
# E13: bigram estimate / m as a function of rho = delta / r (calib4.py --list).
cd ~/wap_run
module purge; module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python calib4.py --task $SLURM_ARRAY_TASK_ID
