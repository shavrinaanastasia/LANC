#!/bin/bash
#SBATCH --job-name=lanc_dlg_gpu
#SBATCH --account=proj_1897
#SBATCH --partition=gpu-ef-quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:50:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=logs/gpu_%A_%a.log
# Each array task = 1 GPU with PROCS worker processes; every process claims the next free unit
# (lock files shared with the CPU workers), from the beginning of the list. Finished units are skipped,
# so a preempted/requeued or resubmitted task simply continues.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PROCS=${PROCS:-6}
nvidia-smi --query-gpu=name,memory.total --format=csv
for p in $(seq 1 $PROCS); do
  $PY hse_dialogues.py work --workers 1 --claims --order asc --device cuda --threads 1 \
     > logs/gpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_p${p}.log 2>&1 &
  sleep 20   # stagger start-up so processes do not race for the same first units
done
wait
$PY hse_dialogues.py status
