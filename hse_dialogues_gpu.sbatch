#!/bin/bash
#SBATCH --job-name=lanc_dlg_gpu
#SBATCH --account=proj_1897
#SBATCH --partition=gpu-ef-quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --array=0-ARRAY_LAST
#SBATCH --requeue
#SBATCH --output=logs/gpu_%A_%a.log
# Each array task = 1 GPU running PROCS independent worker processes (SmolLM2-135M is tiny,
# one process cannot saturate a GPU). Global worker id = task*PROCS + p, total = tasks*PROCS.
# Finished dialogues are skipped, so a preempted/requeued task just continues.
module purge; module load Python/Google_Colab_GPU_2025
PY=/opt/software/python/envs/google_colab_gpu_2025/bin/python
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PROCS=${PROCS:-6}
TASKS=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 ))
TOTAL=$(( TASKS * PROCS ))
nvidia-smi --query-gpu=name,memory.total --format=csv
for p in $(seq 0 $((PROCS-1))); do
  W=$(( SLURM_ARRAY_TASK_ID * PROCS + p ))
  $PY hse_dialogues.py work --worker $W --workers $TOTAL --threads 1 --device cuda > logs/gpu_${SLURM_ARRAY_JOB_ID}_w${W}.log 2>&1 &
done
wait
$PY hse_dialogues.py status
