#!/bin/bash
#SBATCH --job-name=lanc_cbow_dims
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/cbow_dims_%j.log
# After the short-dialogue corpus: export -> lemmatize (spaCy) -> frame-specific CBOW (min_count 1 and 5,
# d = 5/10/15/20/30) -> intrinsic dimension (Schweinhart, TwoNN, FisherS, Hidalgo) per frame.
set -e
cd ~/lanc_run
export HF_HOME=~/lanc_run/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
COLAB=/opt/software/python/envs/google_colab_gpu_2025/bin/python
$COLAB hse_short_dialogues.py status
$COLAB hse_short_dialogues.py export
$COLAB dialogue_cbow.py lemmatize
module purge; module load Python/Anaconda
python dialogue_cbow.py train --min-count 1
python dialogue_cbow.py train --min-count 5
cd ~/wap_run
python dialogue_dims.py --src ~/lanc_run/results/short-10turn-cbow-v1/cbow --out dialogue_dims_results
echo "=== finished $(date) ==="
