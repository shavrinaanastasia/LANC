#!/bin/bash
#SBATCH --job-name=wap_cbow3
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=wap_cbow3-%j.log
# CBOW trigrams (concatenated word vectors, 15/30/45 dims), 3 random ladders.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host $(hostname) $(date)"
python -u wap_add_trigrams.py wap_embeddings_cbow.npz wap_sentences.json.gz wap_embeddings_cbow3.npz
python -u wap_schweinhart.py --emb wap_embeddings_cbow3.npz --source cbow --ngrams 3 --variants all --out wap_results_alt --seeds 1 2 3
echo "=== finished $(date) ==="
