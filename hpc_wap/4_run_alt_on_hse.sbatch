#!/bin/bash
#SBATCH --job-name=wap_alt_cpu
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=wap_alt_cpu-%j.log
# CBOW (paper Table 5 method) and ruBERT->PCA 5/10/15, words and bigrams, 3 random ladders each.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/Anaconda
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host $(hostname) $(date)"
python -u wap_cbow_hse.py wap_embeddings_alt.npz wap_sentences.json.gz wap_embeddings_alt_cbow.npz
for SRC in cbow bert_pca; do
  python -u wap_schweinhart.py --emb wap_embeddings_alt_cbow.npz --source $SRC --variants all --out wap_results_alt --seeds 1 2 3
done
echo "=== finished $(date) ==="
