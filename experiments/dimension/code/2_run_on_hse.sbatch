#!/bin/bash
#SBATCH --job-name=wap_schweinhart
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=wap_schweinhart-%j.log

# War and Peace, Schweinhart intrinsic dimension (Gromov et al. 2024 protocol).
# Inputs in the same folder: schweinhart_mst.py validate_synthetic.py wap_schweinhart.py wap_embeddings.npz
# Memory: exact Prim MST keeps no distance matrix, the largest cloud (~250k x 30 floats) is < 100 MB.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/Anaconda   # anaconda 2024_02: numpy 1.26, scipy 1.11, numba 0.59
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$SLURM_SUBMIT_DIR/.numba_cache"

echo "host $(hostname)  cpus $SLURM_CPUS_PER_TASK  $(date)"
python -c "import sys, numpy, scipy, numba; print(sys.version); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'numba', numba.__version__)"

echo "=== 1. exact-MST self-check against scipy dense MST ==="
python -u schweinhart_mst.py

echo "=== 2. synthetic sets of known dimension, ambient 15 and 30 ==="
python -u validate_synthetic.py --n 50000 --ambient 15 30 --out synthetic_validation_hse.json

echo "=== 3. War and Peace: words and bigrams, d = 5, 10, 15 ==="
python -u wap_schweinhart.py --emb wap_embeddings.npz --out wap_results --seeds 1 2 3

echo "=== finished $(date) ==="
