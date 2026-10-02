#!/bin/bash
#SBATCH --job-name=wap_bert768
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=wap_bert768-%j.log
# Full 768-dim ruBERT vectors (bigrams 1536, trigrams 2304 dims), exact MST on the GPU in float64.
# Words and bigrams: 3 random ladders; trigrams: 1 ladder (each MST at full size ~20 min).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/PyTorch_GPU_v2.4
echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u schweinhart_mst.py
python -u wap_add_trigrams.py wap_embeddings_alt.npz wap_sentences.json.gz wap_embeddings_alt3_gpu.npz
python -u wap_schweinhart.py --emb wap_embeddings_alt3_gpu.npz --source bert_full --ngrams 1 2 --dims 768 \
       --variants all --backend torch --out wap_results_alt --seeds 1 2 3
python -u wap_schweinhart.py --emb wap_embeddings_alt3_gpu.npz --source bert_full --ngrams 3 --dims 768 \
       --variants all --backend torch --out wap_results_alt --seeds 1
echo "=== finished $(date) ==="
