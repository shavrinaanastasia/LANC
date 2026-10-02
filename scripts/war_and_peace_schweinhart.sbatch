#!/bin/bash
#SBATCH --job-name=war_peace_schweinhart
#SBATCH --account=proj_1897
#SBATCH --partition=rocky
#SBATCH --constraint=type_b
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=war_peace_schweinhart-%j.log
#SBATCH --error=war_peace_schweinhart-%j.err

set -euo pipefail
module purge
module load python
source activate war-and-peace-schweinhart
ws_allocate war-peace-schweinhart 30
WORKSPACE=$(ws_find war-peace-schweinhart)
mkdir -p "$WORKSPACE/input" "$WORKSPACE/output"
cp "$HOME/war_and_peace_ru.txt" "$WORKSPACE/input/"
cp "$HOME/war_and_peace_metadata.json" "$WORKSPACE/input/"
python -u scripts/war_and_peace_svd.py \
  --text "$WORKSPACE/input/war_and_peace_ru.txt" \
  --metadata "$WORKSPACE/input/war_and_peace_metadata.json" \
  --output "$WORKSPACE/output/svd"

# Deliberate stop: exact high-dimensional MST job is not submitted until
# the post-SVD unigram/bigram counts and dense-memory preflight are reviewed.
echo "SVD complete; no MST calculation was run." 
