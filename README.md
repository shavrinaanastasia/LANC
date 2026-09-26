# LANC: Agent Language Geometry

Reproducible MVP for testing how interaction frame and a PCA intervention at the 15th SmolLM2 decoder block affect dialogue-level language geometry. The primary outcome is dialogue PHD, not an estimate from ten utterance embeddings.

## Environment

Requires Python 3.10-3.12 and a CUDA-capable machine is recommended. Install the locked direct dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Workflow

```powershell
python -m agent_language_geometry.cli validate-config --config configs/phase1.yaml
python -m agent_language_geometry.cli smoke --config configs/base.yaml
python -m agent_language_geometry.cli calibrate-pca --config configs/calibration.yaml
python -m agent_language_geometry.cli run-phase1 --config configs/phase1.yaml
python -m agent_language_geometry.cli run-phase2 --config configs/phase2.yaml --pca-artifact artifacts/calibration/<calibration_run_id>/pca_basis.npz
python -m agent_language_geometry.cli analyze --config configs/analysis.yaml --run-dir results/<run_id>
python -m agent_language_geometry.cli report --run-dir results/<run_id>
```

`smoke` runs two evaluation stimuli and one seed across the identity interaction-frame conditions. A Phase 2 smoke run requires a real PCA artifact and uses `configs/phase2.yaml` after calibration. The full MVP creates 216 Phase 1 dialogues, 432 Phase 2 dialogues, and 48 disjoint calibration dialogues. It has not been run unless an ignored `results/<run_id>/manifest.json` exists.

Validate the separate Schweinhart MST-growth implementation on known synthetic shapes:

```bash
python -m agent_language_geometry.cli schweinhart-controls \
  --points 100000 \
  --sample-sizes 10000,20000,40000,60000,80000,100000
```

This command writes an immutable ignored run directory. Its ladder is explicitly supplied because Gromov et al. do not publish the exact synthetic-control ladder.

## Exploratory Long Dialogues

`configs/phase1_long.yaml` is a separate exploratory protocol. It preserves the 216 Phase 1 cells (24 stimuli x 3 interaction frames x 3 generation seeds) and prompt texts, but generates each dialogue until it reaches at least 10,000 generated English-style words. Once the chat context would exceed 6,000 tokens, the oldest complete chat messages are removed. It is not a replacement for preregistered ten-turn Phase 1.

```powershell
python -m agent_language_geometry.cli run-phase1-long --config configs/phase1_long.yaml
```

For a long GPU job, set a stable non-null `run_id` before the first command. A stopped Colab job can then resume only after configuration and completed-record hashes are verified:

```powershell
python -m agent_language_geometry.cli run-phase1-long --config configs/phase1_long_colab.yaml --resume
```

For a throughput benchmark only, never as a substitute for the complete design:

```powershell
python -m agent_language_geometry.cli run-phase1-long --config configs/phase1_long.yaml --limit-cards 1 --limit-seeds 1
```

Run directories are immutable. Do not reuse `run_id`; resume is intentionally not implemented until record-level hash checking is added.

## GitHub Desktop

This folder is already a Git repository with remote `https://github.com/shavrinaanastasia/LANC.git`. Open GitHub Desktop, choose **File > Add local repository**, and select `C:\Users\masog\OneDrive\Desktop\Документы\LANC`. If it is not shown, use **File > Options > Git** to set your identity and restart Desktop. The command-line `git` is not on the system `PATH`, but GitHub Desktop includes its own Git; this does not affect Desktop. OpenCode works with the folder itself, not through a GitHub Desktop attachment.
