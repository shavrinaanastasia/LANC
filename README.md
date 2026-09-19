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

Run directories are immutable. Do not reuse `run_id`; resume is intentionally not implemented until record-level hash checking is added.

## GitHub Desktop

This folder is already a Git repository with remote `https://github.com/shavrinaanastasia/LANC.git`. Open GitHub Desktop, choose **File > Add local repository**, and select `C:\Users\masog\OneDrive\Desktop\Документы\LANC`. If it is not shown, use **File > Options > Git** to set your identity and restart Desktop. The command-line `git` is not on the system `PATH`, but GitHub Desktop includes its own Git; this does not affect Desktop. OpenCode works with the folder itself, not through a GitHub Desktop attachment.
