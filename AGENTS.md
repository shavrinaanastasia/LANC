# Repository Instructions

- Preserve the preregistered MVP design in `RESEARCH.md` and record forced deviations in `DECISIONS.md`.
- Generated outputs belong in ignored run directories. Never overwrite an existing run directory.
- Do not use dialogue-level generation seeds as independent statistical replicates.
- The primary intrinsic-dimension estimator is `phd.py`; MLE and TwoNN are only cross-checks.
- `gromov_pooled.py` must hard-fail with `INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL` when its ladder is not supported.
