# Experiment log

One entry per experiment (finished, running, queued). Each entry: goal, where it ran, code, Slurm jobs,
outputs, status, key result. Design decisions and deviations are recorded in `DECISIONS.md`.
Large raw outputs stay on HSE cHARISMa (project proj_1897, user aoshavrina); small tables and reports
are copied into `experiments/`.

Paths on HSE: `~/lanc_run` (dialogue generation), `~/wap_run` (dimension estimation).
Dimension code mirror: `experiments/dimension/code/`.

## Status board (2026-10-03)

| ID | Experiment | Status |
|---|---|---|
| E1 | MVP Phase 1 / Phase 2 (Colab) | done, see `reports/MVP_REPORT.md` |
| E2 | War and Peace intrinsic dimension (SVD, CBOW, ruBERT) | done |
| E3 | Estimator validation on synthetic shapes | done |
| E4 | Long dialogues 10k words (216) | running: 140/216 on 2026-10-03 00:40 MSK |
| E5 | Short 10-turn dialogues for frame CBOW (10 800) | queued after E4 |
| E6 | Calibration on trajectories (Gromov proposal) | held (partly done) |
| E7 | Pseudolanguage: true dimension vs estimate (SVD, CBOW) | done |
| E8 | Pseudolanguage: small BERT from scratch | running |
| E9 | Layer sweep of pca_32 over 30 layers (short + long) | queued last |

---

## E1. MVP (Colab, 2026-09-19)
Two SmolLM2-135M agents, 24 stimuli x 3 frames x 3 seeds, 10 turns; Phase 2 = pca_k after layer 15.
Report: `reports/MVP_REPORT.md`. Phase 2 exploratory (calibration cap violation, see DECISIONS.md).

## E2. War and Peace intrinsic dimension (HSE + laptop, 2026-09-29 .. 10-01)
Goal: reproduce the protocol of "A Language and Its Dimensions: Intrinsic Dimensions of Language Fractal
Structures" (Complexity, 2024) on one novel. 361 chapters, 17 900 lemmas, words / bigrams / trigrams.
Embeddings: SVD (entropy-weighted word x chapter), CBOW (d 5/10/15), ruBERT -> PCA (5/10/15), ruBERT-768.
Estimators: Schweinhart (exact MST, ladders, admissible alpha; range = min-max over all admissible alpha of all
ladders), TwoNN, FisherS, Hidalgo.
Code: `experiments/dimension/code/wap_*.py`, `schweinhart_mst.py`, `id_methods.py`, `id_run.py`, sbatch 2..5.
Jobs: 4363712 (SVD), 4364103 (CBOW), 4364182 (trigrams), 4364206 (BERT->PCA), 4364265 + 4365040 (BERT-768),
4365900 (TwoNN/FisherS/Hidalgo).
Reports: `experiments/dimension/reports/voina_i_mir_polnyi_otchet.pdf`, `REPORT_ru.md`,
`metody_razmernosti_obyasnenie.pdf`.
Key: CBOW bigrams fall inside the article's ranges (d = 5/10/15); BERT-768 bigrams most consistent across methods
(Schweinhart 14.5-18.2, TwoNN 13.9, Hidalgo 14.2/15.0); CBOW words grow with d (no plateau, see E7).

## E3. Estimator validation on synthetic shapes (HSE)
Cubes 5D/8D, sphere S6, Sierpinski carpet, Swiss roll, 2D+6D mixture embedded in 15D (and 30D).
Code: `validate_synthetic.py`, `id_run.py --only-synthetic`. Results: `experiments/dimension/results/synthetic_validation.json`.
Key: Schweinhart error 0.4-12 %; FisherS overestimates curved surfaces; only Hidalgo separates the 2D+6D mixture.

## E4. Long dialogues, 10 000 generated words (HSE, from 2026-10-01)
3 frames x 24 cards x seeds 1103/2207/3301, rolling 6000-token context, SmolLM2-135M, CPU.
Code: `hse_dialogues.py`, `configs/phase1_long_hse.yaml` (run_id phase1-long-10000w-hse-v1), `hse_dialogues_cpu.sbatch`.
Jobs: 4367823 (CPU array, 22 -> 15 workers), 4371611 (7 extra workers), 4371989 (6 workers, after E8),
4369354 (finalize, after all workers). GPU array cancelled (queue blocked by QOS).
Interruptions: 2026-10-02 7 workers cancelled to give cores to E7; 2026-10-03 6 workers cancelled to give
slots to E8 (their in-progress units re-queued via lock removal).
Status 2026-10-03 00:40 MSK: 140/216 (neutral 72, cooperation 68, competition 0). ~4.6 h per dialogue on 2 CPU.

## E5. Short 10-turn dialogues for frame-specific CBOW (HSE, queued)
3 frames x 24 cards x seeds 1..150 = 10 800 dialogues, MVP protocol.
Code: `hse_short_dialogues.py` (run short-10turn-cbow-v1), `hse_short_cpu.sbatch`, then `dialogue_cbow.py`
(lemmatize -> CBOW per frame, min_count 1 and 5) -> `~/wap_run/wap_schweinhart.py --source cbow`.
Job: 4369353 (after E4 array 4367823).

## E6. Calibration on trajectories of known dimension (Gromov proposal, HSE)
Tori T2..T10, spheres S2..S8, Lorenz attractor (2.06); trajectories partitioned into "words" by random / k-means /
ant-colony centres; SVD and CBOW embeddings; four estimators.
Code: `calib_trajectories.py`, `~/wap_run/9_calib_array.sbatch`. Job: 4370967 (30 tasks; tasks 0-17 ran,
the rest held since 2026-10-03 to free slots for E8; restart-safe per cloud).
Pilot: k-means centres form a near-regular lattice -> TwoNN ~ 8 on T2; random partition is the reference.

## E7. Pseudolanguage: true dimension vs estimate (HSE, 2026-10-02)
Notebook "Pseudolanguage.ipynb" generalised: cube [0,1]^m minus 4 balls, m = 2, 3, 4, 6; 5000 points = words;
1000 texts, lengths from `experiments/dimension/data/russian_Prussner_wolframe100.csv`;
walks dist (p ~ 1/d^2, notebook) and eps (uniform within eps = 1.5 V^(-1/m)).
Embeddings: points (truth), SVD, CBOW notebook settings, CBOW War-and-Peace settings; d = 2/5/10/15; words and bigrams.
Code: `pseudolang.py`, `8b_pseudolang_array.sbatch`. Job 4370978 (8 tasks). Table: `experiments/dimension/results/pseudo_table.csv`.
Report: `experiments/dimension/reports/psevdoyazyk_istinnaya_razmernost.pdf`.
Key: CBOW on the local (eps) text recovers m with +0.2..0.9 bias and a plateau d = 10 -> 15
(Schweinhart d = 15: 2.6 / 3.5 / 4.3 / 6.2); SVD overestimates and grows with d; FisherS on CBOW ~ 6-7 for any m;
the dist walk is non-local for m >= 3 (step / nearest-neighbour distance 4-8) and all estimates grow with d.

## E8. Pseudolanguage: small BERT trained from scratch (HSE, 2026-10-03)
Same texts as E7 (regenerated with identical seeds). BERT 4 layers x 256, 4 heads, MLM, 30 epochs (cap 240 min),
contextual vectors averaged per word; PCA to 2/5/10/15 and full 256; words and bigrams; both walks.
Code: `pseudo_bert.py`, `8c_pseudo_bert.sbatch`. Job 4371985 (8 tasks). Smoke: ~2.5 min per epoch on 2 CPU.

## E9. Layer sweep of the nested-PCA intervention (HSE, queued last)
pca_32 after each decoder block L = 1..30 (MVP used only L = 15). Per-layer bases from one identity calibration
(24 calibration cards x 2 seeds, MVP cap 128 per dialogue x agent, same positions for all layers).
Short: neutral, 24 cards x seeds 1..50 per layer (36 000). Long: neutral, 8 cards x seed 1103 per layer (240).
Identity controls reused from E5 / E4.
Code: `hse_layer_sweep.py`, `hse_layer_calib.sbatch`, `hse_layer_short.sbatch`, `hse_layer_long.sbatch`.
Jobs: 4372011 (calibration, after E5), 4372012 (short, after calibration); long to be submitted after E8
(submit limit 100 jobs). All with nice 20000.
