# Experiment log

One entry per experiment (finished, running, queued). Each entry: goal, where it ran, code, Slurm jobs,
outputs, status, key result. Design decisions and deviations are recorded in `DECISIONS.md`.
Large raw outputs stay on HSE cHARISMa (project proj_1897, user aoshavrina); small tables and reports
are copied into `experiments/`.

Paths on HSE: `~/lanc_run` (dialogue generation), `~/wap_run` (dimension estimation).
Dimension code mirror: `experiments/dimension/code/`.

## Status board (2026-10-04 07:30 MSK)

| ID | Experiment | Status |
|---|---|---|
| E1 | MVP Phase 1 / Phase 2 (Colab) | done, see `reports/MVP_REPORT.md` |
| E2 | War and Peace intrinsic dimension (SVD, CBOW, ruBERT) | done |
| E3 | Estimator validation on synthetic shapes | done |
| E4 | Long dialogues 10k words (216) | done: 216/216, finalized (4369354) |
| E5 | Short 10-turn dialogues for frame CBOW (10 800) | done: 10800/10800, CBOW dimensions done (4373243); report v1 |
| E6 | Calibration on trajectories (Gromov proposal) | done 30/30; report v2 |
| E7 | Pseudolanguage: true dimension vs estimate (SVD, CBOW) | done |
| E8 | Pseudolanguage: small BERT from scratch | done |
| E10 | Calibration round 2 (Gromov 2026-10-05): d up to 100, fractals, V / trajectories, FNN-Cao blind protocol | 50/53 configs done (mftetra, two large part-B grids running); report v1 |
| E9 | Layer sweep of pca_32 over 30 layers (short + long) | short 36000/36000 done 2026-10-05; long running (17 workers) |

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
2026-10-03 evening: 216/216, finalize job 4369354 COMPLETED.

## E5. Short 10-turn dialogues for frame-specific CBOW (HSE, queued)
3 frames x 24 cards x seeds 1..150 = 10 800 dialogues, MVP protocol.
Code: `hse_short_dialogues.py` (run short-10turn-cbow-v1), `hse_short_cpu.sbatch`, then `dialogue_cbow.py`
(lemmatize -> CBOW per frame, min_count 1 and 5) -> `~/wap_run/wap_schweinhart.py --source cbow`.
Job: 4369353 (after E4 array 4367823). Analysis job 4373243 (after 4369353, `hse_cbow_pipeline.sbatch`):
export -> lemmatize -> CBOW min_count 1 and 5, d = 5/10/15/20/30 -> `experiments/dimension/code/dialogue_dims.py`
(same four estimators as the calibration; d = 20/30 added to check the plateau criterion).
Done 2026-10-04 01:56 MSK: 3600 dialogues per frame, ~640-700k lemma tokens and ~6000 types per frame.
Table: `experiments/dimension/results/dialogue_dims_table.csv` (60 rows); report
`experiments/dimension/reports/razmernost_dialogov_po_freimam.pdf` (`code/dialogue_report.py`).
Key: frames are indistinguishable (differences <= 1-1.5, same as min_count 1 vs 5). Words: no plateau (TwoNN 4.6 -> 8.7 ->
11 -> 13 -> 15 for d = 5..30; Schweinhart has no admissible alpha for d >= 15) -- like the dist control or m >= 8 objects.
Bigrams: plateau (Schweinhart 7.8-10.4, Hidalgo 6-9 for d = 10..30; TwoNN 7.0 -> 9.5 slowly). FisherS uninformative.

## E6. Calibration on trajectories of known dimension (Gromov proposal, HSE)
Tori T2..T10, spheres S2..S8, Lorenz attractor (2.06); trajectories partitioned into "words" by random / k-means /
ant-colony centres; SVD and CBOW embeddings; four estimators.
Code: `calib_trajectories.py`, `~/wap_run/9_calib_array.sbatch`. Job: 4370967 (30 tasks; tasks 0-17 ran,
the rest held since 2026-10-03 to free slots for E8; restart-safe per cloud).
Pilot: k-means centres form a near-regular lattice -> TwoNN ~ 8 on T2; random partition is the reference.
Table: `experiments/dimension/results/calib_table.csv` (540 rows). torus6+aco hit the 2-day limit on node cn-040 (each measure 3-7 h); rerun as 4376873_22 on cn-010 finished in 15 min. Report: `experiments/dimension/reports/kalibrovka_traektorii.pdf`
(built by `experiments/dimension/code/calib_report.py`).
Key: CBOW + Schweinhart / TwoNN / Hidalgo plateau in d (10 -> 30) for m <= 6 and rank objects correctly, but words are
overestimated (Schweinhart 1.2-1.9 m, TwoNN / Hidalgo 1.5-2.5 m); on bigrams CBOW + TwoNN / Hidalgo at d = 15-30 match the
bigram reference (cell-centre bigrams) within ~10 % for m >= 4. SVD grows with d for every estimator. FisherS is wrong
already on the reference (tori ~1.6 m, spheres ~m + 1). k-means breaks TwoNN / Hidalgo on the reference; aco ~ random.
No plateau for T8 / T10, and 5000 centres are too few for m = 10. Open question: why the word bias is much larger than
on the pseudolanguage (+0.2..0.9).

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
Code: `pseudo_bert.py`, `8c_pseudo_bert.sbatch`. Job 4371985 (8 tasks, 84-124 min training each); rows `bert_pca`/`bert_full`
in `experiments/dimension/results/pseudo_table.csv`. MLM accuracy eps 0.19-0.33; dist m >= 3 ~0.10 (nothing learnable).
Key: mean contextual BERT vectors overestimate and do not separate m = 3/4/6 (eps, TwoNN, PCA 15: 4.5 / 8.0 / 8.9 / 7.9;
full 256: 21 / 29 / 36 / 38); no plateau in d; PCA 15 keeps only 28-48 % of variance. Implies the War-and-Peace ruBERT
numbers are likely overestimates. Report restructured (intro with methods, then one section per estimator + embedding).

## E9. Layer sweep of the nested-PCA intervention (HSE; short done, long running)
pca_32 after each decoder block L = 1..30 (MVP used only L = 15). Per-layer bases from one identity calibration
(24 calibration cards x 2 seeds, MVP cap 128 per dialogue x agent, same positions for all layers).
Short: neutral, 24 cards x seeds 1..50 per layer (36 000). Long: neutral, 8 cards x seed 1103 per layer (240).
Identity controls reused from E5 / E4.
Code: `hse_layer_sweep.py`, `hse_layer_calib.sbatch`, `hse_layer_short.sbatch`, `hse_layer_long.sbatch`.
Jobs: 4372011 (calibration, after E5), 4372012 (short, after calibration); 4372289 (long, 17 workers -- submit limit
100 jobs; lock-file claims, so fewer workers only make it slower). Initially nice 20000; on 2026-10-04 nice -> 0, and on 2026-10-05 `--mem=16G` removed: Slurm nodes report 1 MB of memory, so any --mem request can never be satisfied (job 4372011 had been pending for 1.5 days). Bases: 32 PCs explain 45-71 % of variance per layer. Short sweep 4372012 started 2026-10-05 02:47 MSK, ~2300 dialogues/h.

Short sweep done 2026-10-05 (36 000/36 000, no errors). Text statistics per layer (`short_layer_stats.py`,
`results/short_layer_stats.csv`, neutral frame, 1200 dialogues per layer): intervention at L01-L11 breaks generation
-- 7-62 % empty utterances (peak L08), verbatim repeated utterances 5-14 %, trigram repetition 0.33-0.69, type-token
ratio 0.10-0.20; from L12 on the dialogues are normal (no empty or repeated utterances, trigram repetition mostly
0.01-0.29, TTR 0.22-0.47, 200-460 words). Long sweep 4372289: 16/240 records by 2026-10-06 07:00 MSK, 25/240 by 13:05, no errors.
2026-10-06 02:09 workers _2,_3,_13,_15,_16 requeue-held to give calib3 (E12) the CPUs (Anastasia asked to raise E12's
priority); their units moved to `locks_released`; released 07:05 (they start as calib3 tasks finish).

## E10. Calibration round 2 (Gromov, 2026-10-05; HSE, first priority)
Gromov's three questions: (1) larger embedding dimension and fractal / multifractal objects; (2) effect of the number of
points and of trajectories; (3) a model of the real situation -- m unknown, Whitney / Takens only indirectly (FNN, "FNF"
-- meaning to be clarified with Gromov) -- how to recover m from estimates across d and methods.
Code: `experiments/dimension/code/calib2.py`, `10_calib2_array.sbatch` (job 4377311, tasks 0-28, excludes the slow node cn-040).
Part A (tasks 0-20), V = 5000, 500 trajectories x 1500 steps, random partition, truth / SVD / CBOW d = 5, 10, 15, 30, 50, 100,
words and bigrams, 4 estimators. Objects: T2..T12, S4, S8, S12, Lorenz (2.06), Roessler and Lorenz-96 (N = 6, 10)
with the Kaplan-Yorke dimension computed from the Lyapunov spectrum (Lorenz 2.06, Roessler 2.01 checked), self-similar
fractals with random walks on the level-L cell graph (Sierpinski gasket 1.585, tetrahedron 2, 4-simplex 2.32,
7-simplex 3, carpet 1.893, Menger sponge 2.727), multifractal measures (gasket p = .6/.25/.15: D0 1.585, D1 1.353,
D2 1.168; tetrahedron p = .4/.3/.2/.1: D0 2, D1 1.846, D2 1.737) via a Metropolis walk.
Blind-protocol observables per object: FNN (Kennel, Rtol 2/5/10/15 and Atol 2), Cao E1 / E2 and TwoNN of k-gram clouds
for word sequences (truth, CBOW 15, CBOW 30; k = 1..8) and for the classical scalar delay embedding of the first
coordinate (k = 1..20). The inverse model (observables -> m, leave-one-object-out) is fitted offline.
Part B (tasks 21-28): T4, T6, S4, Lorenz; V = 1000, 2000, 10000, 20000 (500 trajectories) and 100, 250, 1000, 2000
trajectories (V = 5000); CBOW d = 15, 30.
Smoke test on HSE (mftetra, tiny) passed end to end.
FNF (Gromov's reply, 2026-10-05): "false neighbours" per Malinetskii & Potapov 2000, sec. 11.4.2 and 13.3 -- two kinds:
FNN (window w = (m-1)tau too small, vanish as m grows) and FNF, false neighbours on folds (w too large; appear at
scales above eps_fr; the correlation-integral slope then grows ~linearly with m). Added to calib2.py: correlation-integral
slopes at C = 1e-3 / 1e-2 / 1e-1 per k, and -- using the known original states -- the share of reconstruction nearest
neighbours that are far apart in the original space (> 3x / > 10x the true NN distance). Lorenz check (scalar x, tau = 28):
FNN 0.97 / 0.24 / 0.01 for k = 1 / 2 / 3; false-in-original minimal at k = 3-4 (14 %) and rising to 37 % at k = 10 (FNF);
small-scale slope 2.0 for all k >= 3, large-scale slope 1.6 -> 4.6 (FNF signature). Tasks 0-1 (torus2, torus4) had already
started with the previous version: their *_fnn.json must be recomputed after they finish.
Results v1 (2026-10-05 21:00 MSK; `results/calib2_table.csv`, `results/calib2_aux.json`, report `reports/kalibrovka_raund2.pdf`):
- CBOW plateau persists to d = 100 for m <= 8 (T8 TwoNN 14.5 / 14.8 / 14.4 at d = 30 / 50 / 100); for m = 10-12 the plateau
  starts only at d ~ 50 (~2m, Whitney); SVD grows with d up to 100 for every object. Bigram estimates saturate at 12-14 for m >= 8.
- Fractals: Schweinhart on cell centres ~ D0 (gasket 1.5, carpet 1.7, sponge 2.5); TwoNN / Hidalgo ~1 (finite-resolution
  segments). CBOW gives 3.2-4.1 for all fractals with D <= 3 (no discrimination). Multifractal centres ~ D2 rather than D0.
- Flows recovered far better than Brownian walks: Lorenz-96 N = 10 (D_KY 6.55) CBOW bigrams 6.6-6.8.
- Vocabulary size V matters more than corpus length: T4 CBOW words 9.2 -> 6.6 for V 1000 -> 20000.
- FNN (Kennel) -> 0 for every object incl. Brownian; true false-neighbour share and the correlation-integral slope reveal FNF
  (Lorenz min 14 % at k = 3-4, 43 % at k = 12; L96 N = 10 >= 60 %: FNN and FNF coexist); Cao E2 separates flows (0.02-0.25)
  from Brownian walks (~1). k-gram clouds of cell centres saturate at D for flows, grow with k for Brownian walks.
- Blind inverse model (log m ~ log estimate, two groups by E2 at k = 2, leave-one-object-out): median error 13-16 %, max ~40 %.

## E11. Which calibration group is natural language? (2026-10-06, HSE Jupyter)
Cao E2 / FNN / k-gram slopes (`calib2.delay_stats`) on CBOW vectors of War and Peace lemmas in book order (200-word chunks,
1254 chunks) vs. the same chunks with words shuffled. Code `experiments/dimension/code/wap_delay.py`, output `~/wap_run/wap_delay.json`.
- CBOW d = 15: E2(k = 2, 3, 4) = 0.996 / 0.997 / 1.017 (shuffled 1.002 / 0.995 / 0.997); slope at C = 0.01 for k = 1..8:
  2.3 -> 12.5 (shuffled 2.0 -> 13.2); k-gram TwoNN at k = 8: 24.6 (shuffled 27.8).
- CBOW d = 30: E2 1.012 / 0.996 / 1.025; slope 2.3 -> 12.0 (shuffled 2.0 -> 13.4).
=> War and Peace belongs to the Brownian group (E2 ~ 1, k-gram dimension grows ~1.4 per word), indistinguishable from
shuffled order at this resolution apart from a slightly slower k-gram growth. Not a low-dimensional deterministic flow.
Calibration (Brownian group, CBOW d = 15 TwoNN, comparable N ~ 10k bigrams: 6.6 -> m ~ 4-5; full N = 152k: 9.3 -> ~6.5)
-- rough, the TwoNN value depends on N.
Short dialogues, neutral frame (quick run, d = 30): E2 = 0.92 (dedup 0.94, shuffled 1.00) -- also Brownian group, at its edge.
Full dialogue run (job 4378690, `dialogue_delay.py`): E2(k = 2) real / dedup / shuffled --
neutral 0.90-0.92 / 0.92-0.94 / 0.99-1.00, competition 0.91-0.92 / 0.92-0.95 / 1.00-1.01, cooperation 0.88 / 0.91-0.94 / 1.00
(d = 15-30). Dialogues sit at the Brownian/flow boundary; verbatim repeats pull E2 down (removing them gives 0.91-0.95).

## E12. Robustness of the CBOW over-estimation factor inside the Brownian group (2026-10-06, done)
Question (Anastasia): is the factor 1.5-2.5 universal? One factor at a time on T2, T4, T6, S4, S8: step x0.5/2/3,
Zipf-like non-uniform cells (kappa 1.5/3/5, achieved rank-frequency slope recorded; War and Peace-like skew at kappa ~5),
V = 18000, sentence length 10/40, CBOW window 2/10; truth + CBOW d = 15/30, words and bigrams, 4 estimators plus TwoNN on
all bigrams (War and Peace comparability). Code `calib3.py`, `12_calib3_array.sbatch`, job 4378998 (30 tasks). Smoke test OK.

Done 2026-10-06 (30/30 tasks, 60 configurations, 360 clouds). `calib3_agg.py` -> `results/calib3_table.csv`;
report `experiments/dimension/reports/ustoichivost_mnozhitelya_E12.pdf` (`calib3_report.py`). Factor = estimate / m (bigrams / 2m).
- Words, CBOW d = 15, TwoNN: 1.24-3.78 over all variants (d = 30: up to 4.15). Baseline 1.5 (S8) .. 2.4 (T2), falls with m.
  Hidalgo ~ TwoNN; Schweinhart 1.1-2.2.
- No effect (<= 0.2): sentence length 10/20/40, CBOW window 2/5/10, step x0.5.
- Effects: V = 18000 lowers it (1.2-1.9); step x2/x3 raises it (d = 15 up to 2.3, d = 30 up to 3.9); Zipf-like frequencies
  raise it only for small m (T2 2.4 -> 2.9 -> 3.8 at kappa 3 / 5).
- Bigrams: truth bigram clouds give only 0.5-0.8 of 2m (consecutive points close); CBOW adds 10-30 %; Zipf skew lowers
  further (kappa 5: 0.5-0.6 of 2m).
- Achieved rank-frequency slope: base -0.15, kappa 1.5/3/5: -0.50/-0.94/-1.19 (top-1 % share 0.06/0.14/0.25).
- War and Peace reference, same code (`~/wap_run/wap_c3ref.py`): V 17900, slope -1.20, top-1 % share 0.44; CBOW d = 15 TwoNN
  words (5000 sample) 12.95, bigrams (10 000) 6.6, all 152 568 bigrams 9.3.
- Inverting log(est) ~ log m per variant: words -> m 6.9-13.2 (base 8.5, V18000 10.1, kappa 5 13.2 but slope 0.31 -> unstable);
  bigrams (10k) -> 2.8-6.2; all bigrams -> 5.5-8.6. Words and bigrams disagree ~2x => the Brownian cell model describes text
  only partly; one factor cannot give m of language. Working interval m ~ 5-10.
- Next: calibration matched to War and Peace simultaneously (V ~ 18k, slope -1.2, top-1 % 44 %, ~150k bigrams, m = 4-12);
  if words/bigrams still disagree, a model with memory is needed.

## E13. Bigram estimate / m as a function of rho = delta / r (2026-10-06, running)
Question (Anastasia): is the bigram over-estimation always 1.3-2 x m? Hypothesis: for Brownian walks bigram estimate / m =
f(rho), rho = |x_{t+1} - x_t| / nearest-neighbour distance in the bigram cloud, rising from 1 (pairs hug the diagonal) to 2,
the same curve for every m; for deterministic flows ~1 (L96 N = 10 bigrams 6.6-6.8 at D_KY 6.55, E10).
Part 1 (exact coordinates, no CBOW): T2..T12, S4, S8, S12; step x0.1..x5 (10 values); raw consecutive pairs and V = 5000
cell bigrams; N = 1k..100k; TwoNN, rho, words TwoNN, correlation slopes at 3 scales, Hidalgo at N = 3000.
Part 4: War and Peace CBOW d = 5/10/15, the same quantities for words and all 152 568 bigrams at N = 1k..all.
Parts 2-3 (CBOW on the same grid; flows and fractal walks) only if part 1 collapses onto one curve.
Code `calib4.py`, `13_calib4_array.sbatch` (md5 bcc083b7..., 3d1d1446...), job 4380885 (10 tasks, 4 CPU each).
First numbers (T2, raw pairs, step x0.5): N 1k -> 100k gives rho 0.20 -> 1.04 and TwoNN / m 1.03 -> 1.89 -- as predicted.
Cell bigrams behave differently (T2 step x0.35: rho 2.0 at N = 30k but TwoNN / m = 1.0) -- to be understood.

## Synthesis: accuracy of CBOW + estimator (2026-10-06)
All calibrations pooled (E6, E7, E10, E12; 27 objects, ~1500 clouds; `results/calib_pooled.csv`, `code/calib_synthesis.py`).
Report `experiments/dimension/reports/tochnost_cbow_metodov.pdf`.
- Estimators alone on exact points: words TwoNN / Hidalgo / Schweinhart = 1.00 / 1.00 / 1.02 m (10-90 %: within +-5..12 %);
  FisherS 1.56 m (unusable). Bigrams on exact points: 1.41 / 1.47 / 1.58 m (not 2m).
- CBOW d >= 15 (Brownian objects, 15): words TwoNN 1.97 m (10-90 %: 1.50-2.46), Hidalgo 2.00, Schweinhart 1.62 (1.33-1.91);
  bigrams 1.63 / 1.74 / 1.96 m. Ratio falls with m (Spearman -0.5..-0.73). d = 5 underestimates (0.7 m).
- Corrected (log-log inverse model), leave-one-object-out: median error 6-17 %, 90 % within 21-36 %.
- Leave-one-condition-family-out: partition / T / sentence / window within +-7 %; V18000 words TwoNN/Hid -25..-32 %;
  Zipf bigrams TwoNN/Hid -15..-36 %; held-out generator (pseudolanguage E7) -34..-56 % for every method.
- War and Peace through the pooled model: words 7-9, bigrams 3.4-4.4 -> disagreement beyond the +-25 % in-family error,
  i.e. text outside the calibration family; m ~ 3.5-9 compatible with all methods. Schweinhart is the most stable estimator.
