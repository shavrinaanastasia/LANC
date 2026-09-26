# LANC MVP Report

## Status

The preregistered two-agent MVP Phase 1 was executed on a Google Colab Tesla T4 using the pinned `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The external analysis encoder was pinned `roberta-base` revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. A subsequent implementation audit found that the Phase 2 calibration artifact violated its preregistered per-dialogue-agent vector cap; the completed Phase 2 run is therefore exploratory and requires rerunning after corrected calibration.

The full experimental data are present locally in ignored run directories:

| Component | Run / artifact | Scope |
| --- | --- | --- |
| PCA calibration | `results/20260919T124847Z-7aaef464f6` | Superseded: violated the 128-vectors-per-dialogue-agent cap |
| PCA basis | `artifacts/calibration/20260919T124847Z-7aaef464f6/pca_basis.npz` | Exploratory only; not preregistration-compliant |
| Phase 1 | `results/20260919T132914Z-24e0c18acf` | 216 dialogues: 24 stimuli x 3 frames x 3 seeds |
| Phase 2 | `results/20260919T150055Z-0b8a8b09a6` | Exploratory: 432 dialogues using the superseded PCA basis |
| Corrected Phase 1 inference | `results/20260919T192403Z-phase1-statistics-correction` | Immutable statistical reanalysis |
| Corrected Phase 2 inference | `results/20260919T192403Z-phase2-statistics-correction` | Immutable statistical reanalysis |

The dialogue-level PHD unit was a cloud of contextual embeddings from generated tokens only. Prompts, stimulus text, system messages, and speaker labels were excluded. Generation seeds were aggregated within each `stimulus_id x condition`; primary inference used 24 paired stimulus blocks, not 72 dialogue values as independent observations.

## Estimator And Validity

Primary PHD used alpha 1, `n_min=40`, eight sample sizes, seven subsets per size, three internal regression seeds, and `d = 1 / (1 - kappa)`. It used unnormalised, uncentred RoBERTa contextual token embeddings. The PHD validity gate passed for every generated dialogue.

| Phase | Condition(s) | Valid PHD | Invalid PHD | Token-truncated dialogues |
| --- | --- | ---: | ---: | ---: |
| 1 | competition, cooperation, neutral | 72 each | 0 each | 3 total |
| 2 | identity, pca_128, pca_32, pca_8, pca_2, pca_1 | 72 each | 0 each | 1 total |

No condition crossed the preregistered 20% invalid-PHD `DEGENERATED_OR_UNSTABLE` gate. This does not itself establish language quality; the degeneration controls below remain necessary.

## Phase 1

Mean dialogue PHD values were 5.627 for competition, 4.933 for cooperation, and 6.083 for neutral. All primary contrasts used seed-aggregated paired stimulus values (`n=24`).

| Contrast | Mean difference | 95% bootstrap CI | Holm-adjusted sign-flip p | Paired dz |
| --- | ---: | --- | ---: | ---: |
| competition - neutral | -0.457 | [-1.090, 0.140] | 0.1775 | -0.287 |
| cooperation - neutral | -1.150 | [-1.742, -0.565] | 0.0039 | -0.760 |
| competition - cooperation | 0.694 | [0.120, 1.269] | 0.0698 | 0.466 |

The supported primary result is lower dialogue PHD in cooperation than neutral conversation under this model, prompts, and estimator. Competition-neutral and competition-cooperation do not pass the Phase 1 Holm threshold.

Phase 1 degeneration summaries are directionally relevant: cooperation had more repeated 3-grams (0.506 versus 0.371 neutral), lower distinct-2 (0.442 versus 0.560), greater self-copy (0.585 versus 0.459), and a longer maximal repeated substring (83.3 versus 57.0 words). Therefore the cooperation-neutral PHD difference cannot be interpreted as a pure interaction-frame geometry effect independent of output degeneration.

## Exploratory Phase 2 Degeneration Panel

This panel precedes Phase 2 PHD inference. Values are dialogue-level condition means. Repetition, copy, and looping are controls, not inferential outcomes.

| Condition | Mean PHD | Repeated 3-gram fraction | Distinct-2 | Self-copy rate | Cross-speaker copy rate | Max repeated substring (words) |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 6.083 | 0.371 | 0.560 | 0.459 | 0.411 | 57.2 |
| pca_128 | 4.079 | 0.644 | 0.280 | 0.678 | 0.618 | 68.2 |
| pca_32 | 6.949 | 0.115 | 0.685 | 0.300 | 0.303 | 5.3 |
| pca_8 | 6.846 | 0.473 | 0.274 | 0.644 | 0.644 | 8.3 |
| pca_2 | 6.701 | 0.356 | 0.455 | 0.252 | 0.248 | 13.6 |
| pca_1 | 7.853 | 0.200 | 0.598 | 0.249 | 0.248 | 9.2 |

`pca_128` has substantially higher repetition and copy than identity. `pca_8` has elevated repetition and very high self/cross-speaker copy. Any PHD comparison involving those conditions requires an explicit degeneration qualification. `pca_32`, `pca_2`, and `pca_1` do not show the same pattern in these controls, but no intervention can be interpreted as a direct measure of linguistic richness from PHD alone.

## Exploratory Phase 2 PHD Inference

All contrasts use 24 seed-aggregated paired stimulus blocks. Identity is not treated as PCA-576 in an ordered-rank trend.

| Contrast | Mean difference | 95% bootstrap CI | Holm-adjusted sign-flip p | Paired dz |
| --- | ---: | --- | ---: | ---: |
| pca_128 - identity | -2.005 | [-2.523, -1.493] | 0.0005 | -1.545 |
| pca_32 - identity | 0.866 | [0.312, 1.422] | 0.0177 | 0.621 |
| pca_8 - identity | 0.762 | [0.231, 1.306] | 0.0206 | 0.557 |
| pca_2 - identity | 0.618 | [0.088, 1.150] | 0.0366 | 0.457 |
| pca_1 - identity | 1.770 | [1.264, 2.281] | 0.0005 | 1.382 |

Every prespecified contrast passes Holm correction in this exploratory run. The pattern is non-monotonic: `pca_128` is lower than identity, while ranks 32, 8, 2, and 1 are higher. This is a useful debugging observation, but it is not confirmatory Phase 2 evidence because the PCA calibration artifact violated its vector-cap protocol.

The `pca_128` decrease co-occurs with a strong repetition/copy increase and must not be presented as clean evidence for lower linguistic dimensionality. The `pca_8` increase also co-occurs with strong copying. Effects for `pca_32`, `pca_2`, and `pca_1` are less visibly confounded by the recorded lexical degeneration controls, but remain model-, layer-, basis-, prompt-, and encoder-specific.

## Why Degeneration Can Raise Or Lower PHD

PHD in this study is not a count of unique words, semantic topics, model neurons, or literal dimensions of language. It is a scaling estimate for the Euclidean minimum-spanning-tree geometry of contextual RoBERTa token embeddings. A repeated surface word can still receive different contextual embeddings because its location, neighbouring tokens, and full dialogue context differ.

`pca_128` illustrates a comparatively structured degeneration. The matched dialogue repeatedly returns to closely related templates such as "a person that is a person" and "The result is not a product." This is consistent with high repeated-3-gram fraction, low distinct-2, and high self/cross-speaker copy. If many contextual token embeddings fall into the same nearby regions, the MST is joined by relatively short edges and its energy can grow more slowly with subset size. The fitted slope kappa is then lower, and `d = 1 / (1 - kappa)` is lower. This is a plausible explanation for the lower `pca_128` PHD, not direct evidence that its language is intrinsically lower-dimensional in a semantic sense.

Very severe restrictions can instead produce unstable token-level variation. A text may look nonsensical to a reader while still placing contextual embeddings in several locally distinct regions because token sequences, punctuation, positions, and residual context vary. The MST scaling slope can then move closer to one, increasing PHD. This explains why a high PHD, including the high `pca_1` mean, cannot be interpreted as high language quality or richness. If all token embeddings were literally identical, the MST energy would collapse toward zero and PHD would be invalid rather than high.

The current design cannot identify the exact internal mechanism. Testing it requires the roadmap controls: random orthonormal and bottom-PCA subspaces, more hook layers, hidden-state trajectory analysis, and modelling PHD jointly with degeneration metrics.

## Calibration Protocol Deviation

The preregistered calibration rule is to pool all layer-15 hidden vectors from the five generated utterances of one agent in one dialogue, then deterministically retain no more than 128 vectors for that `dialogue x agent` block. The first implementation applied the 128-vector cap separately to each utterance. Because each utterance had at most 48 generated tokens, this retained more than 128 vectors for many dialogue-agent blocks and yielded a 20,934-vector calibration cloud.

The implementation has been corrected and tested to pool all five segments before applying the cap once. A new calibration artifact must be fit from the 48 disjoint calibration dialogues, followed by a new Phase 2 run and statistical analysis. The existing Phase 1 results are unaffected. The old Phase 2 raw data and reanalysis are retained as transparent exploratory outputs, not deleted or relabelled as confirmatory evidence.

## Statistical Correction

The original Colab `statistics.json` files incorrectly reported sign-flip p-values of 1.0. The null statistic was implemented as the mean of absolute signed differences, rather than the absolute mean of signed differences. This makes the null statistic at least the observed absolute mean by construction.

The implementation now correctly computes `abs(mean(sign * paired_difference))`. A regression test with 16 consistently nonzero paired differences was added. The immutable corrected reanalysis directories above were generated from the saved `analysis.jsonl` tables; raw dialogue records and PHD values were not altered. The corrected files, not the initial `statistics.json`, are authoritative for inference in this report.

## Gromov Protocol

No Gromov-aligned pooled result is reported. The optional pooled module hard-fails with `INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL` unless its predeclared sample-size ladder is supported. This MVP does not provide a corpus-scale unique word/n-gram cloud compatible with that protocol, and no exploratory number is substituted.

## Limitations And Next Steps

- The PCA basis was calibrated on disjoint stimuli, but only one basis and one intervention layer were tested.
- The model and external encoder are frozen and pinned, but conclusions are not general statements about language or all language models.
- Generation seeds are technical repeats rather than independent primary replicates; inference correctly uses paired stimuli.
- PHD is a geometric estimator, not evidence that an output is richer, better, safer, or more meaningful.
- Degeneration controls are lexical/semantic diagnostics, not a complete causal adjustment.
- Follow `ROADMAP.md` for random-subspace and bottom-PCA controls, multiple hook layers, PCA stability, higher-order topology, and broader model/human baselines.
