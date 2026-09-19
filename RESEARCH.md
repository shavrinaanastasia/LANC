# Research Design

## Question

Two logical agents share frozen `HuggingFaceTB/SmolLM2-135M-Instruct` weights but have independent prompts, message histories, and per-turn RNG generators. Does interaction framing alter dialogue PHD? In neutral dialogue, does a nested PCA projection after decoder block 15 (human numbering; `model.layers[14]` zero-based) alter PHD?

```text
interaction frame / PCA rank --> generated dialogue --> contextual RoBERTa token cloud --> PHD
                                  |                                      |
                                  +--> degeneration controls <------------+
stimulus ------------------------> paired blocking
```

Phase 1 compares `competition`, `cooperation`, and `neutral` under identity. Phase 2 compares neutral `identity`, `pca_128`, `pca_32`, `pca_8`, `pca_2`, and `pca_1`. All but intervention are locked in `configs/base.yaml`.

## Point, Cloud, And Replicate

- A point is one non-special contextual `roberta-base` token embedding from generated dialogue text.
- A cloud is the 50-510 content-token cloud of one complete dialogue.
- A dialogue yields exactly one PHD result.
- Generation seeds are technical repeats, averaged within `stimulus_id x condition`.
- Stimuli, not seeds or PHD subset seeds, are independent paired blocks for primary inference.

Ten utterances are not ten viable PHD points. The Appendix-B estimator needs `n_min=40`, eight scales, and content-token eligibility `N >= 50`; ten high-dimensional utterance embeddings have rank at most nine and an unstable MST. Sequential turns are dependent, and a single dialogue cannot characterize a condition distribution.

## Estimand And Gates

PHD follows Tulchinskii et al.: alpha 1, eight sample sizes from 40 to N, seven random subsets per size, median MST length, three regression seeds, and `d = 1/(1-kappa)`. Primary token embeddings are not normalized, whitened, centred, or PCA-reduced. For N above 510, text-order token embeddings are deterministically truncated and flagged. For N below 50, PHD is invalid rather than imputed.

The condition effect is the paired mean difference of seed-averaged dialogue PHD across 24 stimuli. Primary contrasts are prespecified in `HYPOTHESES.md`. Invalidity above 20% labels a condition `DEGENERATED_OR_UNSTABLE`; no claim about dimensional reduction is made without that qualification.

Gromov pooled analysis is different: it pools unique unigrams/bigrams into a corpus cloud and uses an MST growth ladder. It is not a replication from ten dialogues. The optional implementation hard-fails with `INSUFFICIENT_SAMPLE_FOR_GROMOV_PROTOCOL` below its predeclared ladder.
