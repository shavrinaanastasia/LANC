# Decisions

## 2026-09-19: Initial MVP implementation

The repository was initially empty apart from Git attributes. The preregistered design in `RESEARCH.md` was implemented without a scientific design deviation. Full model execution is deliberately not simulated when the local Python/runtime dependencies are unavailable.

## 2026-09-19: Resume policy

The prompt permits resume only with record hash checking. This MVP refuses existing run directories rather than offering an unsafe partial resume. Record-level verified resume remains implementation work, not a change to scientific estimands.

## 2026-09-19: Per-turn RNG implementation

`transformers==4.43.3` rejects `generator=` for this SmolLM2 generation implementation. The code uses `torch.random.fork_rng` and seeds the fork from the preregistered `(generation_seed, stimulus_id, turn_index, speaker)` mapping instead. The random streams remain isolated from global experiment state and condition-free; the API-level replacement does not change the experimental seed mapping or the estimand.

## 2026-09-20: Calibration vector-cap implementation error

The first full Colab calibration run `20260919T124847Z-7aaef464f6` sampled up to 128 layer-15 vectors per generated utterance rather than no more than 128 vectors per `dialogue x agent` block. Its 20,934-vector PCA artifact therefore violates the preregistered memory cap. Phase 2 run `20260919T150055Z-0b8a8b09a6`, which used that artifact, is retained as exploratory and must not be described as confirmatory MVP evidence. The implementation now concatenates all five generated utterance segments for one agent in a dialogue and then deterministically samples at most 128 vectors once. A fresh calibration and Phase 2 run are required.

## 2026-09-26: Gromov Appendix-A synthetic validation

The Gromov, Borodin, and Yerbolova paper does not disclose the sample-size ladder used for its synthetic controls. The separate `schweinhart-controls` command therefore uses an explicitly supplied, immutable ladder rather than claiming an exact paper replication. It implements the published alpha sweep and MST-growth relation on sphere, Swiss roll, Sierpinski carpet, and Menger sponge controls. These known-shape controls validate numerical behaviour only; they do not alter the preregistered dialogue-PHD estimand.

## 2026-09-26: Exploratory long Phase 1 dialogue generation

The requested long-dialogue study is a separate exploratory extension, not a replacement for preregistered Phase 1. It preserves the 24 evaluation stimuli, three interaction frames, three generation seeds, model revision, sampling settings, and prompt texts. Each of the resulting 216 dialogues stops at no fewer than 10,000 generated English-style words. Because SmolLM2 has an 8,192-token context window, the generator retains only the most recent complete chat messages whose rendered prompt occupies at most 6,000 tokens. It records dropped-message counts, largest prompt length, generated words, turns, and completion status per dialogue. This rolling-context procedure changes the long-run conversational process and results must not be described as the original ten-turn MVP.

## 2026-09-26: Long-dialogue resumability

Unlike the original MVP commands, the exploratory long-dialogue command writes one immutable, content-hashed record after each completed dialogue. Resume requires the same explicit `run_id`, matching configuration hash, matching scope (frames, stimuli, and seeds), and valid hashes for every existing record. It never overwrites a run manifest, dialogue record, or final JSONL file. This exception is required because the GPU run is expected to outlast an individual Colab session.

## 2026-09-26: Colab Python 3.13 environment

Current Colab runtimes use Python 3.13, while the repository's locked local dependency set targets Python 3.10-3.12 and includes numerical wheels unavailable for Python 3.13. The exploratory GPU notebook therefore preserves Colab's CUDA PyTorch and preinstalled numerical stack, installs a current compatible Transformers version in the 4.x line, and installs the local package with `--no-deps --ignore-requires-python`. The run manifest records the actual package versions. This is an environment deviation for the exploratory long-dialogue extension only; it does not alter the original pinned MVP results.

## 2026-09-26: Separate remainder run from unit 26

The initial long-dialogue run saved 25 immutable records before GPU access became unavailable. One record, global unit 13 (`competition`, `E05`, seed `1103`), has status `incomplete` after reaching the 800-turn safety limit and is retained without deletion. A new account may generate global units 26-216 under a distinct `run_id` with `--start-unit-index 26`. This is a separate remainder run with 191 records, not a hash-verified resume of the original 216-unit run. Any later combined dataset must retain source-run provenance, preserve the incomplete unit 13, and be reported as a split-run exploratory extension.

## 2026-09-29: War and Peace source edition

The exploratory single-novel adaptation uses the user-provided Russian plain-text file `Война и мир.txt`, SHA-256 `14935bc6b407949ed8d375ee6e725370ed55ed3233205c20e7a2ac17c9e373c2`, downloaded from `https://librabook.lat/download/voyna-i-mir/`. The source site's edition and license terms were not independently verified. The protocol is a chapter-as-document adaptation, not a replication of Gromov et al.'s 6,429-text Russian literature corpus.
