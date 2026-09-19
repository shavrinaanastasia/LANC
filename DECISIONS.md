# Decisions

## 2026-09-19: Initial MVP implementation

The repository was initially empty apart from Git attributes. The preregistered design in `RESEARCH.md` was implemented without a scientific design deviation. Full model execution is deliberately not simulated when the local Python/runtime dependencies are unavailable.

## 2026-09-19: Resume policy

The prompt permits resume only with record hash checking. This MVP refuses existing run directories rather than offering an unsafe partial resume. Record-level verified resume remains implementation work, not a change to scientific estimands.

## 2026-09-19: Per-turn RNG implementation

`transformers==4.43.3` rejects `generator=` for this SmolLM2 generation implementation. The code uses `torch.random.fork_rng` and seeds the fork from the preregistered `(generation_seed, stimulus_id, turn_index, speaker)` mapping instead. The random streams remain isolated from global experiment state and condition-free; the API-level replacement does not change the experimental seed mapping or the estimand.
