"""Layer sweep of the nested-PCA intervention: pca_32 after each of the 30 decoder blocks in turn.

Generalises the MVP intervention (pca_k after human layer 15 = model.layers[14]) to every layer L = 1..30.
Exploratory, see DECISIONS.md 2026-10-03.

1. calibrate  -- one identity calibration on the disjoint calibration stimuli (24 cards x seeds 1103, 2207,
                 neutral, 10 turns, as configs/calibration.yaml). For every generated position the outputs of ALL
                 30 blocks are captured with forward hooks (pre-final-norm, the same tensor the intervention
                 replaces). Per dialogue x agent the five segments are pooled and at most 128 positions are
                 kept (MVP rule; the same positions for every layer); one PCA basis per layer is fitted.
                 -> artifacts/calibration/layer-sweep-v1/pca_basis_L01.npz ... L30.npz
2. short      -- 10-turn dialogues (MVP protocol), neutral, 24 evaluation cards x seeds 1..50, pca_32 at layer L,
                 for L = 1..30 (36 000 dialogues, corpus for per-layer CBOW). Identity control = the neutral part
                 of run short-10turn-cbow-v1 with the same seeds. Shards results/<run>/short_shards/wNNNN.jsonl.
3. long       -- 10 000-word dialogues (phase1_long protocol, rolling 6000-token context), neutral,
                 8 cards (E01 E02 E05 E06 E09 E13 E17 E21: every stratum, two twice) x seed 1103, pca_32 at L,
                 L = 1..30 (240 dialogues). Identity control = run phase1-long-10000w-hse-v1 (same cards/seed).
                 Lock-file claims, one record file per unit.

  python hse_layer_sweep.py calibrate
  python hse_layer_sweep.py prepare
  python hse_layer_sweep.py short --worker K --workers M
  python hse_layer_sweep.py long
  python hse_layer_sweep.py status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:  # some cHARISMa nodes lack libsndfile; transformers then crashes importing audio utils we never use
    import soundfile  # noqa: F401
except (OSError, ImportError):
    sys.modules["soundfile"] = None

from agent_language_geometry.config import load_config  # noqa: E402
from agent_language_geometry.reproducibility import (  # noqa: E402
    canonical_json, file_sha256, git_state, sha256, software_manifest, write_manifest,
)

RUN_ID = "layer-sweep-pca32-v1"
BASIS_DIR = ROOT / "artifacts" / "calibration" / "layer-sweep-v1"
RANK = 32
LAYERS = list(range(1, 31))          # human numbering; hook on model.model.layers[L - 1]
SHORT_SEEDS = list(range(1, 51))
LONG_CARDS = ["E01", "E02", "E05", "E06", "E09", "E13", "E17", "E21"]
LONG_SEED = 1103


def run_dir():
    return ROOT / "results" / RUN_ID


def eval_cards():
    from hse_dialogues import cards_of
    return cards_of(load_config(ROOT / "configs" / "base.yaml"))


def basis_path(layer):
    return BASIS_DIR / f"pca_basis_L{layer:02d}.npz"


class LayerPCA:
    """pca_RANK projection of the output of decoder block `layer` (human numbering), every prefill/decode call."""

    def __init__(self, layer, basis, rank):
        from agent_language_geometry.interventions import LayerIntervention
        self.layer, self.inner = layer, LayerIntervention(f"pca_{rank}", basis)
        self.handle = None

    def install(self, model):
        self.handle = model.model.layers[self.layer - 1].register_forward_hook(
            lambda _m, _a, out: self.inner.transform(out))
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ----------------------------------------------------------------------------------------------- calibration
def calibrate(threads, limit_cards=0, out_dir=None):
    global BASIS_DIR
    if out_dir:
        BASIS_DIR = Path(out_dir)
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from agent_language_geometry.calibration import deterministic_vector_sample
    from agent_language_geometry.dialogue import generate_dialogue
    from agent_language_geometry.interventions import PCABasis
    from agent_language_geometry.model_runtime import load_runtime

    if threads:
        torch.set_num_threads(threads)
    config = load_config(ROOT / "configs" / "calibration.yaml")
    cards = [json.loads(x) for x in (ROOT / config["paths"]["calibration_stimuli"]).read_text("utf-8").splitlines() if x]
    if {c["stimulus_id"] for c in cards} & {c["stimulus_id"] for c in eval_cards()}:
        raise ValueError("calibration and evaluation stimuli must be disjoint")
    seeds = config["generation"]["seeds"]
    if limit_cards:  # smoke test only
        cards, seeds = cards[:limit_cards], seeds[:1]
    BASIS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_runtime(config["model"], device="cpu")
    n_layers = len(model.model.layers)
    assert n_layers == 30, n_layers
    cap, sseed = config["calibration"]["max_vectors_per_dialogue_per_agent"], config["calibration"]["sampling_seed"]
    per_layer = [[] for _ in range(n_layers)]
    counts, t0 = [], time.time()

    @torch.inference_mode()
    def all_layers(prompt_ids, ids):
        seq = torch.cat([prompt_ids[0], torch.tensor(ids, device=prompt_ids.device)]).unsqueeze(0)
        cap_ = [None] * n_layers
        hs = [model.model.layers[i].register_forward_hook(
            lambda _m, _a, out, i=i: cap_.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
            for i in range(n_layers)]
        try:
            model(seq, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return np.stack([c[0, prompt_ids.shape[1]:].float().cpu().numpy() for c in cap_])  # (30, T, 576)

    for card in cards:
        for seed in seeds:
            segs = {"A": [], "B": []}
            generate_dialogue(model, tokenizer, card, "neutral", seed, config["generation"],
                              on_generated_tokens=lambda p, ids, spk, _t: segs[spk].append(all_layers(p, ids)))
            for spk, chunks in segs.items():
                allv = np.concatenate(chunks, axis=1)                  # (30, T_total, 576)
                idx = np.arange(allv.shape[1])
                keep = deterministic_vector_sample(idx[:, None], cap, sseed, f"{card['stimulus_id']}|{seed}|{spk}")[:, 0]
                for i in range(n_layers):
                    per_layer[i].append(allv[i, keep])
                counts.append({"stimulus_id": card["stimulus_id"], "seed": seed, "speaker": spk,
                               "available": int(allv.shape[1]), "selected": int(len(keep))})
            print(f"calibration {card['stimulus_id']} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    info = {}
    for i in range(n_layers):
        cloud = np.concatenate(per_layer[i]).astype(np.float32)
        pca = PCA(n_components=min(cloud.shape), svd_solver="full", random_state=0).fit(cloud)
        PCABasis(pca.mean_.astype(np.float32), pca.components_.astype(np.float32),
                 pca.explained_variance_.astype(np.float32)).save(basis_path(i + 1))
        evr = pca.explained_variance_ratio_
        info[f"L{i + 1:02d}"] = {"n_vectors": int(len(cloud)), "sha256": file_sha256(basis_path(i + 1)),
                                 "explained_top32": float(evr[:RANK].sum()), "explained_top8": float(evr[:8].sum())}
    (BASIS_DIR / "manifest.json").write_text(canonical_json(
        {"counts": counts, "layers": info, "model_revision": config["model"]["revision"],
         "config_hash": config["_config_hash"], "git": git_state(ROOT),
         "stimuli_hash": file_sha256(ROOT / config["paths"]["calibration_stimuli"]),
         "rule": "pool 5 segments per dialogue x agent, <=128 positions (same positions for all layers)"}) + "\n")
    print("bases written", {k: round(v["explained_top32"], 3) for k, v in info.items()}, flush=True)


# ----------------------------------------------------------------------------------------------- plan / units
def prepare():
    rd = run_dir()
    if rd.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run directory: {rd}")
    if not all(basis_path(l).exists() for l in LAYERS):
        raise FileNotFoundError("run `calibrate` first")
    rd.mkdir(parents=True)
    (rd / "short_shards").mkdir(); (rd / "long_records").mkdir(); (rd / "locks").mkdir()
    base, long_cfg = load_config(ROOT / "configs" / "base.yaml"), load_config(ROOT / "configs" / "phase1_long_hse.yaml")
    plan = {"run_id": RUN_ID, "rank": RANK, "layers": LAYERS, "frame": "neutral",
            "short": {"stimulus_ids": [c["stimulus_id"] for c in eval_cards()], "seeds": SHORT_SEEDS,
                      "order": "seed-major (seed, layer, card)", "identity_control": "short-10turn-cbow-v1 neutral"},
            "long": {"stimulus_ids": LONG_CARDS, "seed": LONG_SEED, "order": "layer-major",
                     "identity_control": "phase1-long-10000w-hse-v1 neutral"},
            "bases": json.loads((BASIS_DIR / "manifest.json").read_text())["layers"]}
    (rd / "plan.json").write_text(canonical_json(plan) + "\n")
    write_manifest(rd, {"run_id": RUN_ID, "config_hash_short": base["_config_hash"],
                        "config_hash_long": long_cfg["_config_hash"], "plan": plan, "git": git_state(ROOT),
                        "execution": "HSE cHARISMa, Slurm CPU workers (hse_layer_sweep.py)"})
    print(f"prepared {rd}: {len(short_units())} short, {len(long_units())} long", flush=True)


def short_units():
    cards = eval_cards()
    return [(L, c, s) for s in SHORT_SEEDS for L in LAYERS for c in cards]


def long_units():
    by = {c["stimulus_id"]: c for c in eval_cards()}
    return [(L, by[sid], LONG_SEED) for L in LAYERS for sid in LONG_CARDS]


def skey(L, card, seed):
    return f"L{L:02d}__{card['stimulus_id']}__{seed}"


def short_done():
    keys = set()
    for shard in (run_dir() / "short_shards").glob("*.jsonl"):
        for line in shard.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(f"{r['intervention'][:3]}__{r['stimulus_id']}__{r['generation_seed']}")
    return keys


# ----------------------------------------------------------------------------------------------- workers
def _load(device, threads):
    import torch
    from agent_language_geometry.interventions import PCABasis
    from agent_language_geometry.model_runtime import load_runtime
    if threads:
        torch.set_num_threads(threads)
    model, tokenizer = load_runtime(load_config(ROOT / "configs" / "base.yaml")["model"], device=device)
    return torch, model, tokenizer, PCABasis


def work_short(worker, workers, threads, device):
    from agent_language_geometry.dialogue import generate_dialogue
    from agent_language_geometry.schemas import DialogueRecord
    config = load_config(ROOT / "configs" / "base.yaml")
    mine = [u for i, u in enumerate(short_units()) if i % workers == worker]
    done = short_done()
    todo = [u for u in mine if skey(*u) not in done]
    print(f"short worker {worker}/{workers}: {len(mine)} units, {len(todo)} to do", flush=True)
    if not todo:
        return
    torch, model, tokenizer, PCABasis = _load(device, threads)
    bases = {}
    gen_hash = sha256(canonical_json({"generation": config["generation"]}))
    job = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")}
    shard = run_dir() / "short_shards" / f"w{worker:04d}.jsonl"
    t_all = time.time()
    with shard.open("a", encoding="utf-8") as fh:
        for n, (L, card, seed) in enumerate(todo, 1):
            if L not in bases:
                bases[L] = PCABasis.load(basis_path(L))
            started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
            hook = LayerPCA(L, bases[L], RANK).install(model)
            try:
                utt = generate_dialogue(model, tokenizer, card, "neutral", seed, config["generation"])
            finally:
                hook.remove()
            rec = DialogueRecord(
                RUN_ID, "layer_sweep_short", card["stimulus_id"], sha256(canonical_json(card)), "neutral",
                f"L{L:02d}_pca_{RANK}", RANK, seed, utt, config["model"]["revision"], gen_hash, git_state(ROOT),
                software_manifest(), started, datetime.now(timezone.utc).isoformat(), "complete", None,
                protocol_metadata={"layer_human": L, "layer_zero": L - 1, "hook_calls": hook.inner.calls,
                                   "basis_sha256": file_sha256(basis_path(L)), "slurm": job,
                                   "seconds": round(time.time() - t0, 2)}).as_dict()
            fh.write(canonical_json({**rec, "record_hash": sha256(canonical_json(rec))}) + "\n")
            fh.flush(); os.fsync(fh.fileno())
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done, {(time.time() - t_all) / n:.1f}s per dialogue", flush=True)


def work_long(threads, device):
    from hse_dialogues import claim
    from agent_language_geometry.dialogue import generate_long_dialogue
    from agent_language_geometry.schemas import DialogueRecord
    config = load_config(ROOT / "configs" / "phase1_long_hse.yaml")
    rec_dir, locks = run_dir() / "long_records", run_dir() / "locks"
    todo = [u for u in long_units() if not (rec_dir / f"{skey(*u)}.json").exists()]
    print(f"long worker: {len(todo)} units left", flush=True)
    if not todo:
        return
    torch, model, tokenizer, PCABasis = _load(device, threads)
    dev = str(next(model.parameters()).device)
    gen_hash = sha256(canonical_json({"generation": config["generation"], "long_dialogue": config["long_dialogue"]}))
    job = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")}
    for L, card, seed in todo:
        path = rec_dir / f"{skey(L, card, seed)}.json"
        if path.exists():
            continue
        lock = claim(locks, path.stem, dev)
        if lock is None:
            continue
        if path.exists():
            lock.unlink(missing_ok=True)
            continue
        basis = PCABasis.load(basis_path(L))
        started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
        hook = LayerPCA(L, basis, RANK).install(model)
        try:
            utt, meta = generate_long_dialogue(model, tokenizer, card, "neutral", seed,
                                               config["generation"], config["long_dialogue"])
        finally:
            hook.remove()
        ok = bool(meta["target_reached"])
        rec = DialogueRecord(
            RUN_ID, "layer_sweep_long", card["stimulus_id"], sha256(canonical_json(card)), "neutral",
            f"L{L:02d}_pca_{RANK}", RANK, seed, utt, config["model"]["revision"], gen_hash, git_state(ROOT),
            software_manifest(), started, datetime.now(timezone.utc).isoformat(),
            "complete" if ok else "incomplete", None if ok else "TARGET_WORD_COUNT_NOT_REACHED",
            protocol_metadata={**meta, "layer_human": L, "layer_zero": L - 1, "hook_calls": hook.inner.calls,
                               "basis_sha256": file_sha256(basis_path(L)), "device": dev, "slurm": job,
                               "seconds": round(time.time() - t0, 1)}).as_dict()
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(canonical_json({**rec, "record_hash": sha256(canonical_json(rec))}) + "\n", encoding="utf-8")
        try:
            with path.open("x", encoding="utf-8") as fh:
                fh.write(tmp.read_text(encoding="utf-8"))
        except FileExistsError:
            pass
        tmp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
        print(f"saved L{L:02d} {card['stimulus_id']}: {meta['generated_words']} words, {meta['turns_generated']} turns, "
              f"{time.time() - t0:.0f}s", flush=True)


def status():
    s = short_done()
    longs = list((run_dir() / "long_records").glob("*.json"))
    per = {}
    for k in s:
        per[k[:3]] = per.get(k[:3], 0) + 1
    print(f"short {len(s)}/{len(short_units())} (per layer: {dict(sorted(per.items()))}); "
          f"long {len(longs)}/{len(long_units())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "prepare", "short", "long", "status"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit-cards", type=int, default=0)
    ap.add_argument("--basis-dir", default=None)
    a = ap.parse_args()
    if a.mode == "calibrate":
        calibrate(a.threads, a.limit_cards, a.basis_dir)
    elif a.mode == "prepare":
        prepare()
    elif a.mode == "short":
        work_short(a.worker, a.workers, a.threads, a.device)
    elif a.mode == "long":
        work_long(a.threads, a.device)
    else:
        status()


if __name__ == "__main__":
    main()
