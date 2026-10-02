"""Many short (10-turn, MVP protocol) dialogues per interaction frame, as a corpus for frame-specific CBOW.

Design (exploratory, see DECISIONS.md 2026-10-02):
  frames x 24 evaluation cards x S generation seeds (seeds 1..S), generated with the unchanged MVP
  `generate_dialogue` (10 turns, max_new_tokens 48, same sampling) -> ~300 generated words per dialogue.
  Units are ordered seed-major (seed, card, frame), so any prefix of the run is balanced over frames and cards.

Storage: each worker appends finished dialogues to its own shard results/<run_id>/shards/w<k>.jsonl
(one canonical-JSON record with record_hash per line, flushed + fsynced). A unit is done if its key
<frame>__<stimulus>__<seed> occurs in any shard, so workers can be killed / requeued / rerun at any time.

  python hse_short_dialogues.py prepare --seeds 150
  python hse_short_dialogues.py work --worker K --workers M [--threads 2 --device cpu]
  python hse_short_dialogues.py status
  python hse_short_dialogues.py export      # one corpus file per frame: corpus_<frame>.jsonl (+ summary)
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
    import types
    sys.modules["soundfile"] = None  # find_spec() -> None: transformers treats soundfile as absent

from agent_language_geometry.config import load_config  # noqa: E402
from agent_language_geometry.reproducibility import (  # noqa: E402
    canonical_json, git_state, sha256, software_manifest, write_manifest,
)

RUN_ID = "short-10turn-cbow-v1"
CONFIG = "configs/base.yaml"
FRAMES = ["competition", "cooperation", "neutral"]


def cards():
    from hse_dialogues import cards_of  # same 24 cards and the same validity checks as the long run
    return cards_of(load_config(ROOT / CONFIG))


def run_dir():
    return ROOT / "results" / RUN_ID


def plan():
    return json.loads((run_dir() / "plan.json").read_text(encoding="utf-8"))


def units(p, cs):
    by_id = {c["stimulus_id"]: c for c in cs}
    out = []
    for seed in p["seeds"]:
        for sid in p["stimulus_ids"]:
            for frame in p["frames"]:
                out.append((frame, by_id[sid], seed))
    return out


def key(frame, card, seed):
    return f"{frame}__{card['stimulus_id']}__{seed}"


def done_keys():
    keys = set()
    for shard in (run_dir() / "shards").glob("*.jsonl"):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a line cut by a kill: ignored, unit will be redone
                    continue
                keys.add(f"{r['interaction_frame']}__{r['stimulus_id']}__{r['generation_seed']}")
    return keys


def prepare(n_seeds):
    rd = run_dir()
    if rd.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run directory: {rd}")
    config = load_config(ROOT / CONFIG)
    cs = cards()
    rd.mkdir(parents=True)
    (rd / "shards").mkdir()
    p = {"run_id": RUN_ID, "frames": FRAMES, "stimulus_ids": [c["stimulus_id"] for c in cs],
         "seeds": list(range(1, n_seeds + 1)), "turns": config["generation"]["turns"],
         "order": "seed-major (seed, card, frame)"}
    (rd / "plan.json").write_text(canonical_json(p) + "\n", encoding="utf-8")
    write_manifest(rd, {"run_id": RUN_ID, "config": config, "config_hash": config["_config_hash"],
                        "plan": p, "git": git_state(ROOT),
                        "execution": "HSE cHARISMa, Slurm workers (hse_short_dialogues.py)"})
    print(f"prepared {rd}: {len(units(p, cs))} dialogues", flush=True)


def work(worker, workers, threads, device):
    import torch
    from agent_language_geometry.dialogue import generate_dialogue
    from agent_language_geometry.model_runtime import load_runtime
    from agent_language_geometry.schemas import DialogueRecord

    if threads:
        torch.set_num_threads(threads)
    config = load_config(ROOT / CONFIG)
    man = json.loads((run_dir() / "manifest.json").read_text(encoding="utf-8"))
    if man["config_hash"] != config["_config_hash"]:
        raise RuntimeError("configuration hash differs from the prepared run")
    p, cs = plan(), cards()
    mine = [u for i, u in enumerate(units(p, cs)) if i % workers == worker]
    done = done_keys()
    todo = [u for u in mine if key(*u) not in done]
    print(f"worker {worker}/{workers}: {len(mine)} units, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, tokenizer = load_runtime(config["model"], device=device)
    dev = str(next(model.parameters()).device)
    gen_hash = sha256(canonical_json({"generation": config["generation"]}))
    job = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")}
    shard = run_dir() / "shards" / f"w{worker:04d}.jsonl"
    t_all = time.time()
    with shard.open("a", encoding="utf-8") as fh:
        for n, (frame, card, seed) in enumerate(todo, 1):
            started = datetime.now(timezone.utc).isoformat()
            t0 = time.time()
            utterances = generate_dialogue(model, tokenizer, card, frame, seed, config["generation"])
            record = DialogueRecord(
                RUN_ID, "short_cbow", card["stimulus_id"], sha256(canonical_json(card)), frame, "identity", None,
                seed, utterances, config["model"]["revision"], gen_hash, git_state(ROOT), software_manifest(),
                started, datetime.now(timezone.utc).isoformat(), "complete", None,
                protocol_metadata={"device": dev, "torch_threads": torch.get_num_threads(), "slurm": job,
                                   "seconds": round(time.time() - t0, 2), "config_hash": config["_config_hash"]},
            ).as_dict()
            fh.write(canonical_json({**record, "record_hash": sha256(canonical_json(record))}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t_all
                print(f"{n}/{len(todo)} done, {el / n:.1f}s per dialogue", flush=True)


def status():
    p, cs = plan(), cards()
    us = units(p, cs)
    done = done_keys()
    per = {f: sum(1 for u in us if u[0] == f and key(*u) in done) for f in p["frames"]}
    print(f"{len(done & {key(*u) for u in us})}/{len(us)} dialogues; by frame {per}", flush=True)


def export():
    import re
    p, cs = plan(), cards()
    wanted = {key(*u) for u in units(p, cs)}
    best = {}
    for shard in sorted((run_dir() / "shards").glob("*.jsonl")):
        for line in shard.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = r.pop("record_hash", None)
            if h != sha256(canonical_json(r)):
                continue
            k = f"{r['interaction_frame']}__{r['stimulus_id']}__{r['generation_seed']}"
            if k in wanted and k not in best:  # first valid copy wins (duplicates only after a requeue)
                best[k] = r
    summary = {}
    for frame in p["frames"]:
        rows = [best[k] for k in sorted(best) if k.startswith(frame + "__")]
        out = run_dir() / f"corpus_{frame}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"stimulus_id": r["stimulus_id"], "seed": r["generation_seed"],
                                     "utterances": [u["text"] for u in r["utterances"]]}, ensure_ascii=False) + "\n")
        words = sum(len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", u["text"])) for r in rows for u in r["utterances"])
        summary[frame] = {"dialogues": len(rows), "generated_words": words}
    (run_dir() / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "work", "status", "export"])
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.seeds)
    elif a.mode == "work":
        work(a.worker, a.workers, a.threads, a.device)
    elif a.mode == "status":
        status()
    else:
        export()


if __name__ == "__main__":
    main()
