"""Long Phase-1 dialogues (3 frames x 24 cards x 3 seeds = 216) on HSE cHARISMa, in parallel.

Same protocol and record format as `cli.run_long_phase1` (generate_long_dialogue, DialogueRecord,
record_hash, file names <frame>__<stimulus>__<seed>.json, final dialogues.jsonl), but the 216 units
are split over independent Slurm workers:

  python hse_dialogues.py prepare  --config configs/phase1_long_hse.yaml
  python hse_dialogues.py work     --config ... --worker K --workers M   (units with (index-1) % M == K)
  python hse_dialogues.py status   --config ...
  python hse_dialogues.py finalize --config ...                           (checks all 216, writes dialogues.jsonl)

A worker skips units whose record already exists, and records are created with open("x"), so any
worker can be rerun / requeued safely. Only the light modules are imported (no sklearn/scipy needed).
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

from agent_language_geometry.config import load_config  # noqa: E402
from agent_language_geometry.reproducibility import (  # noqa: E402
    canonical_json, git_state, sha256, software_manifest, write_manifest,
)


def cards_of(config):
    from agent_language_geometry.dialogue import FORBIDDEN_FRAME_MARKERS
    cards = [json.loads(x) for x in (ROOT / config["paths"]["evaluation_stimuli"]).read_text("utf-8").splitlines() if x]
    # same checks as cli._validate_stimuli
    if len(cards) != 24 or len({c["stimulus_id"] for c in cards}) != 24:
        raise ValueError("A stimulus set must contain exactly 24 unique cards")
    strata = [c["stratum"] for c in cards]
    if any(strata.count(s) != 4 for s in set(strata)) or len(set(strata)) != 6:
        raise ValueError("Stimuli require six strata with four cards each")
    for c in cards:
        for frame, markers in FORBIDDEN_FRAME_MARKERS.items():
            rendered = "".join(c.get(frame, {}).values()) if isinstance(c.get(frame), dict) else ""
            if any(m in rendered.lower() for m in markers):
                raise ValueError(f"Frame marker leakage in {c['stimulus_id']} {frame}")
    return cards


def units_of(config, cards):
    all_units = [(f, c, s) for f in config["frames"] for c in cards for s in config["generation"]["seeds"]]
    return [(i, f, c, s) for i, (f, c, s) in enumerate(all_units, start=1)]


def run_dir_of(config):
    return ROOT / config["paths"]["results_root"] / config["run_id"]


def record_path(records_dir, frame, stimulus_id, seed):
    return records_dir / f"{frame}__{stimulus_id}__{seed}.json"


def read_record(path, run_id, config_hash):
    record = json.loads(path.read_text(encoding="utf-8"))
    h = record.pop("record_hash", None)
    if h != sha256(canonical_json(record)):
        raise RuntimeError(f"Invalid long-dialogue record hash: {path}")
    if record.get("run_id") != run_id:
        raise RuntimeError(f"Different run ID: {path}")
    if record.get("protocol_metadata", {}).get("long_config_hash") != config_hash:
        raise RuntimeError(f"Different configuration: {path}")
    return record


def check_run(config):
    rd = run_dir_of(config)
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    if man.get("config_hash") != config["_config_hash"]:
        raise RuntimeError("configuration hash does not match the prepared run")
    return rd


def prepare(config):
    if not config.get("run_id"):
        raise ValueError("set run_id in the config")
    cards = cards_of(config)
    rd = run_dir_of(config)
    if rd.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run directory: {rd}")
    rd.mkdir(parents=True)
    write_manifest(rd, {"run_id": config["run_id"], "config": config, "config_hash": config["_config_hash"],
                        "git": git_state(ROOT),
                        "execution": "HSE cHARISMa, parallel Slurm workers (hse_dialogues.py)"})
    (rd / "dialogue_records").mkdir()
    scope = {"frames": config["frames"], "stimulus_ids": [c["stimulus_id"] for c in cards],
             "seeds": config["generation"]["seeds"]}
    (rd / "long_dialogue_scope.json").write_text(canonical_json(scope) + "\n", encoding="utf-8")
    print(f"prepared {rd} with {len(units_of(config, cards))} units", flush=True)


STALE_SECONDS = {"cpu": 14 * 3600, "cuda": 3600}  # a claim older than this is considered abandoned


def claim(locks_dir, name, dev):
    """Atomically reserve a unit so CPU and GPU workers never generate the same dialogue twice."""
    lock = locks_dir / (name + ".lock")
    if lock.exists():
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            age = time.time() - float(info["time"])
            limit = STALE_SECONDS.get(info.get("device", "cpu").split(":")[0], STALE_SECONDS["cpu"])
        except Exception:
            age, limit = 0.0, STALE_SECONDS["cpu"]
        if age < limit:
            return None
        lock.unlink(missing_ok=True)  # stale claim (job died / was preempted)
    try:
        with lock.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": time.time(), "device": dev, "job": os.environ.get("SLURM_JOB_ID"),
                                 "host": os.uname().nodename, "pid": os.getpid()}))
    except FileExistsError:
        return None
    return lock


def work(config, worker, workers, threads, device, order="asc", use_claims=False):
    import torch
    from agent_language_geometry.dialogue import generate_long_dialogue
    from agent_language_geometry.model_runtime import load_runtime
    from agent_language_geometry.schemas import DialogueRecord

    if threads:
        torch.set_num_threads(threads)
    rd = check_run(config)
    records_dir = rd / "dialogue_records"
    cards = cards_of(config)
    mine = [u for u in units_of(config, cards) if (u[0] - 1) % workers == worker]
    todo = [u for u in mine if not record_path(records_dir, u[1], u[2]["stimulus_id"], u[3]).exists()]
    if order == "desc":
        todo = todo[::-1]
    locks_dir = rd / "locks"
    locks_dir.mkdir(exist_ok=True)
    print(f"worker {worker}/{workers}: {len(mine)} units, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, tokenizer = load_runtime(config["model"], device=device)
    dev = str(next(model.parameters()).device)
    gen_hash = sha256(canonical_json({"generation": config["generation"], "long_dialogue": config["long_dialogue"]}))
    job = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")}
    # pass 2 (GPU only): when nothing unclaimed is left, also take units still being ground by CPU workers;
    # whoever finishes first writes the record (open "x"), the other result is discarded.
    passes = [(u, 1) for u in todo] + ([(u, 2) for u in todo] if use_claims and dev.startswith("cuda") else [])
    for (index, frame, card, seed), pass_no in passes:
        path = record_path(records_dir, frame, card["stimulus_id"], seed)
        if path.exists():
            continue
        lock = None
        if use_claims and pass_no == 1:
            lock = claim(locks_dir, path.stem, dev)
            if lock is None:
                continue
        elif use_claims and pass_no == 2:
            other = locks_dir / (path.stem + ".lock")
            try:
                if not json.loads(other.read_text(encoding="utf-8")).get("device", "").startswith("cpu"):
                    continue  # another GPU process is on it
            except Exception:
                pass
        if path.exists():  # finished by someone else while we were claiming
            if lock is not None:
                lock.unlink(missing_ok=True)
            continue
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        utterances, meta = generate_long_dialogue(model, tokenizer, card, frame, seed,
                                                  config["generation"], config["long_dialogue"])
        ok = bool(meta["target_reached"])
        record = DialogueRecord(
            rd.name, "phase1_long", card["stimulus_id"], sha256(canonical_json(card)), frame, "identity", None,
            seed, utterances, config["model"]["revision"], gen_hash, git_state(ROOT), software_manifest(),
            started, datetime.now(timezone.utc).isoformat(), "complete" if ok else "incomplete",
            None if ok else "TARGET_WORD_COUNT_NOT_REACHED",
            protocol_metadata={**meta, "long_config_hash": config["_config_hash"], "global_unit_index": index,
                               "device": dev, "torch_threads": torch.get_num_threads(), "slurm": job,
                               "seconds": round(time.time() - t0, 1)},
        ).as_dict()
        full = {**record, "record_hash": sha256(canonical_json(record))}
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(canonical_json(full) + "\n", encoding="utf-8")
        try:
            with path.open("x", encoding="utf-8") as fh:  # never overwrite an existing record
                fh.write(tmp.read_text(encoding="utf-8"))
        except FileExistsError:
            pass
        tmp.unlink(missing_ok=True)
        if lock is not None:
            lock.unlink(missing_ok=True)
        print(f"saved unit {index} {frame} {card['stimulus_id']} seed {seed}: {meta['generated_words']} words, "
              f"{meta['turns_generated']} turns, {time.time() - t0:.0f}s on {dev}", flush=True)


def status(config):
    rd = check_run(config)
    cards = cards_of(config)
    units = units_of(config, cards)
    done = [u for u in units if record_path(rd / "dialogue_records", u[1], u[2]["stimulus_id"], u[3]).exists()]
    per = {}
    for u in done:
        per[u[1]] = per.get(u[1], 0) + 1
    print(f"{len(done)}/{len(units)} records; by frame {per}", flush=True)
    return len(done) == len(units)


def finalize(config):
    rd = check_run(config)
    cards = cards_of(config)
    paths = [record_path(rd / "dialogue_records", f, c["stimulus_id"], s) for _, f, c, s in units_of(config, cards)]
    names = {p.name for p in (rd / "dialogue_records").glob("*.json")}
    if names != {p.name for p in paths}:
        raise RuntimeError(f"record set incomplete or out of scope: {len(names)} files")
    rows = [read_record(p, rd.name, config["_config_hash"]) for p in paths]
    out = rd / "dialogues.jsonl"
    if out.exists():
        raise FileExistsError(out)
    out.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    inc = sum(r["status"] != "complete" for r in rows)
    words = [r["protocol_metadata"]["generated_words"] for r in rows]
    print(f"wrote {out}: {len(rows)} dialogues, incomplete {inc}, words min {min(words)} max {max(words)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "work", "status", "finalize"])
    ap.add_argument("--config", default="configs/phase1_long_hse.yaml")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--order", default="asc", choices=["asc", "desc"], help="desc: start from the last unit")
    ap.add_argument("--claims", action="store_true", help="reserve units with lock files (CPU and GPU share the run)")
    a = ap.parse_args()
    config = load_config(ROOT / a.config)
    if a.mode == "prepare":
        prepare(config)
    elif a.mode == "work":
        work(config, a.worker, a.workers, a.threads, a.device, a.order, a.claims)
    elif a.mode == "status":
        status(config)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
