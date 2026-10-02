"""Schweinhart intrinsic dimension of War and Peace word / bigram clouds (Gromov et al. 2024 Table 4 protocol).

For n in {1 (words), 2 (bigrams)} and d in {5, 10, 15}:
  cloud  = rows U[:, :d]  (words)  or  [U[a, :d], U[b, :d]]  (bigrams, concatenation)
  ladder = nested random subsets, geometric sizes from N/16 to N (paper: 1e5..N, impossible here)
  MST    = exact Euclidean (Prim), one tree per size, reused for every alpha
  alpha  = 1e-4, 0.1, ..., 10
  keep   = alphas whose 95% CI for d and for (d-alpha)/d are both <= 10% wide
  report = min / max admissible d  (the paper's min d_Schw / max d_Schw)
Two variants: "all" = every unique n-gram is a point (paper), "distinct" = duplicate coordinates
collapsed (hapax words of one chapter share identical SVD rows).

Each finished configuration is written immediately to <out>/results/<config>.json (restart-safe).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from schweinhart_mst import DEFAULT_ALPHAS, geometric_ladder, schweinhart, summarise, threads_info

PAPER_TABLE5_RU_CBOW = {  # Complexity 2024, Table 5 (CBOW), Russian
    (1, 5): (4.45, 4.62), (1, 10): (5.04, 7.18), (1, 15): (5.98, 9.57),
    (2, 5): (3.82, 6.79), (2, 10): (4.85, 9.43), (2, 15): (5.73, 10.51),
}
PAPER_TABLE4_RU_SVD = {  # Complexity 2024, Table 4 (SVD), Russian
    (1, 5): (4.65, 5.52), (1, 10): (7.14, 8.20), (1, 15): (9.81, 12.78),
    (2, 5): (5.11, 7.79), (2, 10): (6.48, 8.52), (2, 15): (8.26, 10.70),
}


def word_vectors(data, source: str, d: int) -> np.ndarray:
    if source == "svd":
        return data["word_u"][:, :d]
    if source == "cbow":
        return data[f"cbow{d}"]
    if source == "bert_pca":
        return data["bert_pca15"][:, :d]
    if source == "bert_full":
        return data["bert768"]
    raise ValueError(source)


def paper_ref(source: str, n: int, d: int):
    table = PAPER_TABLE5_RU_CBOW if source == "cbow" else PAPER_TABLE4_RU_SVD
    return table.get((n, d))


def cloud(data, n: int, d: int, source: str = "svd") -> np.ndarray:
    u = np.asarray(word_vectors(data, source, d), dtype=np.float64)
    if n == 1:
        return u
    g = data["bigrams"] if n == 2 else data["trigrams"]  # concatenation of constituents (paper)
    return np.hstack([u[g[:, i]] for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="wap_embeddings.npz")
    ap.add_argument("--out", default="wap_results")
    ap.add_argument("--ngrams", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--variants", nargs="+", default=["all", "distinct"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--min-frac", type=float, default=1 / 16)
    ap.add_argument("--max-points", type=int, default=0, help="pilot: subsample cloud first (0 = all)")
    ap.add_argument("--source", default="svd", choices=["svd", "cbow", "bert_pca", "bert_full"])
    ap.add_argument("--backend", default="auto", choices=["auto", "numba", "numpy", "torch"])
    a = ap.parse_args()

    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    data = np.load(a.emb)
    info = {"host": platform.node(), "backend": threads_info(), "args": vars(a)}
    print(json.dumps(info), flush=True)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))

    for n in a.ngrams:
        for d in a.dims:
            base = cloud(data, n, d, a.source)
            if a.source == "bert_full":
                d = base.shape[1] // n
            for variant in a.variants:
                name = f"{a.source}_n{n}_d{d}_{variant}"
                target = out / "results" / f"{name}.json"
                if target.exists():
                    print(f"skip {name} (done)", flush=True)
                    continue
                x = np.unique(base, axis=0) if variant == "distinct" else base
                if a.max_points and x.shape[0] > a.max_points:
                    x = x[np.random.default_rng(0).permutation(x.shape[0])[: a.max_points]]
                N = x.shape[0]
                sizes = geometric_ladder(N, int(N * a.min_frac), a.steps)
                print(f"== {name}: N={N} ambient={x.shape[1]} sizes={sizes}", flush=True)
                t0 = time.time()
                runs = []
                for seed in a.seeds:
                    r = schweinhart(x, sizes, DEFAULT_ALPHAS, seed=seed, backend=a.backend,
                                    log=lambda s: print(s, flush=True))
                    r["summary"] = summarise(r["fits"])
                    print(f"  seed {seed}: {r['summary']}", flush=True)
                    runs.append(r)
                mins = [r["summary"]["min_d"] for r in runs if r["summary"]["min_d"] is not None]
                maxs = [r["summary"]["max_d"] for r in runs if r["summary"]["max_d"] is not None]
                res = {
                    "config": name, "source": a.source, "ngram": n, "d": d, "variant": variant, "N": N,
                    "ambient_dim": int(x.shape[1]), "sizes": sizes,
                    "paper_ru_svd_min_max": paper_ref(a.source, n, d),
                    "min_d_mean": float(np.mean(mins)) if mins else None,
                    "max_d_mean": float(np.mean(maxs)) if maxs else None,
                    "min_d_over_seeds": [float(v) for v in mins],
                    "max_d_over_seeds": [float(v) for v in maxs],
                    "seconds": time.time() - t0,
                    "runs": runs,
                }
                target.write_text(json.dumps(res))
                print(f"== {name} done in {res['seconds']:.0f}s: min {res['min_d_mean']} max {res['max_d_mean']}"
                      f" | paper {res['paper_ru_svd_min_max']}", flush=True)

    rows = []
    for f in sorted((out / "results").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({k: r[k] for k in ("config", "ngram", "d", "variant", "N", "min_d_mean", "max_d_mean")}
                    | {"paper_min": (r["paper_ru_svd_min_max"] or [None, None])[0],
                       "paper_max": (r["paper_ru_svd_min_max"] or [None, None])[1]})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(open(out / "summary.csv", encoding="utf-8").read(), flush=True)


if __name__ == "__main__":
    main()
