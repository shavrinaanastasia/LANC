"""Intrinsic dimension of frame-specific CBOW spaces (short-dialogue corpus), same estimators and settings as
the pseudolanguage / trajectory calibration (calib_trajectories.measure: Schweinhart over 2 ladders, TwoNN,
FisherS, Hidalgo; at most 10 000 points per cloud).

  python dialogue_dims.py --src ~/lanc_run/results/short-10turn-cbow-v1/cbow --out dialogue_dims_results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from calib_trajectories import measure

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="dialogue_dims_results")
ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 15, 20, 30])
ap.add_argument("--ladders", type=int, default=2)
ap.add_argument("--max-points", type=int, default=10000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
for f in sorted(Path(a.src).expanduser().glob("cbow_*_mc*.npz")):
    tag = f.stem  # cbow_<frame>_mc<k>
    Z = np.load(f)
    for d in a.dims:
        if f"cbow{d}" not in Z:
            continue
        U = Z[f"cbow{d}"]
        for n in (1, 2):
            target = out / f"{tag}_d{d}_n{n}.json"
            if target.exists():
                continue
            X = U if n == 1 else np.hstack([U[Z["bigrams"][:, 0]], U[Z["bigrams"][:, 1]]])
            t0 = time.time()
            r = measure(X, a.ladders, a.max_points, 5000)
            r.update({"source": tag, "d": d, "ngram": n, "vocab": int(len(U)), "seconds": time.time() - t0})
            target.write_text(json.dumps(r))
            sw = r["schweinhart"]
            print(f"{tag} d={d} n={n} N={r['N']} | Schw {sw['min']}-{sw['max']} | TwoNN {r['twonn']:.2f} | "
                  f"FisherS {r['fishers']} | Hidalgo {np.round(r['hidalgo']['d_k'], 2).tolist()} ({r['seconds']:.0f}s)", flush=True)
