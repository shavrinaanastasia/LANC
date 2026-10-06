"""E13: is the bigram estimate / m a universal function of rho = delta / r ?

delta = typical distance between the two halves of a bigram (|x_{t+1} - x_t|), r = typical nearest-neighbour distance in the
bigram cloud. Hypothesis: for Brownian walks estimate(bigrams) / m = f(rho), rising from ~1 (rho << 1: pairs hug the
diagonal) to ~2 (rho >> 1: the second half looks independent), the same curve for every m and object.

Part 1 (tasks 0..8): exact coordinates only (no CBOW). Objects T2..T12, S4, S8, S12; 500 walks x 1500 steps; step multiplier
0.1 .. 5 (step 1 = calib3 baseline, ~ one cell of V = 5000); two kinds of bigram cloud:
  cont  -- consecutive raw walk points (x_t, x_{t+1});
  cells -- V = 5000 random cell centres, text as in calib2/3 (repeats of a cell merged), unique bigrams of centres.
For N = 1k .. 100k sub-samples: TwoNN, rho (median and mean), words-cloud TwoNN at the same N (should be ~ m), correlation
slopes at three scales, Hidalgo at N = 3000.
Part 4 (task 9): War and Peace CBOW d = 15 (and d = 5, 10): the same quantities for words and all 152 568 unique bigrams at
N = 1k .. all, to place the text on the curve and to test whether the growth 6.6 -> 9.3 of its bigram TwoNN with N follows rho.

  python calib4.py --list ; python calib4.py --task i
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

import calib_trajectories as CT
import id_methods as M

OUT = Path("calib4_results")
OBJ = ["torus2", "torus4", "torus6", "torus8", "torus10", "torus12", "sphere4", "sphere8", "sphere12"]
STEPS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
NS = [1000, 3000, 10000, 30000, 100000]
V = 5000


def corr_slopes(X, rng, n=3000):
    from scipy.spatial.distance import pdist
    S = X[rng.permutation(len(X))[: min(n, len(X))]]
    d = np.sort(pdist(S)); d = d[d > 0]
    out = {}
    for lo, hi in ((1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1)):
        r0, r1 = d[int(lo * len(d))], d[int(hi * len(d))]
        out[f"{lo:g}-{hi:g}"] = {"slope": float(np.log(hi / lo) / np.log(r1 / r0)) if r1 > r0 else None, "r_lo": float(r0), "r_hi": float(r1)}
    return out


def cloud_stats(P, half, N, rng, hid=False):
    """P: bigram cloud (n, 2*half). Sub-sample N rows, return TwoNN, rho and friends."""
    idx = rng.permutation(len(P))[:N]
    X = P[idx]
    dist, _ = M.knn(X, 2)
    ok = dist[:, 0] > 0
    tw = M.twonn_from_mu(dist[ok, 1] / dist[ok, 0])
    r_med, r_mean = float(np.median(dist[ok, 0])), float(dist[ok, 0].mean())
    dl = np.linalg.norm(X[:, half:] - X[:, :half], axis=1)
    res = {"N": int(len(X)), "twonn": tw, "r_med": r_med, "r_mean": r_mean, "delta_med": float(np.median(dl)),
           "delta_mean": float(dl.mean()), "rho_med": float(np.median(dl) / r_med), "rho_mean": float(dl.mean() / r_mean)}
    if hid:
        try:
            h = M.hidalgo(np.unique(X, axis=0), K=2, seed=0, Niter=3000)
            res["hidalgo"] = {"d_k": h["d_k"], "p_k": h["p_k"]}
        except Exception as e:  # noqa: BLE001
            res["hidalgo"] = {"error": str(e)}
    return res


def words_twonn(W, N, rng):
    X = np.unique(W, axis=0)
    X = X[rng.permutation(len(X))[:N]]
    return M.twonn(X, decimation=(1,))["d"]


def run_object(obj, n_traj=500, length=1500):
    m = int(obj.replace("torus", "").replace("sphere", ""))
    for st in STEPS:
        fn = OUT / f"{obj}_step{st:g}.json"
        if fn.exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng(sum(map(ord, obj)) + int(st * 100))
        sigma = st * (np.pi if obj.startswith("torus") else 1.0) * V ** (-1 / m)
        traj = CT.trajectories(obj, n_traj, length, rng, sigma)
        D = traj.shape[-1]
        pts = traj.reshape(-1, D)
        res = {"object": obj, "m": m, "step": st, "sigma": sigma, "ambient": D, "cont": [], "cells": []}
        cont = np.concatenate([traj[:, :-1], traj[:, 1:]], axis=2).reshape(-1, 2 * D)
        centres = pts[np.random.default_rng(1).choice(len(pts), V, replace=False)]
        docs = CT.to_texts(traj, centres)
        big = CT.bigrams_of(docs)
        cells = np.hstack([centres[big[:, 0]], centres[big[:, 1]]])
        res["n_cells_bigrams"] = int(len(cells))
        for kind, P, W in (("cont", cont, pts), ("cells", cells, centres)):
            r2 = np.random.default_rng(0)
            for N in NS:
                if N > len(P) or (N > 30000 and 2 * D > 16):  # brute-force kNN above 8 dims per half: cap at 30k
                    continue
                s = cloud_stats(P, D, N, r2, hid=(N == 3000))
                s["words_twonn"] = words_twonn(W, min(N, len(W)), r2)
                if N == 10000:
                    s["corr"] = corr_slopes(P, r2)
                res[kind].append(s)
                print(obj, st, kind, N, f"TwoNN {s['twonn']:.2f} (/m {s['twonn'] / m:.2f}) rho {s['rho_med']:.2f} words {s['words_twonn']:.2f}", flush=True)
        res["seconds"] = time.time() - t0
        fn.write_text(json.dumps(res))


def run_wap():
    z = np.load("wap_embeddings_cbow.npz", allow_pickle=True)
    B = z["bigrams"]
    for d in (15, 10, 5):
        fn = OUT / f"wap_cbow{d}.json"
        if fn.exists():
            continue
        W = z[f"cbow{d}"]
        P = np.hstack([W[B[:, 0]], W[B[:, 1]]])
        res = {"object": "war_and_peace", "d": d, "n_bigrams": int(len(P)), "bigrams": [], "words": []}
        r2 = np.random.default_rng(0)
        for N in NS[:-1] + [100000, len(P)]:
            if N > len(P) or any(x["N"] == N for x in res["bigrams"]):
                continue
            s = cloud_stats(P, d, N, r2, hid=(N == 3000))
            if N == 10000:
                s["corr"] = corr_slopes(P, r2)
            res["bigrams"].append(s)
            print("wap", d, "bigrams", N, f"TwoNN {s['twonn']:.2f} rho {s['rho_med']:.2f}", flush=True)
        for N in (1000, 3000, 10000, len(W)):
            res["words"].append({"N": int(N), "twonn": words_twonn(W, N, r2)})
        fn.write_text(json.dumps(res))


def tasks():
    return OBJ + ["wap"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--task", type=int)
    a = ap.parse_args(); T = tasks()
    if a.list:
        for i, t in enumerate(T): print(i, t)
        return
    OUT.mkdir(exist_ok=True)
    t = T[a.task]
    run_wap() if t == "wap" else run_object(t)


if __name__ == "__main__":
    main()
