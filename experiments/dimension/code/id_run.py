"""TwoNN, FisherS and Hidalgo on (1) sets of known dimension and (2) War and Peace embeddings.

Clouds (words n=1 and bigrams n=2, bigram = concatenation of word vectors):
  svd  d=5/10/15  (wap_embeddings.npz, word_u)        distinct rows only (duplicates break r1 > 0)
  cbow d=5/10/15  (wap_embeddings_cbow.npz)
  bert_pca d=5/10/15 and bert_full 768 (wap_embeddings_alt.npz)
TwoNN on all points (+ decimation), FisherS on <= 30 000 random points, Hidalgo on <= 5 000 random points.
Each result is written at once to <out>/<name>.json (restart-safe).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import id_methods as M
from validate_synthetic import embed, objects


def swiss_roll(n, rng):
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21 * rng.random(n)
    return np.c_[t * np.cos(t), h, t * np.sin(t)]


def mixture(n, rng):  # heterogeneous: half a 2-d plane, half a 6-d cube (Hidalgo's use case)
    A = np.zeros((n // 2, 6)); A[:, :2] = rng.random((n // 2, 2))
    B = rng.random((n - n // 2, 6)); B[:, 0] += 1.5
    return np.vstack([A, B])


def run_all(X, n_fisher, n_hidalgo, seed=0, h_iter=10000):
    rng = np.random.default_rng(seed)
    res = {"N": int(len(X)), "ambient": int(X.shape[1])}
    t = time.time(); res["twonn"] = M.twonn(X, seed=seed); res["t_twonn"] = time.time() - t
    Xf = X[rng.permutation(len(X))[:n_fisher]] if len(X) > n_fisher else X
    t = time.time(); f = M.fishers(Xf); res["fishers"] = f; res["t_fishers"] = time.time() - t
    Xh = X[rng.permutation(len(X))[:n_hidalgo]] if len(X) > n_hidalgo else X
    t = time.time(); res["hidalgo"] = M.hidalgo(Xh, K=2, seed=seed, Niter=h_iter); res["t_hidalgo"] = time.time() - t
    return res


def line(name, r, true=None):
    tn, fs, hg = r["twonn"]["d"], r["fishers"]["d"], r["hidalgo"]
    s = f"{name:28s} N={r['N']:>7d} D={r['ambient']:>4d} | TwoNN {tn:6.2f} | FisherS {fs if fs is None else round(fs, 2)}" \
        f" | Hidalgo d_k={[round(x, 2) for x in hg['d_k']]} p_k={[round(x, 2) for x in hg['p_k']]}"
    if true is not None:
        s = f"[true {true:.3f}] " + s
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="id_results")
    ap.add_argument("--synthetic-n", type=int, default=20000)
    ap.add_argument("--n-fisher", type=int, default=30000)
    ap.add_argument("--n-hidalgo", type=int, default=5000)
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--only-synthetic", action="store_true")
    ap.add_argument("--hidalgo-iter", type=int, default=10000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    print("numba:", M.HAVE_NUMBA, flush=True)

    if not a.skip_synthetic:
        rng = np.random.default_rng(20260930)
        objs = list(objects(a.synthetic_n, rng))
        objs.append(("swiss_roll", 2.0, swiss_roll(a.synthetic_n, rng)))
        objs.append(("mixture_2d_6d", float("nan"), mixture(a.synthetic_n, rng)))
        for name, true, p in objs:
            f = out / f"synthetic_{name}.json"
            if f.exists():
                continue
            X = embed(p, 15, rng)
            r = run_all(X, a.n_fisher, a.n_hidalgo, h_iter=a.hidalgo_iter); r["true_d"] = true
            f.write_text(json.dumps(r)); print(line(name, r, true), flush=True)

    if a.only_synthetic:
        return
    svd = np.load("wap_embeddings.npz"); cb = np.load("wap_embeddings_cbow.npz"); bt = np.load("wap_embeddings_alt.npz")
    big = svd["bigrams"]
    sources = []
    for d in (5, 10, 15):
        sources += [("svd", d, svd["word_u"][:, :d]), ("cbow", d, cb[f"cbow{d}"]), ("bert_pca", d, bt["bert_pca15"][:, :d])]
    sources.append(("bert_full", 768, bt["bert768"].astype(np.float64)))
    for n in (1, 2):
        for src, d, U in sources:
            name = f"{src}_n{n}_d{d}"
            f = out / f"{name}.json"
            if f.exists():
                continue
            U = np.asarray(U, dtype=np.float64)
            X = U if n == 1 else np.hstack([U[big[:, 0]], U[big[:, 1]]])
            X = np.unique(X, axis=0)  # distinct points (only SVD has duplicates)
            t = time.time()
            r = run_all(X, a.n_fisher, a.n_hidalgo, h_iter=a.hidalgo_iter); r["source"] = src; r["ngram"] = n; r["d"] = d
            r["seconds"] = time.time() - t
            f.write_text(json.dumps(r)); print(line(name, r) + f"  ({r['seconds']:.0f}s)", flush=True)
    print("=== all done", flush=True)


if __name__ == "__main__":
    main()
