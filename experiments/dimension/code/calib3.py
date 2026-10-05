"""E12: how stable is the CBOW over-estimation factor inside the Brownian group?

Brownian objects (torus2, torus4, torus6, sphere4, sphere8), baseline as in calib2 (V = 5000, 500 trajectories x 1500
steps, step ~ one cell, sentences of 20 words, CBOW window 5, uniform cells). One factor changed at a time:
  step   -- Brownian step x 0.5 / 2 / 3
  zipf   -- non-uniform cells: centres drawn with density ~ exp(kappa * g(x)), g a smooth random field, kappa = 1.5 / 3 / 5
            (small cells in dense regions -> skewed, Zipf-like word frequencies; the achieved skew is recorded)
  vocab  -- V = 18000 (War and Peace size)
  sent   -- sentence length 10 / 40
  window -- CBOW window 2 / 10
Embeddings: truth, CBOW d = 15 and 30; words and bigrams; the four estimators (bigram clouds capped at 10 000 points as in
calib2) plus TwoNN on ALL unique bigrams (to compare with War and Peace, which has ~150 000).

  python calib3.py --list ; python calib3.py --task i
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

import calib_trajectories as CT
import id_methods as M

OUT = Path("calib3_results")
OBJ = ["torus2", "torus4", "torus6", "sphere4", "sphere8"]
BASE = dict(step=1.0, kappa=0.0, vocab=5000, sent=20, window=5)
GROUPS = {"base": [{}], "step": [{"step": 0.5}, {"step": 2.0}, {"step": 3.0}], "zipf": [{"kappa": 1.5}, {"kappa": 3.0}, {"kappa": 5.0}],
          "vocab": [{"vocab": 18000}], "sent": [{"sent": 10}, {"sent": 40}], "window": [{"window": 2}, {"window": 10}]}


def tasks():
    return [(o, g) for o in OBJ for g in GROUPS]


def field(X, rng, n_waves=6):
    """Smooth random field on the ambient coordinates (sum of random cosines), standardised."""
    W = rng.normal(0, 2.0, (n_waves, X.shape[1])); ph = rng.uniform(0, 2 * np.pi, n_waves)
    g = np.cos(X @ W.T + ph).sum(1)
    return (g - g.mean()) / g.std()


def partition_weighted(X, V, kappa, rng):
    if kappa == 0:
        return X[rng.choice(len(X), V, replace=False)]
    w = np.exp(kappa * field(X, np.random.default_rng(123)))
    return X[rng.choice(len(X), V, replace=False, p=w / w.sum())]


def cbow(docs, d, window):
    from gensim.models import Word2Vec
    sents = [[str(w) for w in s] for doc in docs for s in doc]
    m = Word2Vec(sents, vector_size=d, sg=0, window=window, negative=5, min_count=1, epochs=30, seed=1, workers=1)
    V = max(int(w) for w in m.wv.index_to_key) + 1
    out = np.full((V, d), np.nan)
    for w in m.wv.index_to_key:
        out[int(w)] = m.wv[w]
    return out


def run(obj, cfg, n_traj=500, length=1500):
    c = dict(BASE, **cfg)
    tag = f"{obj}_" + "_".join(f"{k}{c[k]}" for k in ("step", "kappa", "vocab", "sent", "window"))
    if (OUT / f"{tag}_done.json").exists():
        return
    m = int(obj.replace("torus", "").replace("sphere", ""))
    rng = np.random.default_rng(sum(map(ord, obj)))
    sigma = c["step"] * (np.pi if obj.startswith("torus") else 1.0) * c["vocab"] ** (-1 / m)
    traj = CT.trajectories(obj, n_traj, length, rng, sigma)
    X = traj.reshape(-1, traj.shape[-1])
    centres = partition_weighted(X, c["vocab"], c["kappa"], np.random.default_rng(1))
    docs = CT.to_texts(traj, centres, sent_len=c["sent"])
    big = CT.bigrams_of(docs)
    words = np.concatenate([np.asarray(s) for d in docs for s in d])
    used, cnt = np.unique(words, return_counts=True)
    f = np.sort(cnt)[::-1]; r = np.arange(1, len(f) + 1)
    sl = np.polyfit(np.log(r[: max(10, len(f) // 2)]), np.log(f[: max(10, len(f) // 2)]), 1)[0]
    meta = {"object": obj, "true_dim": m, **c, "sigma": sigma, "used_words": int(len(used)), "tokens": int(len(words)),
            "bigrams": int(len(big)), "zipf_slope": float(sl), "freq_top1pct_share": float(f[: max(1, len(f) // 100)].sum() / f.sum())}
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1)); print(meta, flush=True)
    embs = {("truth", 0): centres}
    for d in (15, 30):
        C = cbow(docs, d, c["window"])
        embs[("cbow", d)] = np.vstack([C, np.full((len(centres) - len(C), d), np.nan)]) if len(C) < len(centres) else C
    for (emb, d), W in embs.items():
        for n in (1, 2):
            fn = OUT / f"{tag}_{emb}_d{d}_n{n}.json"
            if fn.exists():
                continue
            ok = ~np.isnan(W).any(1); ok[np.setdiff1d(np.arange(len(W)), used)] = False
            if n == 1:
                Xc = W[ok]
            else:
                b = big[ok[big[:, 0]] & ok[big[:, 1]]]; Xc = np.hstack([W[b[:, 0]], W[b[:, 1]]])
            t0 = time.time()
            try:
                res = CT.measure(Xc, 2, 10000, 5000)
            except ValueError:
                res = CT.measure(Xc + np.random.default_rng(0).normal(0, 1e-6 * (float(np.nanstd(Xc)) or 1), Xc.shape), 2, 10000, 5000)
            if n == 2:
                U = np.unique(Xc, axis=0)
                res["twonn_all"] = M.twonn(U[np.random.default_rng(0).permutation(len(U))[:200000]], seed=0)["d"]
                res["N_all"] = int(min(len(U), 200000))
            res.update({"object": obj, "embedding": emb, "d": d, "ngram": n, "seconds": time.time() - t0, **c})
            fn.write_text(json.dumps(res))
            print(tag, emb, d, n, "TwoNN", res["twonn"], "all", res.get("twonn_all"), f"{res['seconds']:.0f}s", flush=True)
    (OUT / f"{tag}_done.json").write_text("{}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--task", type=int)
    a = ap.parse_args(); T = tasks()
    if a.list:
        for i, t in enumerate(T): print(i, t)
        return
    OUT.mkdir(exist_ok=True)
    obj, g = T[a.task]
    for cfg in GROUPS[g]:
        run(obj, cfg)


if __name__ == "__main__":
    main()
