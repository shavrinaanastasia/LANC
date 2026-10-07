"""E15: what makes CBOW recover m on the pseudolanguage (E7, ~m) but give 1.6-2.5 m on tori (E10-E14)?

Cross-swap the three ways the two generators differ, one at a time, on the same m:
  manifold : torus T^m (closed, flat; words = points on it)  vs  cube [0,1]^m minus 4 balls (pseudolanguage)
  walk     : brown -- continuous Gaussian steps (torus: angle sigma = pi V^(-1/m); cube: sigma = 0.5 V^(-1/m), reflecting,
                      steps into holes rejected), positions quantised to the nearest word, repeats merged (calib2/3 style)
             eps   -- walk on the word points themselves: next word uniform among words within eps
                      (cube: eps = 1.5 V^(-1/m) as in E7; torus: the same in angle units x 2 pi), no repeats possible
  text     : sent20 -- 500 walks, sentences of 20 words (calib2/3 style, ~ 450k tokens)
             long   -- 1000 texts, lengths from russian_Prussner_wolframe100.csv / k, each text one CBOW sentence (E7 style)
m = 2, 4, 6; V = 5000 word points, uniform on the manifold in both cases. 3 x 2^3 = 24 configurations.
Measured: exact coordinates and CBOW d = 15 / 30 (seed 1, as everywhere), words and bigrams, TwoNN / Schweinhart / Hidalgo /
FisherS (calib_trajectories.measure); plus tokens, typical step / nearest-neighbour distance, share of repeated words.

  python calib6.py --list ; python calib6.py --task i
"""
from __future__ import annotations

import argparse, itertools, json, time
from pathlib import Path
import numpy as np

import calib_trajectories as CT

OUT = Path("calib6_results")
V = 5000
CONFIGS = [dict(m=m, manifold=a, walk=w, text=t) for m, a, w, t in itertools.product((2, 4, 6), ("torus", "cube"), ("brown", "eps"), ("sent20", "long"))]


def name(c):
    return f"m{c['m']}_{c['manifold']}_{c['walk']}_{c['text']}"


# ------------------------------------------------------------------ manifolds
def holes(m):
    import pseudolang as PL
    return PL.holes_nd(m, 4)


def inside_holes(x, H):
    bad = np.zeros(len(x), bool)
    for c, r in H:
        bad |= np.linalg.norm(x - c, axis=1) <= r
    return bad


def word_points(c, rng):
    m = c["m"]
    if c["manifold"] == "torus":
        th = rng.uniform(0, 2 * np.pi, (V, m))
        return th, np.hstack([np.cos(th), np.sin(th)]) / np.sqrt(m)   # intrinsic (angles), embedded coordinates
    H = holes(m); pts = []
    while sum(len(p) for p in pts) < V:
        x = rng.uniform(0, 1, (4 * V, m)); pts.append(x[~inside_holes(x, H)])
    x = np.vstack(pts)[:V]
    return x, x


def text_lengths(c, rng, n_tokens_target=None):
    if c["text"] == "sent20":
        return None
    import pseudolang as PL
    lens, _, _ = PL.text_lengths("russian_Prussner_wolframe100.csv", V, rng)
    return [int(rng.choice(lens)) for _ in range(1000)]


# ------------------------------------------------------------------ walks
def brown_words(c, intr, n_words, rng, start=None):
    """Continuous walk, quantised to the nearest word, repeats merged; returns >= n_words word ids."""
    from sklearn.neighbors import NearestNeighbors
    m = c["m"]
    if c["manifold"] == "torus":
        emb = lambda th: np.hstack([np.cos(th), np.sin(th)])
        nn = NearestNeighbors(n_neighbors=1).fit(emb(intr))
        sigma = np.pi * V ** (-1 / m)
        x = intr[rng.integers(V)].copy()
        step = lambda x: x + sigma * rng.standard_normal(m)
        to_e = lambda X: emb(np.asarray(X))
    else:
        H = holes(m); nn = NearestNeighbors(n_neighbors=1).fit(intr)
        sigma = 0.5 * V ** (-1 / m)
        x = intr[rng.integers(V)].copy()
        def step(x):
            y = np.abs(x + sigma * rng.standard_normal(m)); y = 1 - np.abs(1 - y)   # reflect into [0, 1]
            return x if inside_holes(y[None], H)[0] else y
        to_e = lambda X: np.asarray(X)
    out = []
    while len(out) < n_words:
        block = []
        for _ in range(4 * (n_words - len(out)) + 50):
            x = step(x); block.append(x.copy())
        ids = nn.kneighbors(to_e(block), return_distance=False)[:, 0]
        for i in ids:
            if not out or out[-1] != i:
                out.append(int(i))
    return out[:n_words]


def eps_neighbours(c, intr):
    from sklearn.neighbors import NearestNeighbors
    m = c["m"]
    if c["manifold"] == "torus":
        eps = 2 * np.pi * 1.5 * V ** (-1 / m)
        X = np.hstack([np.cos(intr), np.sin(intr)]); r = eps  # chord distance of (cos, sin) ~ angular distance
    else:
        eps = 1.5 * V ** (-1 / m); X = intr; r = eps
    nb = NearestNeighbors(radius=r).fit(X).radius_neighbors(X, return_distance=False)
    nn1 = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X, return_distance=False)[:, 1]
    return [np.setdiff1d(n, [i]) if len(n) > 1 else np.array([nn1[i]]) for i, n in enumerate(nb)]


def eps_words(nb, n_words, rng):
    i = int(rng.integers(V)); out = [i]
    for _ in range(n_words - 1):
        i = int(rng.choice(nb[i])); out.append(i)
    return out


def make_docs(c, intr, rng):
    lens = text_lengths(c, rng)
    nb = eps_neighbours(c, intr) if c["walk"] == "eps" else None
    gen = (lambda n: eps_words(nb, n, rng)) if nb is not None else (lambda n: brown_words(c, intr, n, rng))
    if lens is None:  # sent20: 500 walks of ~ 900 words (calib3 gave ~ 460k tokens from 500 x 1500 steps)
        docs = []
        for _ in range(500):
            w = gen(920)
            docs.append([w[i:i + 20] for i in range(0, len(w), 20)])
        return docs
    return [[gen(L)] for L in lens if L > 1]


def cbow(docs, d):
    from gensim.models import Word2Vec
    sents = [[str(w) for w in s] for doc in docs for s in doc]
    m = Word2Vec(sents, vector_size=d, sg=0, window=5, negative=5, min_count=1, epochs=30, seed=1, workers=1)
    out = np.full((V, d), np.nan)
    for w in m.wv.index_to_key:
        out[int(w)] = m.wv[w]
    return out


def measure(X):
    X = X[~np.isnan(X).any(1)]
    try:
        r = CT.measure(X, 2, 10000, 5000)
    except ValueError:
        r = CT.measure(X + np.random.default_rng(0).normal(0, 1e-6 * (float(np.std(X)) or 1), X.shape), 2, 10000, 5000)
    h = r["hidalgo"]
    return {"N": r["N"], "twonn": r["twonn"], "schw": r["schweinhart"]["median"], "schw_min": r["schweinhart"]["min"],
            "schw_max": r["schweinhart"]["max"], "hid": float(np.dot(h["d_k"], h["p_k"])), "fishers": r["fishers"]}


def run(c):
    fn = OUT / f"{name(c)}.json"
    if fn.exists():
        return
    t0 = time.time(); rng = np.random.default_rng(c["m"] * 100 + len(c["manifold"]) * 10 + len(c["walk"]) + len(c["text"]))
    intr, emb = word_points(c, rng)
    docs = make_docs(c, intr, rng)
    words = np.concatenate([np.asarray(s) for d in docs for s in d])
    count = np.bincount(words, minlength=V); used = count > 0
    big = CT.bigrams_of(docs)
    from sklearn.neighbors import NearestNeighbors
    nnd = NearestNeighbors(n_neighbors=2).fit(emb).kneighbors(emb)[0][:, 1]
    stepd = np.linalg.norm(emb[big[:, 1]] - emb[big[:, 0]], axis=1)
    res = {**c, "case": name(c), "tokens": int(len(words)), "used": int(used.sum()), "bigrams": int(len(big)),
           "docs": len(docs), "sentences": int(sum(len(d) for d in docs)),
           "step_over_nn": float(np.median(stepd) / np.median(nnd)), "count_median": float(np.median(count[used])),
           "count_top1pct_share": float(np.sort(count)[::-1][: V // 100].sum() / count.sum())}
    print(res, flush=True)
    res["truth"] = {"words": measure(emb[used]), "bigrams": measure(np.hstack([emb[big[:, 0]], emb[big[:, 1]]]))}
    res["cbow"] = {}
    for d in (15, 30):
        W = cbow(docs, d)
        res["cbow"][str(d)] = {"words": measure(W[used]), "bigrams": measure(np.hstack([W[big[:, 0]], W[big[:, 1]]]))}
        print(name(c), d, {k: round(v["twonn"], 2) for k, v in res["cbow"][str(d)].items()}, f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = time.time() - t0
    fn.write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.list:
        for i, c in enumerate(CONFIGS): print(i, name(c))
        return
    OUT.mkdir(exist_ok=True)
    run(CONFIGS[a.task])


if __name__ == "__main__":
    main()
