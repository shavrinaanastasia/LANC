"""Pseudolanguage (notebook "Pseudolanguage.ipynb") added to the calibration on objects of known dimension.

Notebook recipe (kept):  points uniform in the unit square, n = 5000 before cutting k circular holes
(HOLES list of the notebook); every point is a pseudo-word; a text is a random walk over the points;
next word chosen with probability ~ 1 / distance^2 ("dist" walk, final pipeline of the notebook) or
uniformly among points closer than eps ("eps" walk, first attempt); text lengths = Russian text lengths
(russian_Prussner_wolframe100.csv) divided by k = 60000 // n_words; 1000 texts; CBOW (gensim defaults
as in the notebook: window 5, 5 epochs, min_count 3) for R = 2, 3, 5.

Extensions for the dimension question: the same construction in the m-dimensional cube (m = 2..6) with
spherical holes, so the true dimension m is known; embeddings SVD (word x text, as in the article) and
CBOW with the notebook settings and with our War-and-Peace settings (30 epochs, min_count 1), d = 2..15;
the four estimators (calib_trajectories.measure) on words and bigrams; reference = the points themselves.

  python pseudolang.py --dims-obj 2 3 4 6 --walks dist eps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from calib_trajectories import measure, svd_gromov

HOLES_2D = [(0.6278074516646731, 0.470100438958845, 0.0432615257778067),
            (0.37469647400651906, 0.750911853581118, 0.06816506852404625),
            (0.9120667218505845, 0.8704931132774198, 0.058779359756224),
            (0.4615297119105358, 0.3798722031157237, 0.1273663835888241),
            (0.25705321006066895, 0.5229188046782313, 0.056126898015020674),
            (0.638371106321473, 0.0869068200947261, 0.0760185678889761),
            (0.8392414924899623, 0.7693675377746602, 0.047224095760708545),
            (0.6718818429285832, 0.33672580183276746, 0.061514434507242116),
            (0.07565037365891619, 0.7504467286750023, 0.07411518358902063),
            (0.8786832426905251, 0.9582481190111606, 0.03214797135430035)]


def holes_nd(m, k, seed=2026, r_min=0.05, r_max=0.25):
    """Non-overlapping balls in [0,1]^m (generalises generate_circles_adaptive of the notebook)."""
    if m == 2:
        return [(np.array(h[:2]), h[2]) for h in HOLES_2D[:k]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        for _ in range(500):
            r = rng.uniform(r_min, r_max)
            c = rng.uniform(r, 1 - r, m)
            if all(np.linalg.norm(c - c2) >= r + r2 for c2, r2 in out):
                out.append((c, r))
                break
    return out


def make_points(m, n, k, seed=2026):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, m))
    keep = np.ones(n, bool)
    for c, r in holes_nd(m, k):
        keep &= np.linalg.norm(x - c, axis=1) > r
    return x[keep]


def walk(data, length, kind, rng, eps):
    i = rng.integers(len(data))
    out = [i]
    sq = (data ** 2).sum(1)
    for _ in range(length - 1):
        d2 = sq + sq[i] - 2 * data @ data[i]
        d2[i] = np.inf
        if kind == "dist":  # notebook: p ~ 1 / squared distance (it says "distance" but uses squared)
            p = 1 / np.maximum(d2, 1e-18)
        else:  # "eps": uniform over neighbours closer than eps
            p = (d2 < eps ** 2).astype(float)
            if p.sum() == 0:
                p = 1 / np.maximum(d2, 1e-18)
        i = rng.choice(len(data), p=p / p.sum())
        out.append(i)
    return out


def text_lengths(path, n_words, fallback_rng):
    k = max(60000 // n_words, 1)
    if Path(path).exists():
        L = np.loadtxt(path, delimiter=",", usecols=0)
        src = path
    else:  # placeholder until the real file is supplied: log-normal, median ~ 2000 words
        L = np.exp(fallback_rng.normal(np.log(2000), 0.9, 10000))
        src = "lognormal placeholder (median 2000)"
    return np.maximum((L // k).astype(int), 5), k, src


def cbow_vectors(texts, d, epochs, min_count, V):
    from gensim.models import Word2Vec
    sents = [[str(w) for w in t] for t in texts]
    m = Word2Vec(sents, vector_size=d, sg=0, window=5, negative=5, min_count=min_count, epochs=epochs,
                 seed=1, workers=1)
    out = np.full((V, d), np.nan)
    for w in m.wv.index_to_key:
        out[int(w)] = m.wv[w]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims-obj", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--holes", type=int, default=4)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--n-texts", type=int, default=1000)
    ap.add_argument("--walks", nargs="+", default=["dist", "eps"])
    ap.add_argument("--emb-dims", type=int, nargs="+", default=[2, 3, 5, 10, 15])
    ap.add_argument("--lengths-csv", default="russian_Prussner_wolframe100.csv")
    ap.add_argument("--ladders", type=int, default=3)
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--hidalgo-iter", type=int, default=5000)
    ap.add_argument("--out", default="pseudo_results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    for m in a.dims_obj:
        data = make_points(m, a.n_points, a.holes)
        V = len(data)
        eps = 1.5 * (1 / V) ** (1 / m)  # ~ a few neighbours on average
        for kind in a.walks:
            tag = f"cube{m}_h{a.holes}_{kind}"
            if (out / f"{tag}_done.json").exists():
                continue
            rng = np.random.default_rng(2026 + m)
            from sklearn.neighbors import NearestNeighbors
            nn_med = float(np.median(NearestNeighbors(n_neighbors=2).fit(data).kneighbors(data)[0][:, 1]))
            lens, k, src = text_lengths(a.lengths_csv, V, rng)
            t0 = time.time()
            texts = [walk(data, int(rng.choice(lens)), kind, np.random.default_rng(i * 8 + 2), eps)
                     for i in range(a.n_texts)]
            steps = np.concatenate([np.linalg.norm(data[t[1:]] - data[t[:-1]], axis=1) for t in texts if len(t) > 1])
            big = np.array(sorted({(x, y) for t in texts for x, y in zip(t, t[1:])}), dtype=np.int64)
            used = np.unique(np.concatenate(texts))
            meta = {"object": f"cube{m} minus {a.holes} holes", "true_dim": m, "walk": kind, "words": V,
                    "words_used": int(len(used)), "tokens": int(sum(map(len, texts))), "bigrams": int(len(big)),
                    "length_divisor_k": k, "lengths_source": src, "eps": eps,
                    "step_median": float(np.median(steps)), "step_p90": float(np.quantile(steps, 0.9)),
                    "nn_distance_median": nn_med,
                    "seconds_texts": time.time() - t0}
            (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1))
            print(meta, flush=True)
            docs = [[t] for t in texts]  # one "sentence" per text, as in the notebook (LineSentence)
            U, _ = svd_gromov(docs, V, max(a.emb_dims))
            embs = {"truth": {m: data}, "svd": {d: U[:, :d] for d in a.emb_dims}}
            embs["cbow_nb"] = {d: cbow_vectors(texts, d, 5, 3, V) for d in a.emb_dims}    # notebook settings
            embs["cbow_wap"] = {d: cbow_vectors(texts, d, 30, 1, V) for d in a.emb_dims}  # War-and-Peace settings
            for emb, by_d in embs.items():
                for d, W in by_d.items():
                    for n in (1, 2):
                        f = out / f"{tag}_{emb}_d{d}_n{n}.json"
                        if f.exists():
                            continue
                        ok = ~np.isnan(W).any(1)
                        mask = np.zeros(V, bool); mask[used] = True; ok &= mask
                        if n == 1:
                            Xc = W[ok]
                        else:
                            b = big[ok[big[:, 0]] & ok[big[:, 1]]]
                            Xc = np.hstack([W[b[:, 0]], W[b[:, 1]]])
                        t1 = time.time()
                        r = measure(Xc, a.ladders, a.max_points, a.hidalgo_iter)
                        r.update({"object": tag, "true_dim": m, "embedding": emb, "d": d, "ngram": n,
                                  "seconds": time.time() - t1})
                        f.write_text(json.dumps(r))
                        sw = r["schweinhart"]
                        print(f"{tag:18s} {emb:8s} d={d:<2d} n={n} N={r['N']:6d} | Schw {sw['min']}-{sw['max']} "
                              f"| TwoNN {r['twonn']:.2f} | FisherS {r['fishers']} | Hidalgo "
                              f"{np.round(r['hidalgo']['d_k'], 2).tolist()} ({r['seconds']:.0f}s)", flush=True)
            (out / f"{tag}_done.json").write_text("{}")


if __name__ == "__main__":
    main()
