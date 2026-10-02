"""Calibration of the text pipeline on objects of KNOWN dimension (proposal of V. A. Gromov).

Idea: a text is a trajectory over a "semantic" object; words are cells of a partition of that object.
So we take an object with known intrinsic dimension m, walk over it (trajectories = documents), discretise
the walk into "words" (cells of a partition), build word vectors exactly as for War and Peace
(SVD of the entropy-weighted word x document matrix, CBOW), and measure the dimension of word and
bigram clouds with the four estimators. Because m is known, this tells which (embedding, estimator)
combination can be trusted. Reference cloud ("truth"): the cell centres in the object's own coordinates.

Objects (true m):  torus T^m in R^(2m) and sphere S^m in R^(m+1) for m = 2, 4, 6 (Brownian walks);
                   Lorenz attractor (correlation dimension ~2.05, box dimension ~2.06; ODE trajectories).
Partitions:        kmeans (MiniBatchKMeans centres), random (random sample points as Voronoi centres),
                   aco (ant-colony clustering, see aco_centres; slower, optional).
Whitney: an m-dim manifold embeds in R^(2m); with word vectors of length d only m <= d/2 is guaranteed
to be representable without self-intersections, so d = 5/10/15 bounds what each embedding can show.

  python calib_trajectories.py --objects torus2 sphere2 torus4 ... --partitions kmeans random
Each finished cloud is written to calib_results/<object>_<partition>_<emb>_d<d>_n<n>.json (restart-safe).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import id_methods as M
from schweinhart_mst import DEFAULT_ALPHAS, geometric_ladder, schweinhart, summarise

TRUE_DIM = {"torus2": 2, "torus4": 4, "torus6": 6, "torus8": 8, "torus10": 10,
            "sphere2": 2, "sphere4": 4, "sphere6": 6, "sphere8": 8, "lorenz": 2.06}


# ----------------------------------------------------------------------------------------- objects
def walk_torus(m, n_traj, length, sigma, rng):
    th = rng.uniform(0, 2 * np.pi, (n_traj, m))
    out = np.empty((n_traj, length, 2 * m))
    for t in range(length):
        out[:, t, :m], out[:, t, m:] = np.cos(th), np.sin(th)
        th = th + sigma * rng.standard_normal(th.shape)
    return out / np.sqrt(m)  # unit-ish scale, same for every m


def walk_sphere(m, n_traj, length, sigma, rng):
    x = rng.standard_normal((n_traj, m + 1))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    out = np.empty((n_traj, length, m + 1))
    for t in range(length):
        out[:, t] = x
        x = x + sigma * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return out


def walk_lorenz(n_traj, length, dt, rng, s=10.0, r=28.0, b=8 / 3, every=2):
    def f(z):
        x, y, w = z[:, 0], z[:, 1], z[:, 2]
        return np.stack([s * (y - x), x * (r - w) - y, x * y - b * w], 1)
    z = rng.normal(0, 1, (n_traj, 3)) + np.array([0, 0, 25])
    for _ in range(2000):  # transient
        k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    out = np.empty((n_traj, length, 3))
    for t in range(length * every):
        if t % every == 0:
            out[:, t // every] = z
        k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
        z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (out - out.reshape(-1, 3).mean(0)) / out.reshape(-1, 3).std()


def trajectories(name, n_traj, length, rng, sigma):
    if name.startswith("torus"):
        return walk_torus(int(name[5:]), n_traj, length, sigma, rng)
    if name.startswith("sphere"):
        return walk_sphere(int(name[6:]), n_traj, length, sigma, rng)
    if name == "lorenz":
        return walk_lorenz(n_traj, length, 0.005, rng)
    raise ValueError(name)


# -------------------------------------------------------------------------------------- partitions
def aco_centres(X, V, rng, n_ants=4, iters=10):
    """Ant-colony-style partition (pheromone-guided k-centres): each ant builds a set of V centres by
    sampling points with probability ~ pheromone * coverage gain; pheromone is reinforced on the centres
    of the best solution (smallest mean distance to nearest centre). Cheap heuristic in the ACO family."""
    from sklearn.neighbors import NearestNeighbors
    S = X[rng.choice(len(X), min(len(X), 10 * V), replace=False)]
    tau = np.ones(len(S))
    best, best_cost = None, np.inf
    for _ in range(iters):
        for _ in range(n_ants):
            p = tau / tau.sum()
            idx = rng.choice(len(S), V, replace=False, p=p)
            d, _ = NearestNeighbors(n_neighbors=1).fit(S[idx]).kneighbors(S)
            cost = d.mean()
            if cost < best_cost:
                best, best_cost = idx, cost
        tau *= 0.9
        tau[best] += 1.0
    return S[best]


def partition(X, V, kind, rng):
    if kind == "random":
        return X[rng.choice(len(X), V, replace=False)]
    if kind == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=V, batch_size=8192, n_init=1, random_state=0, max_iter=50)
        km.fit(X[rng.choice(len(X), min(len(X), 30 * V), replace=False)])
        return km.cluster_centers_
    if kind == "aco":
        return aco_centres(X, V, rng)
    raise ValueError(kind)


def to_texts(traj, centres, sent_len=20):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(centres)
    docs = []
    for tr in traj:
        ids = nn.kneighbors(tr, return_distance=False)[:, 0]
        keep = np.r_[True, ids[1:] != ids[:-1]]  # staying in a cell = one word
        ids = ids[keep]
        docs.append([ids[i:i + sent_len].tolist() for i in range(0, len(ids), sent_len)])
    return docs


# ---------------------------------------------------------------------------------------- embeddings
def svd_gromov(docs, V, dmax=15):
    """s_ij = (1 - eps_i) k_ij / sum_i k_ij  (eps_i: normalised entropy of word i over documents), SVD rows U."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    rows, cols = [], []
    for j, d in enumerate(docs):
        for s in d:
            rows += s
            cols += [j] * len(s)
    K = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, len(docs))).tocsr()
    K.sum_duplicates()
    p = K.multiply(1 / np.maximum(K.sum(1), 1e-12)).tocsr()
    pa = p.toarray()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pa > 0, pa * np.log(pa), 0.0), 1) / np.log(len(docs))
    Sm = K.multiply(1 / np.maximum(np.asarray(K.sum(0)), 1e-12)).multiply((1 - ent)[:, None]).tocsr()
    u, s, vt = svds(Sm.astype(np.float64), k=dmax)
    order = np.argsort(-s)
    return u[:, order], ent


def cbow(docs, d):
    from gensim.models import Word2Vec
    sents = [[str(w) for w in s] for doc in docs for s in doc]
    m = Word2Vec(sents, vector_size=d, sg=0, window=5, negative=5, min_count=1, epochs=30, seed=1, workers=1)
    V = max(int(w) for w in m.wv.index_to_key) + 1
    out = np.full((V, d), np.nan)
    for w in m.wv.index_to_key:
        out[int(w)] = m.wv[w]
    return out


def bigrams_of(docs):
    return np.array(sorted({(a, b) for doc in docs for s in doc for a, b in zip(s, s[1:])}), dtype=np.int64)


# ---------------------------------------------------------------------------------------- estimators
def measure(X, ladders, max_points, h_iter, seed=0):
    rng = np.random.default_rng(seed)
    X = np.unique(X, axis=0)
    if len(X) > max_points:
        X = X[rng.permutation(len(X))[:max_points]]
    N = len(X)
    res = {"N": int(N), "ambient": int(X.shape[1])}
    sizes = geometric_ladder(N, max(N // 16, 50), 10)
    runs = [summarise(schweinhart(X, sizes, DEFAULT_ALPHAS, seed=s, log=lambda *_: None)["fits"]) for s in range(1, ladders + 1)]
    mins = [r["min_d"] for r in runs if r["min_d"] is not None]
    maxs = [r["max_d"] for r in runs if r["max_d"] is not None]
    meds = [r["median_d"] for r in runs if r["min_d"] is not None]
    res["schweinhart"] = {"min": min(mins) if mins else None, "max": max(maxs) if maxs else None,
                          "median": float(np.median(meds)) if meds else None,
                          "ladders_ok": len(mins), "ladders": ladders}
    res["twonn"] = M.twonn(X, seed=seed)["d"]
    res["fishers"] = M.fishers(X[rng.permutation(N)[:20000]] if N > 20000 else X)["d"]
    h = M.hidalgo(X[rng.permutation(N)[:5000]] if N > 5000 else X, K=2, seed=seed, Niter=h_iter)
    res["hidalgo"] = {"d_k": h["d_k"], "p_k": h["p_k"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=["torus2", "sphere2", "torus4", "sphere4", "torus6", "sphere6", "lorenz"])
    ap.add_argument("--partitions", nargs="+", default=["random", "kmeans", "aco"])
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--n-traj", type=int, default=500)
    ap.add_argument("--length", type=int, default=1500)
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 15, 30])
    ap.add_argument("--max-points", type=int, default=10000)
    ap.add_argument("--ladders", type=int, default=2)
    ap.add_argument("--hidalgo-iter", type=int, default=5000)
    ap.add_argument("--out", default="calib_results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    for obj in a.objects:
        rng = np.random.default_rng(sum(map(ord, obj)))
        m = TRUE_DIM[obj]
        # step ~ half the typical spacing between V cells (angle units for the torus, chord units on the sphere)
        sigma = (np.pi if obj.startswith("torus") else 1.0) * a.vocab ** (-1 / m) if obj != "lorenz" else None
        traj = trajectories(obj, a.n_traj, a.length, rng, sigma)
        X = traj.reshape(-1, traj.shape[-1])
        for part in a.partitions:
            tag = f"{obj}_{part}"
            if (out / f"{tag}_done.json").exists():
                continue
            t0 = time.time()
            centres = partition(X, a.vocab, part, np.random.default_rng(1))
            docs = to_texts(traj, centres)
            big = bigrams_of(docs)
            n_tok = sum(len(s) for d in docs for s in d)
            used = np.unique([w for d in docs for s in d for w in s])
            meta = {"object": obj, "true_dim": m, "partition": part, "vocab": int(len(used)), "tokens": int(n_tok),
                    "bigrams": int(len(big)), "documents": len(docs), "sigma": sigma, "seconds_text": time.time() - t0}
            (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1))
            print(meta, flush=True)
            U, _ = svd_gromov(docs, len(centres), max(a.dims))
            embs = {"truth": {0: centres}}
            embs["svd"] = {d: U[:, :d] for d in a.dims}
            embs["cbow"] = {}
            for d in a.dims:
                C = cbow(docs, d)
                C = np.vstack([C, np.full((len(centres) - len(C), d), np.nan)]) if len(C) < len(centres) else C
                embs["cbow"][d] = C
            for emb, by_d in embs.items():
                for d, W in by_d.items():
                    for n in (1, 2):
                        f = out / f"{tag}_{emb}_d{d}_n{n}.json"
                        if f.exists():
                            continue
                        ok = ~np.isnan(W).any(1)
                        ok[np.setdiff1d(np.arange(len(W)), used)] = False
                        if n == 1:
                            Xc = W[ok]
                        else:
                            b = big[ok[big[:, 0]] & ok[big[:, 1]]]
                            Xc = np.hstack([W[b[:, 0]], W[b[:, 1]]])
                        t1 = time.time()
                        r = measure(Xc, a.ladders, a.max_points, a.hidalgo_iter)
                        r.update({"object": obj, "true_dim": m, "partition": part, "embedding": emb, "d": d, "ngram": n,
                                  "seconds": time.time() - t1})
                        f.write_text(json.dumps(r))
                        sw = r["schweinhart"]
                        print(f"{tag:16s} {emb:5s} d={d:<2d} n={n} N={r['N']:6d} | Schw {sw['min']}-{sw['max']} "
                              f"| TwoNN {r['twonn']:.2f} | FisherS {r['fishers']} | Hidalgo {np.round(r['hidalgo']['d_k'], 2).tolist()} "
                              f"({r['seconds']:.0f}s)", flush=True)
            (out / f"{tag}_done.json").write_text("{}")


if __name__ == "__main__":
    main()
