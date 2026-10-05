"""E10: second calibration round (Gromov, 2026-10-05).

Three questions:
  A. What happens at larger embedding dimension d (up to 100), for higher m (T12, S12) and for
     fractal / multifractal objects with known dimension?
  B. How do the results depend on the number of points (vocabulary V) and of trajectories (documents)?
  C. "Blind" setting: true m unknown. Collect what could reveal it indirectly (Whitney / Takens):
     false nearest neighbours (Kennel 1992) and Cao's E1/E2 on delay vectors -- (i) of word vectors
     along the text (k-grams, k = 1..8), (ii) classical scalar delay embedding of the trajectory --
     plus k-gram cloud dimension. The inverse model (observables -> m) is fitted offline with
     leave-one-object-out validation.

Objects
  torus<m>, sphere<m>        Brownian walks as in calib_trajectories.py (m up to 12)
  lorenz                     as before (D ~ 2.06)
  rossler, l96_<N>           flows; reference = Kaplan-Yorke dimension computed here from the Lyapunov spectrum
  gasket, tetra, simplex4, simplex7, carpet, sponge
                             self-similar fractals; random walk on the level-L cell graph (D0 analytic)
  mfgasket, mftetra          multifractal measures on the gasket / tetrahedron (unequal IFS weights),
                             Metropolis walk whose stationary law is the measure; D0, D1, D2 analytic

  python calib2.py --list            # task table
  python calib2.py --task 7          # one array task (restart-safe: finished clouds are skipped)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

import calib_trajectories as CT
import id_methods as M

OUT = Path("calib2_results")

# ------------------------------------------------------------------------------------------- objects
FRACTALS = {
    # name: (kind, n, level, weights)  kind "simplex": n-simplex IFS with ratio 1/2 in R^n; "grid": carpet / sponge
    "gasket": ("simplex", 2, 9, None),
    "tetra": ("simplex", 3, 7, None),
    "simplex4": ("simplex", 4, 6, None),
    "simplex7": ("simplex", 7, 5, None),
    "carpet": ("grid", 2, 5, None),
    "sponge": ("grid", 3, 4, None),
    "mfgasket": ("simplex", 2, 9, (0.60, 0.25, 0.15)),
    "mftetra": ("simplex", 3, 7, (0.40, 0.30, 0.20, 0.10)),
}


def fractal_dims(name):
    kind, n, L, p = FRACTALS[name]
    if kind == "simplex":
        k, r = n + 1, 0.5
    else:
        k, r = (8 if n == 2 else 20), 1 / 3
    p = np.full(k, 1 / k) if p is None else np.asarray(p)
    D = {}
    for q in (0, 1, 2):
        if q == 1:
            D["D1"] = float(-(p * np.log(p)).sum() / -math.log(r))
        else:
            D[f"D{q}"] = float(math.log((p ** q).sum()) / ((q - 1) * math.log(r)))
    return D


def simplex_cells(n, L, p):
    """Level-L cells of the n-simplex gasket: centres, adjacency (shared vertex), cell measure."""
    corners = np.vstack([np.zeros(n), np.eye(n)])  # n+1 corners in R^n
    k = n + 1
    words = np.array(list(itertools.product(range(k), repeat=L)), dtype=np.int16)
    scale = 2 ** L
    # integer coordinates of cell origin: sum corner[w_j] * 2^(L-1-j)
    origin = np.zeros((len(words), n), dtype=np.int64)
    for j in range(L):
        origin += (corners[words[:, j]] * 2 ** (L - 1 - j)).astype(np.int64)
    centres = (origin + corners.mean(0)) / scale
    # vertices of each cell = origin + corner (integer lattice) -> adjacency by shared vertex
    vert_ids = {}
    cell_verts = np.empty((len(words), k), dtype=np.int64)
    for c in range(k):
        V = origin + corners[c].astype(np.int64)
        for i, v in enumerate(map(tuple, V)):
            cell_verts[i, c] = vert_ids.setdefault(v, len(vert_ids))
    w = np.ones(len(words)) if p is None else np.prod(np.asarray(p)[words], axis=1)
    return centres, cell_verts, w / w.sum()


def grid_cells(n, L):
    keep1 = [(i, j) for i in range(3) for j in range(3) if not (i == 1 and j == 1)] if n == 2 else \
        [(i, j, l) for i in range(3) for j in range(3) for l in range(3) if [i, j, l].count(1) < 2]
    cells = np.zeros((1, n), dtype=np.int64)
    for _ in range(L):
        cells = (cells[:, None, :] * 3 + np.array(keep1)[None]).reshape(-1, n)
    centres = (cells + 0.5) / 3 ** L
    return centres, cells


def neighbours_simplex(cell_verts):
    from collections import defaultdict
    by_v = defaultdict(list)
    for i, vs in enumerate(cell_verts):
        for v in vs:
            by_v[v].append(i)
    nb = [set() for _ in range(len(cell_verts))]
    for cs in by_v.values():
        for a in cs:
            nb[a].update(c for c in cs if c != a)
    return [np.array(sorted(s)) for s in nb]


def neighbours_grid(cells):
    idx = {tuple(c): i for i, c in enumerate(cells)}
    n = cells.shape[1]
    nb = []
    for c in cells:
        s = []
        for ax in range(n):
            for sgn in (-1, 1):
                d = c.copy(); d[ax] += sgn
                j = idx.get(tuple(d))
                if j is not None:
                    s.append(j)
        nb.append(np.array(s))
    return nb


def walk_graph(centres, nb, w, n_traj, length, rng):
    """Metropolis random walk on the cell graph with stationary law w (uniform w -> plain fractal)."""
    deg = np.array([len(x) for x in nb])
    cur = rng.choice(len(centres), n_traj, p=w)
    out = np.empty((n_traj, length, centres.shape[1]))
    for t in range(length):
        out[:, t] = centres[cur]
        prop = np.array([nb[c][rng.integers(len(nb[c]))] for c in cur])
        acc = rng.random(n_traj) < np.minimum(1, (w[prop] * deg[cur]) / (w[cur] * deg[prop]))
        cur = np.where(acc, prop, cur)
    return out


def fractal_traj(name, n_traj, length, rng):
    kind, n, L, p = FRACTALS[name]
    if kind == "simplex":
        centres, cv, w = simplex_cells(n, L, p)
        nb = neighbours_simplex(cv)
    else:
        centres, cells = grid_cells(n, L)
        nb = neighbours_grid(cells)
        w = np.full(len(centres), 1 / len(centres))
    traj = walk_graph(centres, nb, w, n_traj, length, rng)
    # small jitter inside the cell (0.1 of the cell size) so that repeated visits are not identical points
    h = 0.5 ** L if kind == "simplex" else 3.0 ** -L
    return traj + rng.uniform(-0.1 * h, 0.1 * h, traj.shape)


# flows ---------------------------------------------------------------------------------------------
def f_rossler(z, a=0.2, b=0.2, c=5.7):
    x, y, w = z[..., 0], z[..., 1], z[..., 2]
    return np.stack([-y - w, x + a * y, b + w * (x - c)], -1)


def f_lorenz(z, s=10.0, r=28.0, b=8 / 3):
    x, y, w = z[..., 0], z[..., 1], z[..., 2]
    return np.stack([s * (y - x), x * (r - w) - y, x * y - b * w], -1)


def f_l96(z, F=8.0):
    return (np.roll(z, -1, -1) - np.roll(z, 2, -1)) * np.roll(z, 1, -1) - z + F


FLOWS = {"rossler": (f_rossler, 3, 0.02), "lorenz_auto": (f_lorenz, 3, 0.005),
         "l96_6": (f_l96, 6, 0.01), "l96_10": (f_l96, 10, 0.01)}


def rk4(f, z, dt):
    k1 = f(z); k2 = f(z + dt / 2 * k1); k3 = f(z + dt / 2 * k2); k4 = f(z + dt * k3)
    return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def kaplan_yorke(f, dim, dt, rng, steps=60000, transient=5000):
    z = rng.normal(0, 1, dim) + (np.array([0, 0, 25.0]) if f is f_lorenz else 0)
    for _ in range(transient):
        z = rk4(f, z, dt)
    Q = np.eye(dim); S = np.zeros(dim); eps = 1e-7
    for _ in range(steps):
        z1 = rk4(f, z, dt)
        J = np.empty((dim, dim))
        for j in range(dim):
            e = np.zeros(dim); e[j] = eps
            J[:, j] = (rk4(f, z + e, dt) - z1) / eps
        Q, R = np.linalg.qr(J @ Q)
        S += np.log(np.abs(np.diag(R)))
        z = z1
    lam = np.sort(S / (steps * dt))[::-1]
    c = np.cumsum(lam)
    j = int(np.max(np.where(c >= 0)[0])) if (c >= 0).any() else 0
    dky = j + 1 + c[j] / abs(lam[j + 1]) if j + 1 < dim else float(dim)
    return float(dky), lam.tolist()


def flow_traj(name, n_traj, length, rng, vocab):
    f, dim, dt = FLOWS[name]
    z = rng.normal(0, 1, (n_traj, dim)) + (np.array([0, 0, 25.0]) if f is f_lorenz else 0)
    for _ in range(3000):
        z = rk4(f, z, dt)
    # choose the sampling stride so that one step ~ the typical distance between neighbouring cell centres
    probe = []
    zz = z[:50].copy()
    for _ in range(4000):
        zz = rk4(f, zz, dt); probe.append(zz.copy())
    P = np.array(probe).reshape(-1, dim)
    from sklearn.neighbors import NearestNeighbors
    C = P[rng.choice(len(P), min(vocab, len(P)), replace=False)]
    spacing = np.median(NearestNeighbors(n_neighbors=2).fit(C).kneighbors(C)[0][:, 1])
    step = np.median(np.linalg.norm(np.diff(np.array(probe)[:, 0], axis=0), axis=1))
    stride = max(1, int(round(spacing / max(step, 1e-12))))
    out = np.empty((n_traj, length, dim))
    for t in range(length * stride):
        if t % stride == 0:
            out[:, t // stride] = z
        z = rk4(f, z, dt)
    mu, sd = out.reshape(-1, dim).mean(0), out.reshape(-1, dim).std()
    return (out - mu) / sd, stride


def make_traj(obj, n_traj, length, vocab, rng):
    info = {}
    if obj.startswith("torus") or obj.startswith("sphere"):
        m = int(obj.replace("torus", "").replace("sphere", ""))
        sigma = (np.pi if obj.startswith("torus") else 1.0) * vocab ** (-1 / m)
        traj = CT.trajectories(obj, n_traj, length, rng, sigma)
        info.update(true_dim=m, sigma=sigma)
    elif obj == "lorenz":
        traj = CT.trajectories("lorenz", n_traj, length, rng, None)
        info.update(true_dim=2.06)
    elif obj in FLOWS:
        traj, stride = flow_traj(obj, n_traj, length, rng, vocab)
        f, dim, dt = FLOWS[obj]
        dky, lam = kaplan_yorke(f, dim, dt, np.random.default_rng(7))
        info.update(true_dim=dky, lyapunov=lam, stride=stride)
    elif obj in FRACTALS:
        traj = fractal_traj(obj, n_traj, length, rng)
        D = fractal_dims(obj)
        info.update(true_dim=D["D0"], **D)
    else:
        raise ValueError(obj)
    return traj, info


# --------------------------------------------------------------------------------- FNN / Cao / k-grams
def sequences(docs):
    return [np.concatenate([np.asarray(s) for s in d]) for d in docs if sum(len(s) for s in d) > 0]


def corr_slopes(R, rng, n=3000, qs=(1e-3, 1e-2, 1e-1)):
    """Local slope of log C(eps) vs log eps of the correlation integral at the scales where C = qs."""
    from scipy.spatial.distance import pdist
    X = R[rng.permutation(len(R))[:n]]
    d = np.sort(pdist(X)); d = d[d > 0]
    out = {}
    for q in qs:
        i1, i2 = int(q * len(d) / 2), int(min(2 * q, 1.0) * len(d)) - 1
        if i1 < 10 or i2 <= i1 or d[i2] <= d[i1]:
            out[str(q)] = None
            continue
        out[str(q)] = float(math.log(i2 / i1) / math.log(d[i2] / d[i1]))
    return out


def delay_stats(series, kmax, n_query=4000, n_ref=30000, theiler=20, rtols=(2, 5, 10, 15), seed=0, orig=None):
    """series: list of arrays (T_i, p) -- multivariate observations along each document / trajectory.
    orig: optional list of arrays (T_i, q) -- the true state at each step (original space).
    Returns per k: FNN fractions (Kennel criterion 1 at several Rtol, + criterion 2), Cao E1, E2, TwoNN of k-cloud,
    correlation-integral slopes (Malinetskii & Potapov 2000, sec. 13.3: growth of the slope with the window signals
    false neighbours on folds, FNF), and -- when orig is given -- the share of reconstruction nearest neighbours whose
    original states are far apart (ratio to the true nearest-neighbour distance > 3 / > 10): FNN at small k, FNF at large k."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    idx = [(i, t) for i, s in enumerate(series) for t in range(len(s) - kmax - 1)]
    if len(idx) < 1000:
        return {"error": "too few windows"}
    idx = np.array(idx)
    ref = idx[rng.choice(len(idx), min(n_ref, len(idx)), replace=False)]
    qry = ref[:min(n_query, len(ref))]
    allx = np.concatenate(series)
    RA = float(np.sqrt((allx.std(0) ** 2).sum())) if allx.ndim > 1 else float(allx.std())

    A = allx if allx.ndim > 1 else allx[:, None]
    off = np.concatenate([[0], np.cumsum([len(x) for x in series])[:-1]])
    g_ref = off[ref[:, 0]] + ref[:, 1]

    def emb(g, k):
        return np.concatenate([A[g + j] for j in range(k)], axis=1)

    if orig is not None:
        O = np.concatenate(orig); O = O if O.ndim > 1 else O[:, None]
        Oref = O[g_ref]
        dO, iO = NearestNeighbors(n_neighbors=30).fit(Oref).kneighbors(Oref[:len(qry)])
        badO = (dO <= 0) | ((ref[iO, 0] == qry[:, None, 0]) & (np.abs(ref[iO, 1] - qry[:, None, 1]) < theiler))
        fO = np.argmax(~badO, axis=1)
        true_nn = np.where((~badO).any(1), dO[np.arange(len(qry)), fO], np.nan)

    res = {}
    prevE = prevEs = None
    for k in range(1, kmax + 1):
        R = emb(g_ref, k); Q = R[:len(qry)]
        nn = NearestNeighbors(n_neighbors=60 if k == 1 else 30).fit(R)
        dist, ind = nn.kneighbors(Q)
        bad = (dist <= 0) | ((ref[ind, 0] == qry[:, None, 0]) & (np.abs(ref[ind, 1] - qry[:, None, 1]) < theiler))
        first = np.argmax(~bad, axis=1); has = (~bad).any(1)
        pick_d = np.where(has, dist[np.arange(len(qry)), first], np.nan)
        pick_j = np.where(has, ind[np.arange(len(qry)), first], -1); ok = pick_j >= 0
        nxt_q = A[g_ref[:len(qry)][ok] + k]
        nxt_r = A[g_ref[pick_j[ok]] + k]
        extra = np.linalg.norm(nxt_q - nxt_r, axis=1)
        Rk = pick_d[ok]
        Rk1 = np.sqrt(Rk ** 2 + extra ** 2)
        fnn = {f"rtol{r}": float(np.mean(extra / Rk > r)) for r in rtols}
        fnn["atol2"] = float(np.mean(Rk1 / RA > 2))
        E = float(np.mean(Rk1 / Rk)); Es = float(np.mean(extra))
        Ru = np.unique(R, axis=0)
        tw = M.twonn(Ru[np.random.default_rng(0).permutation(len(Ru))[:10000]], seed=0)["d"] if len(Ru) > 100 else None
        res[k] = {"fnn": fnn, "E": E, "Estar": Es, "E1": (E / prevE) if prevE else None,
                  "E2": (Es / prevEs) if prevEs else None, "twonn_kcloud": tw, "n": int(ok.sum()),
                  "corr_slope": corr_slopes(R, np.random.default_rng(1))}
        if orig is not None:
            rho = np.linalg.norm(Oref[:len(qry)][ok] - Oref[pick_j[ok]], axis=1) / true_nn[ok]
            rho = rho[np.isfinite(rho)]
            res[k]["false_orig"] = {"gt3": float(np.mean(rho > 3)), "gt10": float(np.mean(rho > 10)),
                                    "median_ratio": float(np.median(rho))}
        # Cao: E1(k-1) = E(k)/E(k-1); stored at k
        prevE, prevEs = E, Es
    return res


def autocorr_lag(x):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * len(x))
    below = np.where(ac < 1 / math.e)[0]
    return int(below[0]) if len(below) else 1


# ---------------------------------------------------------------------------------------------- tasks
OBJ_A = ["torus2", "torus4", "torus6", "torus8", "torus10", "torus12", "sphere4", "sphere8", "sphere12",
         "lorenz", "rossler", "l96_6", "l96_10",
         "gasket", "tetra", "simplex4", "simplex7", "carpet", "sponge", "mfgasket", "mftetra"]
OBJ_B = ["torus4", "torus6", "sphere4", "lorenz"]


def tasks():
    T = [{"part": "A", "obj": o, "vocab": 5000, "n_traj": 500, "dims": [5, 10, 15, 30, 50, 100], "svd": True, "fnn": True}
         for o in OBJ_A]
    for o in OBJ_B:
        T.append({"part": "B_vocab", "obj": o, "grid": [{"vocab": v, "n_traj": 500} for v in (1000, 2000, 10000, 20000)]})
        T.append({"part": "B_ntraj", "obj": o, "grid": [{"vocab": 5000, "n_traj": n} for n in (100, 250, 1000, 2000)]})
    return T


def run_config(obj, vocab, n_traj, dims, svd, fnn, length=1500, ladders=2, max_points=10000, h_iter=5000):
    tag = f"{obj}_V{vocab}_T{n_traj}"
    if (OUT / f"{tag}_done.json").exists():
        return
    rng = np.random.default_rng(sum(map(ord, obj)))
    t0 = time.time()
    traj, info = make_traj(obj, n_traj, length, vocab, rng)
    X = traj.reshape(-1, traj.shape[-1])
    centres = CT.partition(X, vocab, "random", np.random.default_rng(1))
    docs = CT.to_texts(traj, centres)
    big = CT.bigrams_of(docs)
    used = np.unique([w for d in docs for s in d for w in s])
    meta = {"object": obj, "vocab": vocab, "n_traj": n_traj, "used_words": int(len(used)),
            "tokens": int(sum(len(s) for d in docs for s in d)), "bigrams": int(len(big)),
            "ambient": int(X.shape[1]), "seconds_text": time.time() - t0, **info}
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta, flush=True)

    embs = {"truth": {0: centres}}
    if svd:
        U, _ = CT.svd_gromov(docs, len(centres), max(dims))
        embs["svd"] = {d: U[:, :d] for d in dims}
    embs["cbow"] = {}
    for d in dims:
        C = CT.cbow(docs, d)
        embs["cbow"][d] = np.vstack([C, np.full((len(centres) - len(C), d), np.nan)]) if len(C) < len(centres) else C
    for emb, by_d in embs.items():
        for d, W in by_d.items():
            for n in (1, 2):
                f = OUT / f"{tag}_{emb}_d{d}_n{n}.json"
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
                try:
                    r = CT.measure(Xc, ladders, max_points, h_iter)
                except ValueError:  # near-duplicate rows (e.g. words with identical contexts) -> tiny jitter
                    sc = float(np.nanstd(Xc)) or 1.0
                    r = CT.measure(Xc + np.random.default_rng(0).normal(0, 1e-6 * sc, Xc.shape), ladders, max_points, h_iter)
                    r["jitter"] = 1e-6
                r.update({"object": obj, "vocab": vocab, "n_traj": n_traj, "embedding": emb, "d": d, "ngram": n,
                          "seconds": time.time() - t1})
                f.write_text(json.dumps(r))
                sw = r["schweinhart"]
                print(f"{tag} {emb} d={d} n={n} N={r['N']} | Schw {sw['min']}-{sw['max']} | TwoNN {r['twonn']:.2f} "
                      f"| FisherS {r['fishers']:.2f} | Hid {np.round(r['hidalgo']['d_k'], 2).tolist()} ({r['seconds']:.0f}s)",
                      flush=True)
    if fnn:
        f = OUT / f"{tag}_fnn.json"
        if not f.exists():
            seqs = sequences(docs)
            res = {}
            for emb, d in (("truth", 0), ("cbow", 15), ("cbow", 30)):
                W = embs[emb][d]
                keep = [s for s in seqs if not np.isnan(W[s]).any()]
                res[f"words_{emb}_d{d}"] = delay_stats([W[s] for s in keep], 8, orig=[centres[s] for s in keep])
            # classical Takens: scalar observable = first coordinate of the trajectory, delay = 1/e autocorrelation lag
            x = traj[:, :, 0]
            tau_ac = int(np.median([autocorr_lag(x[i]) for i in range(min(50, len(x)))]))
            tau = max(1, min(tau_ac, x.shape[1] // 60))  # keep >= 60 delay samples per trajectory
            ser = [x[i, ::tau] for i in range(len(x))]
            res["scalar_delay"] = {"tau": tau, "tau_autocorr": tau_ac,
                                   "stats": delay_stats(ser, 20, orig=[traj[i, ::tau] for i in range(len(x))])}
            f.write_text(json.dumps(res))
            print(tag, "FNN done", flush=True)
    (OUT / f"{tag}_done.json").write_text("{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    T = tasks()
    if a.list:
        for i, t in enumerate(T):
            print(i, t)
        return
    OUT.mkdir(exist_ok=True)
    t = T[a.task]
    if t["part"] == "A":
        run_config(t["obj"], t["vocab"], t["n_traj"], t["dims"], t["svd"], t["fnn"])
    else:
        for g in t["grid"]:
            run_config(t["obj"], g["vocab"], g["n_traj"], [15, 30], False, False)


if __name__ == "__main__":
    main()
