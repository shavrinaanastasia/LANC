"""Three more intrinsic-dimension estimators, checked against reference code.

TwoNN   Facco, d'Errico, Rodriguez, Laio (2017) Sci. Rep. 7:12140.
        mu_i = r2/r1 ~ Pareto(d); d = slope of -log(1-F(mu)) vs log(mu) through the origin,
        largest 10% of mu discarded (same as scikit-dimension TwoNN). Plus the decimation
        analysis of the paper: estimate on disjoint blocks of N, N/2, ..., N/32 points.
FisherS Albergante, Bac, Zinovyev (2019) arXiv:1901.06328; Bac et al. (2021) Entropy 23:1368.
        centre -> PCA (keep PCs with eigenvalue > max/10) -> whiten -> project on unit sphere;
        p_alpha = mean fraction of points y with <x,y> >= alpha <x,x>;
        n_alpha = W(-ln(1-a^2) / (2 pi p^2 a^2 (1-a^2))) / (-ln(1-a^2));
        single estimate at alpha = 0.9 * max alpha with finite n_alpha (scikit-dimension rule).
Hidalgo Allegra, Facco, Denti, Laio, Mira (2020) Sci. Rep. 10:16193.
        Bayesian mixture of K Pareto components for mu with a Potts-type neighbourhood prior
        (q nearest neighbours share the component with probability zeta). Gibbs sampler ported
        line by line from the authors' gibbs.c (github.com/micheleallegra/Hidalgo):
        K=2, q=3, zeta=0.8 fixed, priors a=b=c=f=1, Niter=10000, burn-in 90%, sample every 10.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import lambertw

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    prange = range


# --------------------------------------------------------------------------- neighbours
def knn(X: np.ndarray, k: int):
    """Exact k nearest neighbours (excluding self): distances, indices."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, D = X.shape
    if D <= 10:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1).fit(X)
        d, i = nn.kneighbors(X)
        return d[:, 1:], i[:, 1:]
    sq = (X * X).sum(1)
    dist = np.empty((n, k))
    idx = np.empty((n, k), dtype=np.int64)
    step = max(1, int(2e8 // (n * 8)))
    for s in range(0, n, step):
        e = min(n, s + step)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (X[s:e] @ X.T)
        d2[np.arange(e - s), np.arange(s, e)] = np.inf
        part = np.argpartition(d2, k, axis=1)[:, :k]
        pv = np.take_along_axis(d2, part, 1)
        o = np.argsort(pv, axis=1)
        idx[s:e] = np.take_along_axis(part, o, 1)
        dist[s:e] = np.sqrt(np.maximum(np.take_along_axis(pv, o, 1), 0))
    return dist, idx


# --------------------------------------------------------------------------- TwoNN
def twonn_from_mu(mu: np.ndarray, discard: float = 0.1) -> float:
    N = len(mu)
    m = np.sort(mu)[: int(N * (1 - discard))]
    F = np.arange(1, len(m) + 1) / N
    x, y = np.log(m), -np.log(1 - F)
    return float((x @ y) / (x @ x))


def twonn(X: np.ndarray, discard: float = 0.1, decimation=(1, 2, 4, 8, 16, 32), seed: int = 0) -> dict:
    d, _ = knn(X, 2)
    ok = d[:, 0] > 0
    mu = d[ok, 1] / d[ok, 0]
    full = twonn_from_mu(mu, discard)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    curve = []
    for b in decimation:
        if b == 1:
            curve.append({"blocks": 1, "n": len(X), "mean": full, "std": 0.0})
            continue
        size = len(X) // b
        if size < 50:
            break
        ests = []
        for j in range(b):
            dj, _ = knn(X[perm[j * size:(j + 1) * size]], 2)
            okj = dj[:, 0] > 0
            ests.append(twonn_from_mu(dj[okj, 1] / dj[okj, 0], discard))
        curve.append({"blocks": b, "n": size, "mean": float(np.mean(ests)), "std": float(np.std(ests))})
    return {"d": full, "n_used": int(ok.sum()), "decimation": curve}


# --------------------------------------------------------------------------- FisherS
@njit(parallel=True, cache=True)
def _count_ge(xy, alphas, counts):  # counts[i, a] += #{j : xy[i, j] >= alphas[a]}
    n, m = xy.shape
    A = alphas.shape[0]
    for i in prange(n):
        for j in range(m):
            v = xy[i, j]
            for a in range(A):
                if v >= alphas[a]:
                    counts[i, a] += 1
                else:
                    break


def fishers(X: np.ndarray, conditional_number: float = 10.0, alphas=None, chunk: int = 1000) -> dict:
    if alphas is None:
        alphas = np.arange(0.6, 1, 0.02)
    alphas = np.asarray(alphas, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    # PCA via SVD, keep eigenvalues > max / conditional_number, whiten, project on sphere
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2 / (len(X) - 1)
    keep = np.where(ev / ev[0] > 1.0 / conditional_number)[0]
    Z = U[:, keep] * S[keep]
    Z = Z / Z.std(0, ddof=1)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    n = len(Z)
    counts = np.zeros((n, len(alphas)), dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        xy = Z[s:e] @ Z.T                     # rows are unit vectors, so <x,x> = 1
        xy[np.arange(e - s), np.arange(s, e)] = -np.inf   # exclude self
        _count_ge(xy, alphas, counts[s:e])
    p = (counts / n).mean(0)
    n_alpha = np.full(len(alphas), np.nan)
    for i, (a, pa) in enumerate(zip(alphas, p)):
        if pa > 0:
            w = np.log(1 - a * a)
            val = np.real(lambertw(-(w / (2 * np.pi * pa * pa * a * a * (1 - a * a))))) / (-w)
            n_alpha[i] = val if np.isfinite(val) else np.nan
    good = np.where(~np.isnan(n_alpha))[0]
    if len(good) == 0:
        return {"d": None, "pcs_kept": int(len(keep)), "n_alpha": n_alpha.tolist(), "alphas": alphas.tolist()}
    a_ref = alphas[good].max() * 0.9
    k = good[np.argmin(np.abs(alphas[good] - a_ref))]
    return {"d": float(n_alpha[k]), "alpha": float(alphas[k]), "pcs_kept": int(len(keep)),
            "n_alpha": n_alpha.tolist(), "p_alpha": p.tolist(), "alphas": alphas.tolist()}


# --------------------------------------------------------------------------- Hidalgo
@njit(cache=True)
def _binom(N, q):
    s = 1.0
    for q1 in range(q):
        s = s * (N - q1) / (q1 + 1)
    return s


@njit(cache=True)
def _zpart(N, N1, zeta, q):
    s = 0.0
    for q1 in range(q + 1):
        s += _binom(N1 - 1, q1) * _binom(N - N1, q - q1) * zeta ** q1 * (1 - zeta) ** (q - q1)
    return s


@njit(cache=True)
def _gibbs(mu, Iin, Iout, Iout_count, Iout_track, K, q, zeta, Niter, burn_in, sampling_rate, seed):
    np.random.seed(seed)
    N = mu.shape[0]
    a = np.ones(K); b = np.ones(K); c = np.ones(K)
    d = np.ones(K); V = np.zeros(K); NN = np.zeros(K, dtype=np.int64)
    p = np.ones(K) / K
    pp = (K - 1.0) / K
    Z = np.empty(N, dtype=np.int64)
    logmu = np.log(mu)
    for i in range(N):
        Z[i] = np.random.randint(K)
        V[Z[i]] += logmu[i]
        NN[Z[i]] += 1
    a1 = a + NN; b1 = b + V; c1 = c + NN
    nsamp = 0
    start = int(math.ceil(Niter * burn_in))
    tot = (Niter - start + sampling_rate - 1) // sampling_rate
    out_d = np.zeros((tot, K)); out_p = np.zeros((tot, K)); out_lik = np.zeros(tot)
    zcount = np.zeros((N, K))
    for it in range(Niter):
        # d_k ~ Gamma(a1, b1) by rejection on [0, 200] (as in gibbs.c)
        for k in range(K):
            while True:
                r1 = np.random.random() * 200.0
                r2 = np.random.random()
                if a1[k] - 1 > 0:
                    rmax = (a1[k] - 1) / b1[k]
                    frac = math.exp(-b1[k] * (r1 - rmax) - (a1[k] - 1) * (math.log(rmax) - math.log(r1)))
                else:
                    frac = math.exp(-b1[k] * r1)
                if frac > r2:
                    d[k] = r1
                    break
        # p (stick-by-stick Beta, as in gibbs.c)
        for k in range(K - 1):
            while True:
                r1 = np.random.random()
                r2 = np.random.random()
                rmax = (c1[k] - 1) / (c1[k] - 1 + c1[K - 1] - 1)
                frac = (r1 / rmax) ** (c1[k] - 1) * ((1 - r1) / (1 - rmax)) ** (c1[K - 1] - 1)
                if frac > r2:
                    r1 = r1 * (1.0 - pp + p[k])
                    p[K - 1] += p[k] - r1
                    pp -= p[k] - r1
                    p[k] = r1
                    break
        p[K - 1] = 1 - pp
        # Z with Potts neighbourhood term
        prob = np.empty(K); gg = np.empty(K)
        for i in range(N):
            gmax = -1e300
            for k1 in range(K):
                n_in = 0.0
                for j in range(q):
                    if Z[Iin[i, j]] == k1:
                        n_in += 1.0
                m_in = 0.0
                for j in range(Iout_count[i]):
                    if Z[Iout[Iout_track[i] + j]] == k1:
                        m_in += 1.0
                zp = _zpart(N, NN[k1], zeta, q)
                g = (n_in + m_in) * math.log(zeta / (1 - zeta)) - math.log(zp)
                g += math.log(_zpart(N, NN[k1] - 1, zeta, q) / zp) * (NN[k1] - 1)
                gg[k1] = g
                if g > gmax:
                    gmax = g
            norm = 0.0
            for k1 in range(K):
                prob[k1] = p[k1] * d[k1] * math.exp(-(d[k1] + 1) * logmu[i] + gg[k1] - gmax)
                norm += prob[k1]
            for k1 in range(K):
                prob[k1] /= norm
            while True:
                r1 = np.random.randint(K)
                if prob[r1] > np.random.random():
                    old = Z[i]
                    NN[old] -= 1; a1[old] -= 1; c1[old] -= 1; V[old] -= logmu[i]; b1[old] -= logmu[i]
                    Z[i] = r1
                    NN[r1] += 1; a1[r1] += 1; c1[r1] += 1; V[r1] += logmu[i]; b1[r1] += logmu[i]
                    break
        if it >= start and (it - start) % sampling_rate == 0 and nsamp < tot:
            lik = 0.0
            for i in range(N):
                lik += math.log(p[Z[i]]) + math.log(d[Z[i]]) - (d[Z[i]] + 1) * logmu[i]
            out_d[nsamp] = d; out_p[nsamp] = p; out_lik[nsamp] = lik
            for i in range(N):
                zcount[i, Z[i]] += 1
            nsamp += 1
    return out_d[:nsamp], out_p[:nsamp], out_lik[:nsamp], zcount / max(nsamp, 1)


def hidalgo(X: np.ndarray, K: int = 2, q: int = 3, zeta: float = 0.8, Niter: int = 10000,
            burn_in: float = 0.9, seed: int = 0, replicas: int = 1) -> dict:
    dist, ind = knn(X, max(q, 2))
    ok = dist[:, 0] > 0
    if not ok.all():
        raise ValueError("duplicate points: pass distinct rows to Hidalgo")
    mu = dist[:, 1] / dist[:, 0]
    N = len(X)
    Iin = np.ascontiguousarray(ind[:, :q], dtype=np.int64)
    # reverse neighbour lists (points for which i is a neighbour)
    src = np.repeat(np.arange(N), q)
    dst = Iin.ravel()
    order = np.argsort(dst, kind="stable")
    Iout = src[order].astype(np.int64)
    Iout_count = np.bincount(dst, minlength=N).astype(np.int64)
    Iout_track = np.concatenate([[0], np.cumsum(Iout_count)[:-1]]).astype(np.int64)
    best = None
    for r in range(replicas):
        dd, pp, lik, pz = _gibbs(mu, Iin, Iout, Iout_count, Iout_track, K, q, zeta, Niter, burn_in, 10,
                                 seed + 10000 * r)
        if best is None or lik.mean() > best[2].mean():
            best = (dd, pp, lik, pz)
    dd, pp, lik, pz = best
    # order components by dimension for readability
    o = np.argsort(dd.mean(0))
    d_mean, d_std, p_mean = dd.mean(0)[o], dd.std(0)[o], pp.mean(0)[o]
    z = np.argmax(pz[:, o], axis=1)
    confident = pz.max(1) >= 0.8
    return {"d_k": d_mean.tolist(), "d_k_std": d_std.tolist(), "p_k": p_mean.tolist(),
            "d_weighted": float((d_mean * p_mean).sum()),
            "frac_confident": float(confident.mean()),
            "cluster_sizes": np.bincount(z[confident], minlength=K).tolist(), "N": N}
