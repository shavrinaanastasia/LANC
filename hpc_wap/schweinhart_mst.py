"""Exact Euclidean MST + Schweinhart estimator (Gromov, Borodin, Yerbolova 2024 protocol).

Key points
- Exact Euclidean MST by Prim's algorithm on the implicit complete graph:
  O(n^2 d) time, O(n d) memory (no n x n distance matrix, so 300k points fit in ~100 MB).
  Uses numba (parallel) when available, otherwise a vectorised NumPy fallback.
- The Euclidean MST does not depend on alpha (x -> x^alpha is monotone), so ONE tree per
  sample size gives E_alpha = sum |e|^alpha for the whole alpha grid.
- Regression ln E_alpha(l) = ln C + (1 - alpha/d) ln l  ->  d = alpha / (1 - slope).
- Admissibility (paper, gamma = 10%): 95% CI of d and 95% CI of slope (d-alpha)/d must each
  be no wider than gamma of the respective point estimate.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import stats

try:  # numba is optional; HSE anaconda has it, the fallback is exact but slower
    import numba
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------------------
# Exact Euclidean MST (Prim)
# ---------------------------------------------------------------------------------------

def _prim_numpy(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    remaining = np.arange(1, n)
    best = np.sum((x[remaining] - x[0]) ** 2, axis=1)
    out = np.empty(n - 1, dtype=np.float64)
    for k in range(n - 1):
        j = int(np.argmin(best))
        out[k] = best[j]
        cur = remaining[j]
        last = remaining.shape[0] - 1
        remaining[j] = remaining[last]
        best[j] = best[last]
        remaining = remaining[:last]
        best = best[:last]
        if last:
            d2 = np.sum((x[remaining] - x[cur]) ** 2, axis=1)
            np.minimum(best, d2, out=best)
    return np.sqrt(out)


if HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def _prim_numba(x):  # pragma: no cover - compiled
        n, dim = x.shape
        nthreads = numba.get_num_threads()
        remaining = np.arange(1, n)
        best = np.empty(n - 1)
        for i in prange(n - 1):
            s = 0.0
            for t in range(dim):
                diff = x[i + 1, t] - x[0, t]
                s += diff * diff
            best[i] = s
        out = np.empty(n - 1)
        loc_val = np.empty(nthreads)
        loc_idx = np.empty(nthreads, dtype=np.int64)
        m = n - 1
        for k in range(n - 1):
            # parallel argmin over blocks
            block = (m + nthreads - 1) // nthreads
            for b in prange(nthreads):
                lo = b * block
                hi = min(m, lo + block)
                bv = np.inf
                bi = -1
                for i in range(lo, hi):
                    if best[i] < bv:
                        bv = best[i]
                        bi = i
                loc_val[b] = bv
                loc_idx[b] = bi
            j = -1
            bv = np.inf
            for b in range(nthreads):
                if loc_idx[b] >= 0 and (loc_val[b] < bv or (loc_val[b] == bv and loc_idx[b] < j)):
                    bv = loc_val[b]
                    j = loc_idx[b]
            out[k] = bv
            cur = remaining[j]
            m -= 1
            remaining[j] = remaining[m]
            best[j] = best[m]
            if m == 0:
                break
            for i in prange(m):
                p = remaining[i]
                s = 0.0
                for t in range(dim):
                    diff = x[p, t] - x[cur, t]
                    s += diff * diff
                if s < best[i]:
                    best[i] = s
        return np.sqrt(out)


def _prim_torch(x: np.ndarray) -> np.ndarray:
    """Prim on a CUDA GPU in float64 (for very high ambient dimension, e.g. 768/1536).
    Squared distances via |a|^2 + |b|^2 - 2ab, one matrix-vector product per step."""
    import torch

    dev = torch.device("cuda")
    X = torch.from_numpy(x).to(dev, torch.float64)
    sq = (X * X).sum(1)
    n = X.shape[0]
    best = torch.full((n,), float("inf"), dtype=torch.float64, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    out = torch.empty(n - 1, dtype=torch.float64, device=dev)
    cur = torch.tensor(0, device=dev)
    for k in range(n - 1):
        done[cur] = True
        d2 = (sq + sq[cur] - 2.0 * (X @ X[cur])).clamp_min_(0.0)
        torch.minimum(best, d2, out=best)
        best.masked_fill_(done, float("inf"))
        cur = torch.argmin(best)
        out[k] = best[cur]
    return np.sqrt(out.cpu().numpy())


def mst_edge_lengths(points: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Return the n-1 edge lengths of the exact Euclidean MST of `points`."""
    x = np.ascontiguousarray(points, dtype=np.float64)
    if x.shape[0] < 2:
        return np.empty(0)
    if backend == "auto":
        backend = "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba":
        return _prim_numba(x)
    if backend == "torch":
        return _prim_torch(x)
    return _prim_numpy(x)


# ---------------------------------------------------------------------------------------
# Schweinhart estimator
# ---------------------------------------------------------------------------------------

DEFAULT_ALPHAS = np.round(np.concatenate([[1e-4], np.arange(0.1, 10.0 + 1e-9, 0.1)]), 4)


def energies_from_edges(edges: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """E_alpha = sum |e|^alpha, with 0^alpha = 0 (duplicate points contribute nothing)."""
    pos = edges[edges > 0]
    if pos.size == 0:
        return np.zeros(len(alphas))
    logs = np.log(pos)
    # sum exp(alpha*log e) computed stably in float64
    return np.array([np.exp(a * logs).sum() for a in alphas])


def geometric_ladder(n_total: int, n_min: int, steps: int) -> list[int]:
    n_min = max(2, min(n_min, n_total))
    sizes = np.unique(np.round(np.geomspace(n_min, n_total, steps)).astype(int))
    return [int(s) for s in sizes]


def fit_alpha(sizes: np.ndarray, energy: np.ndarray, alpha: float, gamma: float = 0.10) -> dict:
    x = np.log(sizes.astype(float))
    y = np.log(energy)
    fit = stats.linregress(x, y)
    k = len(x)
    tq = stats.t.ppf(0.975, k - 2) if k > 2 else float("nan")
    slope, se = float(fit.slope), float(fit.stderr)
    s_lo, s_hi = slope - tq * se, slope + tq * se
    d = alpha / (1.0 - slope) if slope < 1 else float("nan")
    if s_hi < 1:
        d_lo, d_hi = alpha / (1.0 - s_lo), alpha / (1.0 - s_hi)
    else:
        d_lo, d_hi = float("nan"), float("inf")
    d_rel = (d_hi - d_lo) / d if np.isfinite(d) and np.isfinite(d_hi) and d > 0 else float("inf")
    s_rel = (s_hi - s_lo) / abs(slope) if slope != 0 else float("inf")
    admissible = bool(np.isfinite(d) and d > 0 and d_rel <= gamma and s_rel <= gamma)
    return {
        "alpha": float(alpha),
        "slope": slope,
        "slope_ci95": [s_lo, s_hi],
        "dimension": d,
        "dimension_ci95": [d_lo, d_hi],
        "dimension_ci_rel_width": d_rel,
        "slope_ci_rel_width": s_rel,
        "r_squared": float(fit.rvalue ** 2),
        "admissible": admissible,
    }


def schweinhart(
    points: np.ndarray,
    sizes: list[int],
    alphas: np.ndarray = DEFAULT_ALPHAS,
    seed: int = 0,
    gamma: float = 0.10,
    backend: str = "auto",
    log=print,
) -> dict:
    """One nested random ladder: MST at each size, then regression for every alpha."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(points.shape[0])
    energies = []
    extra = []
    for n in sizes:
        t0 = _now()
        edges = mst_edge_lengths(points[perm[:n]], backend)
        energies.append(energies_from_edges(edges, alphas))
        extra.append({"n": n, "zero_edges": int((edges == 0).sum()), "seconds": _now() - t0})
        log(f"    n={n:>8d}  MST {extra[-1]['seconds']:.1f}s  zero-length edges={extra[-1]['zero_edges']}")
    energies = np.array(energies)  # (len(sizes), len(alphas))
    sz = np.array(sizes)
    fits = [fit_alpha(sz, energies[:, i], a, gamma) for i, a in enumerate(alphas)]
    return {"seed": seed, "sizes": sizes, "mst": extra, "energies": energies.tolist(), "fits": fits}


def summarise(fits: list[dict]) -> dict:
    ok = [f for f in fits if f["admissible"]]
    if not ok:
        return {"n_admissible": 0, "min_d": None, "max_d": None, "alpha_range": None}
    ds = [f["dimension"] for f in ok]
    al = [f["alpha"] for f in ok]
    return {
        "n_admissible": len(ok),
        "min_d": float(min(ds)),
        "max_d": float(max(ds)),
        "median_d": float(np.median(ds)),
        "alpha_range": [float(min(al)), float(max(al))],
    }


def _now() -> float:
    import time

    return time.perf_counter()


def threads_info() -> str:
    if HAVE_NUMBA:
        return f"numba {numba.__version__}, threads={numba.get_num_threads()}"
    return f"numpy fallback (no numba), OMP={os.environ.get('OMP_NUM_THREADS')}"


if __name__ == "__main__":  # quick self-check against scipy dense MST
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.default_rng(1)
    for dim in (2, 10, 30):
        p = rng.normal(size=(1500, dim))
        ref = minimum_spanning_tree(squareform(pdist(p))).data.sum()
        backs = ["numpy"] + (["numba"] if HAVE_NUMBA else [])
        try:
            import torch

            if torch.cuda.is_available():
                backs.append("torch")
        except Exception:
            pass
        for be in backs:
            got = mst_edge_lengths(p, be).sum()
            print(f"dim={dim} {be}: {got:.10f} vs scipy {ref:.10f}  rel.err={abs(got-ref)/ref:.2e}")
            assert abs(got - ref) / ref < 1e-9, "MST mismatch"
    print(threads_info())
