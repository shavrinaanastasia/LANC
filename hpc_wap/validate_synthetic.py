"""High-dimensional validation of the exact-MST Schweinhart estimator on sets of known dimension.

Objects are embedded by a random orthogonal map into the SAME ambient dimensions as the
language clouds (15 for words, 30 for bigrams), so the check covers the real code path.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from schweinhart_mst import DEFAULT_ALPHAS, geometric_ladder, schweinhart, summarise, threads_info

def embed(p, ambient, rng):
    q, _ = np.linalg.qr(rng.normal(size=(ambient, ambient)))
    z = np.zeros((p.shape[0], ambient)); z[:, : p.shape[1]] = p
    return z @ q.T

def sierpinski_carpet(n, rng, depth=12):
    pts = np.zeros((n, 2)); scale = 1.0
    cells = np.array([(i, j) for i in range(3) for j in range(3) if not (i == 1 and j == 1)], float)
    for _ in range(depth):
        scale /= 3; pts += cells[rng.integers(0, 8, n)] * scale
    return pts + rng.random((n, 2)) * scale

def objects(n, rng):
    yield "cube_5d", 5.0, rng.random((n, 5))
    yield "cube_8d", 8.0, rng.random((n, 8))
    s = rng.normal(size=(n, 7)); yield "sphere_S6", 6.0, s / np.linalg.norm(s, axis=1, keepdims=True)
    yield "sierpinski_carpet", float(np.log(8) / np.log(3)), sierpinski_carpet(n, rng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ambient", type=int, nargs="+", default=[15, 30])
    ap.add_argument("--out", default="synthetic_validation.json")
    a = ap.parse_args()
    rng = np.random.default_rng(20260929)
    print(threads_info())
    res = []
    for amb in a.ambient:
        for name, true_d, p in objects(a.n, rng):
            x = embed(p, amb, rng)
            sizes = geometric_ladder(a.n, a.n // 16, 9)
            t = time.time()
            r = schweinhart(x, sizes, DEFAULT_ALPHAS, seed=1, log=lambda *_: None)
            s = summarise(r["fits"])
            f1 = next(f for f in r["fits"] if abs(f["alpha"] - 1.0) < 1e-9)
            row = dict(object=name, ambient=amb, true_d=true_d, d_alpha1=f1["dimension"],
                       ape_alpha1=abs(f1["dimension"] - true_d) / true_d * 100, **s, seconds=time.time() - t)
            print(json.dumps(row)); res.append(row)
    json.dump(res, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
