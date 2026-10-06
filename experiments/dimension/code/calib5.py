"""E14: remove CBOW noise instead of calibrating it away.

Idea: estimators are exact on true coordinates, so the CBOW over-estimate comes from the embedding -- most likely from noise
in the vectors (rare words). The noise can be measured on the text itself: retrain CBOW R times on bootstrap resamples of the
documents, align the runs (orthogonal Procrustes), and take the per-word scatter sigma_w. Then estimate dimension
  * on the usual single run (baseline, as in all previous calibrations),
  * on the run-averaged vectors (noise / sqrt(R)),
  * on frequent words only (count >= 5 / 20 / 50),
  * at scales above the noise: TwoNN on sub-samples whose nearest-neighbour distance exceeds c * sigma, and
    correlation-integral slopes at radii r = c * sigma (curves stored; the rule is chosen in the analysis).
Validation: a correction counts only if it gives ~ m on every calibration family AND on the held-out generator
(pseudolanguage, E7), where the E6-E12 correction failed by 34-56 %.

Cases: Brownian walks on tori / spheres (base, Zipf kappa 3/5, V = 18000, step x2), pseudolanguage eps / dist walks,
War and Peace. CBOW d = 15 and 30, R = 5 runs (run 0 = full data, seed 1 = the setting of all earlier experiments).

  python calib5.py --list ; python calib5.py --task i
"""
from __future__ import annotations

import argparse, gzip, json, time
from pathlib import Path
import numpy as np

import calib_trajectories as CT
import id_methods as M

OUT = Path("calib5_results")
R = 5
DIMS = (15, 30)
CASES = (
    [("walk", o, dict()) for o in ("torus2", "torus4", "torus6", "torus8", "sphere4", "sphere8")]
    + [("walk", o, dict(kappa=3.0)) for o in ("torus4", "torus6", "sphere4")]
    + [("walk", "torus4", dict(kappa=5.0)), ("walk", "torus4", dict(vocab=18000)), ("walk", "sphere8", dict(vocab=18000)),
       ("walk", "torus4", dict(step=2.0)), ("walk", "torus6", dict(step=2.0))]
    + [("pseudo", f"cube{m}", dict(walk="eps")) for m in (2, 3, 4, 6)]
    + [("pseudo", f"cube{m}", dict(walk="dist")) for m in (2, 4)]
    + [("wap", "war_and_peace", dict())]
)


def name(c):
    kind, obj, kw = c
    return obj + "".join(f"_{k}{v}" for k, v in sorted(kw.items()))


# ------------------------------------------------------------------------------------------------- texts
def make_walk(obj, step=1.0, kappa=0.0, vocab=5000, sent=20, n_traj=500, length=1500):
    from calib3 import partition_weighted
    m = int(obj.replace("torus", "").replace("sphere", ""))
    rng = np.random.default_rng(sum(map(ord, obj)))
    sigma = step * (np.pi if obj.startswith("torus") else 1.0) * vocab ** (-1 / m)
    traj = CT.trajectories(obj, n_traj, length, rng, sigma)
    X = traj.reshape(-1, traj.shape[-1])
    centres = partition_weighted(X, vocab, kappa, np.random.default_rng(1))
    docs = CT.to_texts(traj, centres, sent_len=sent)
    return docs, centres, m, len(centres)


def make_pseudo(obj, walk="eps", n_texts=1000):
    import pseudolang as PL
    m = int(obj[4:])
    data = PL.make_points(m, 5000, 4); V = len(data)
    eps = 1.5 * (1 / V) ** (1 / m)
    rng = np.random.default_rng(m)
    lens, _, _ = PL.text_lengths("russian_Prussner_wolframe100.csv", V, rng)
    texts = [PL.walk(data, int(rng.choice(lens)), walk, np.random.default_rng(i * 8 + 2), eps) for i in range(n_texts)]
    return [[t] for t in texts], data, m, V


def make_wap():
    z = np.load("wap_embeddings_cbow.npz", allow_pickle=True)
    vocab = [str(v) for v in z["vocabulary"]]; idx = {w: i for i, w in enumerate(vocab)}
    sents = json.load(gzip.open("wap_sentences.json.gz", "rt", encoding="utf-8"))
    ids = [[idx[w] for w in s if w in idx] for s in sents]
    ids = [s for s in ids if s]
    docs = [ids[i:i + 50] for i in range(0, len(ids), 50)]  # blocks of 50 sentences = bootstrap units
    return docs, None, None, len(vocab)


# ------------------------------------------------------------------------------------------------- CBOW runs
def cbow(docs, d, V, seed):
    from gensim.models import Word2Vec
    sents = [[str(w) for w in s] for doc in docs for s in doc]
    m = Word2Vec(sents, vector_size=d, sg=0, window=5, negative=5, min_count=1, epochs=30, seed=seed, workers=1)
    out = np.full((V, d), np.nan)
    for w in m.wv.index_to_key:
        out[int(w)] = m.wv[w]
    return out


def runs_aligned(docs, d, V):
    from scipy.linalg import orthogonal_procrustes
    rng = np.random.default_rng(7)
    E = [cbow(docs, d, V, seed=1)]
    for r in range(1, R):
        boot = [docs[i] for i in rng.integers(0, len(docs), len(docs))]
        E.append(cbow(boot, d, V, seed=1 + r))
    ref = E[0]; mu = np.nanmean(ref, 0)
    al = [ref - mu]
    for X in E[1:]:
        ok = ~np.isnan(X).any(1) & ~np.isnan(ref).any(1)
        Xc = X - np.nanmean(X[ok], 0)
        Q, _ = orthogonal_procrustes(Xc[ok], al[0][ok])
        al.append(Xc @ Q)
    A = np.stack(al)                                  # (R, V, d)
    mean = np.nanmean(A, 0)
    nobs = (~np.isnan(A).any(2)).sum(0)
    sig = np.sqrt(np.nanmean(((A - mean) ** 2).sum(2), 0))  # per-word RMS scatter around the mean
    sig[nobs < 3] = np.nan
    return al[0], mean, sig, nobs


# ------------------------------------------------------------------------------------------------- measurements
def safe_measure(X):
    X = X[~np.isnan(X).any(1)]
    try:
        r = CT.measure(X, 2, 10000, 5000)
    except ValueError:
        r = CT.measure(X + np.random.default_rng(0).normal(0, 1e-6 * (float(np.std(X)) or 1), X.shape), 2, 10000, 5000)
    return {"N": r["N"], "twonn": r["twonn"], "schw": r["schweinhart"]["median"], "schw_min": r["schweinhart"]["min"],
            "schw_max": r["schweinhart"]["max"], "hid": r["hidalgo"], "fishers": r["fishers"]}


def scale_curve(X, sigma, rng):
    """TwoNN and median NN distance on geometric sub-samples; correlation slopes at radii c * sigma."""
    from scipy.spatial.distance import pdist
    X = np.unique(X[~np.isnan(X).any(1)], axis=0)
    out = {"sigma": float(sigma), "twonn_by_n": [], "corr": []}
    n = len(X)
    for size in sorted({int(s) for s in np.geomspace(300, min(n, 20000), 8)}):
        S = X[rng.permutation(n)[:size]]
        dist, _ = M.knn(S, 2); ok = dist[:, 0] > 0
        out["twonn_by_n"].append({"n": size, "r_nn": float(np.median(dist[ok, 0])), "twonn": M.twonn_from_mu(dist[ok, 1] / dist[ok, 0])})
    S = X[rng.permutation(n)[:3000]]
    d = np.sort(pdist(S)); d = d[d > 0]
    radii = sigma * np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    C = np.searchsorted(d, radii) / len(d)
    for i in range(len(radii) - 1):
        if C[i] > 0 and C[i + 1] > C[i]:
            out["corr"].append({"c": float(radii[i] / sigma), "C": float(C[i]), "slope": float(np.log(C[i + 1] / C[i]) / np.log(radii[i + 1] / radii[i]))})
    return out


def run_case(c):
    kind, obj, kw = c
    fn = OUT / f"{name(c)}.json"
    if fn.exists():
        return
    t0 = time.time()
    docs, truth, m, V = make_walk(obj, **kw) if kind == "walk" else make_pseudo(obj, **kw) if kind == "pseudo" else make_wap()
    words = np.concatenate([np.asarray(s) for doc in docs for s in doc])
    count = np.bincount(words, minlength=V)
    big = CT.bigrams_of(docs)
    res = {"case": name(c), "kind": kind, "object": obj, "m": m, "V": int(V), "tokens": int(len(words)), "used": int((count > 0).sum()),
           "bigrams": int(len(big)), **kw, "emb": {}}
    print(res, flush=True)
    rng = np.random.default_rng(0)
    if truth is not None:
        res["truth"] = {"words": safe_measure(truth[count > 0]), "bigrams": safe_measure(np.hstack([truth[big[:, 0]], truth[big[:, 1]]]))}
    for d in DIMS:
        ref, mean, sig, nobs = runs_aligned(docs, d, V)
        used = count > 0
        r_nn_words = M.knn(ref[used & ~np.isnan(ref).any(1)], 1)[0][:, 0]
        e = {"sigma_med": float(np.nanmedian(sig[used])), "r_nn_med": float(np.median(r_nn_words)),
             "noise_to_nn": float(np.nanmedian(sig[used]) / np.median(r_nn_words)),
             "sigma_by_count": {f"{lo}-{hi}": float(np.nanmedian(sig[(count >= lo) & (count < hi)])) if ((count >= lo) & (count < hi)).any() else None
                                for lo, hi in ((1, 5), (5, 20), (20, 50), (50, 200), (200, 10 ** 9))},
             "est": {}, "scale": {}}
        for vname, W in (("single", ref), ("mean", mean)):
            for k in (1, 5, 20, 50):
                keep = count >= k
                if keep.sum() < 300:
                    continue
                e["est"][f"{vname}_words_k{k}"] = safe_measure(W[keep])
                b = big[keep[big[:, 0]] & keep[big[:, 1]]]
                if len(b) >= 300:
                    e["est"][f"{vname}_bigrams_k{k}"] = safe_measure(np.hstack([W[b[:, 0]], W[b[:, 1]]]))
                print(name(c), d, vname, k, {kk: round(v["twonn"], 2) for kk, v in e["est"].items() if kk.startswith(vname) and kk.endswith(f"k{k}")}, flush=True)
            s = float(np.nanmedian(sig[used]))
            e["scale"][f"{vname}_words"] = scale_curve(W[used], s, rng)
            e["scale"][f"{vname}_bigrams"] = scale_curve(np.hstack([W[big[:, 0]], W[big[:, 1]]]), s * np.sqrt(2), rng)
        res["emb"][str(d)] = e
        print(name(c), d, "sigma", e["sigma_med"], "noise/nn", e["noise_to_nn"], f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = time.time() - t0
    fn.write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.list:
        for i, c in enumerate(CASES): print(i, name(c))
        return
    OUT.mkdir(exist_ok=True)
    run_case(CASES[a.task])


if __name__ == "__main__":
    main()
