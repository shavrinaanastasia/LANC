"""E16: how much hidden state does a text carry beyond the current word? Put War and Peace on the scale of E15.

E15: CBOW over-estimates m by 1.1-1.6x when the next word depends only on the current word (walk on words = Markov chain on
words) and by 1.5-2.5x when the walk has a hidden continuous position (next word depends on more than the current word).
Two text-only measurements, identical for calibration texts and War and Peace:

 1. Predictive gain G = H(bigram model) - H(trigram model) on held-out documents (bits per word; interpolated absolute
    discounting, D = 0.75). G > 0 means the previous-but-one word helps to predict the next one.
 2. Markov surrogate: a text with the same sentence lengths sampled from the text's own empirical bigram transitions
    (no hidden state by construction, same vocabulary / frequencies / bigram statistics). Report G_real - G_surrogate
    (sparsity-matched excess) and the CBOW dimension on the real text and on its surrogate: the ratio real / surrogate is
    the part of the CBOW estimate produced by hidden state.
Validation on E15 generators (sentences of 20, m = 2/4/6, torus / cube, brown / eps): eps texts should give excess ~ 0 and
ratio ~ 1; brown texts excess > 0 and ratio ~ (brown / eps) of E15.

  python calib7.py --list ; python calib7.py --task i
"""
from __future__ import annotations

import argparse, gzip, json, math, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

import calib6 as C6

OUT = Path("calib7_results")
CASES = [dict(m=m, manifold=a, walk=w, text="sent20") for m in (2, 4, 6) for a in ("torus", "cube") for w in ("brown", "eps")] + [dict(wap=True)]


def name(c):
    return "war_and_peace" if c.get("wap") else C6.name(c)


def wap_docs():
    z = np.load("wap_embeddings_cbow.npz", allow_pickle=True)
    vocab = [str(v) for v in z["vocabulary"]]; idx = {w: i for i, w in enumerate(vocab)}
    sents = json.load(gzip.open("wap_sentences.json.gz", "rt", encoding="utf-8"))
    ids = [[idx[w] for w in s if w in idx] for s in sents]
    ids = [s for s in ids if len(s) > 1]
    return [ids[i:i + 50] for i in range(0, len(ids), 50)], len(vocab)


# ---------------------------------------------------------------------------------------------- n-gram models
def counts(sents):
    c1, c2, c3 = Counter(), Counter(), Counter()
    for s in sents:
        c1.update(s); c2.update(zip(s, s[1:])); c3.update(zip(s, s[1:], s[2:]))
    ctx2, typ2, ctx3, typ3 = Counter(), Counter(), Counter(), Counter()
    for (v, w), n in c2.items():
        ctx2[v] += n; typ2[v] += 1
    for (u, v, w), n in c3.items():
        ctx3[(u, v)] += n; typ3[(u, v)] += 1
    return c1, c2, c3, ctx2, typ2, ctx3, typ3


def cross_entropy(train, test, V, D=0.75):
    c1, c2, c3, ctx2, typ2, ctx3, typ3 = counts(train)
    N = sum(c1.values())
    p1 = lambda w: (c1[w] + 1) / (N + V)
    def p2(v, w):
        n = ctx2.get(v, 0)
        if n == 0:
            return p1(w)
        return max(c2.get((v, w), 0) - D, 0) / n + D * typ2[v] / n * p1(w)
    def p3(u, v, w):
        n = ctx3.get((u, v), 0)
        if n == 0:
            return p2(v, w)
        return max(c3.get((u, v, w), 0) - D, 0) / n + D * typ3[(u, v)] / n * p2(v, w)
    h2 = h3 = 0.0; k = 0
    for s in test:
        for u, v, w in zip(s, s[1:], s[2:]):
            h2 -= math.log2(p2(v, w)); h3 -= math.log2(p3(u, v, w)); k += 1
    return h2 / k, h3 / k, k


def gain(docs, V, rng):
    order = rng.permutation(len(docs)); cut = int(0.8 * len(docs))
    tr = [s for i in order[:cut] for s in docs[i]]; te = [s for i in order[cut:] for s in docs[i]]
    h2, h3, k = cross_entropy(tr, te, V)
    return {"H2": h2, "H3": h3, "G": h2 - h3, "test_trigrams": k}


def markov_surrogate(docs, V, rng):
    sents = [s for d in docs for s in d]
    first = np.bincount([s[0] for s in sents], minlength=V).astype(float)
    nxt = defaultdict(Counter)
    for s in sents:
        for v, w in zip(s, s[1:]):
            nxt[v][w] += 1
    uni = np.bincount(np.concatenate([np.asarray(s) for s in sents]), minlength=V).astype(float)
    tab = {}
    for v, cnt in nxt.items():
        ws = np.fromiter(cnt.keys(), int); ps = np.fromiter(cnt.values(), float)
        tab[v] = (ws, np.cumsum(ps) / ps.sum())
    cu, cf = np.cumsum(uni) / uni.sum(), np.cumsum(first) / first.sum()
    out = []
    for d in docs:
        nd = []
        for s in d:
            w = int(np.searchsorted(cf, rng.random())); t = [w]
            for _ in range(len(s) - 1):
                if w in tab:
                    ws, cp = tab[w]; w = int(ws[np.searchsorted(cp, rng.random())])
                else:
                    w = int(np.searchsorted(cu, rng.random()))
                t.append(w)
            nd.append(t)
        out.append(nd)
    return out


def cbow_dims(docs, V, d=15):
    import calib_trajectories as CT
    words = np.concatenate([np.asarray(s) for dd in docs for s in dd]); used = np.bincount(words, minlength=V) > 0
    big = CT.bigrams_of(docs)
    W = C6.cbow(docs, d) if V == C6.V else cbow_any(docs, d, V)
    return {"words": C6.measure(W[used]), "bigrams": C6.measure(np.hstack([W[big[:, 0]], W[big[:, 1]]]))}


def cbow_any(docs, d, V):
    from gensim.models import Word2Vec
    sents = [[str(w) for w in s] for doc in docs for s in doc]
    m = Word2Vec(sents, vector_size=d, sg=0, window=5, negative=5, min_count=1, epochs=30, seed=1, workers=1)
    out = np.full((V, d), np.nan)
    for w in m.wv.index_to_key:
        out[int(w)] = m.wv[w]
    return out


def run(c):
    fn = OUT / f"{name(c)}.json"
    if fn.exists():
        return
    t0 = time.time()
    if c.get("wap"):
        docs, V = wap_docs()
    else:
        rng = np.random.default_rng(c["m"] * 100 + len(c["manifold"]) * 10 + len(c["walk"]) + len(c["text"]))  # = calib6
        intr, _ = C6.word_points(c, rng); docs = C6.make_docs(c, intr, rng); V = C6.V
    res = {**c, "case": name(c), "V": V, "tokens": int(sum(len(s) for d in docs for s in d))}
    res["real"] = gain(docs, V, np.random.default_rng(0))
    sur = markov_surrogate(docs, V, np.random.default_rng(1))
    res["surrogate"] = gain(sur, V, np.random.default_rng(0))
    res["excess_G"] = res["real"]["G"] - res["surrogate"]["G"]
    print(name(c), "G real %.3f surrogate %.3f excess %.3f" % (res["real"]["G"], res["surrogate"]["G"], res["excess_G"]), f"{time.time() - t0:.0f}s", flush=True)
    res["cbow_real"] = cbow_dims(docs, V); res["cbow_surrogate"] = cbow_dims(sur, V)
    for k in ("words", "bigrams"):
        print(name(c), k, "TwoNN real %.2f surrogate %.2f" % (res["cbow_real"][k]["twonn"], res["cbow_surrogate"][k]["twonn"]), flush=True)
    res["seconds"] = time.time() - t0
    fn.write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.list:
        for i, c in enumerate(CASES): print(i, name(c))
        return
    OUT.mkdir(exist_ok=True)
    run(CASES[a.task])


if __name__ == "__main__":
    main()
