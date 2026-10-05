"""War and Peace: Cao E2, Kennel FNN and k-gram correlation slopes (calib2.delay_stats) for CBOW word vectors,
to place natural language among the calibration groups (flows / fractal walks / Brownian walks).
Text = lemma sentences in book order, cut into 200-word chunks. Control: words shuffled inside each chunk."""
import gzip, json, time
import numpy as np
from calib2 import delay_stats
from wap_embed_alt import cbow_vectors

z = dict(np.load("wap_embeddings_cbow.npz", allow_pickle=True))
vocab = [str(v) for v in z["vocabulary"]]; idx = {w: i for i, w in enumerate(vocab)}
sents = json.load(gzip.open("wap_sentences.json.gz", "rt", encoding="utf-8"))
print("tokens", sum(len(s) for s in sents), "vocab", len(vocab), flush=True)
emb = {15: z["cbow15"]}
emb[30] = cbow_vectors([sents], vocab, [30], lambda m: None)[30]
chunks, cur = [], []
for s in sents:
    cur += [idx[w] for w in s if w in idx]
    if len(cur) >= 200:
        chunks.append(np.array(cur)); cur = []
rng = np.random.default_rng(0)
variants = {"real": chunks, "shuffled": [rng.permutation(c) for c in chunks]}
out = {}
for d in (15, 30):
    for name, ch in variants.items():
        t0 = time.time()
        r = delay_stats([emb[d][c] for c in ch], 8)
        out[f"wap_d{d}_{name}"] = {str(k): {"fnn10": v["fnn"]["rtol10"], "E1": v["E1"], "E2": v["E2"], "tw": v["twonn_kcloud"],
                                           "cs": v["corr_slope"]} for k, v in r.items()}
        print(d, name, len(ch), "E2(k=2..4)", [round(r[k]["E2"], 3) for k in (2, 3, 4)],
              "slope C=.01 k=1..8", [round(r[k]["corr_slope"]["0.01"] or 0, 1) for k in r], f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open("wap_delay.json", "w"))
print("DONE", flush=True)
