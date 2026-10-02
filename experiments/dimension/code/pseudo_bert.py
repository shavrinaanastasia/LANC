"""BERT on the pseudolanguage: a small BERT trained from scratch (masked-LM) on the same pseudo-texts as
pseudolang.py, then contextual vectors averaged per word (as for War and Peace: ruBERT -> mean over occurrences),
PCA to d = 2/5/10/15 and the full hidden size; the four estimators on words and bigrams.

Texts are regenerated exactly as in pseudolang.py (same seeds), so results are comparable with svd/cbow there.

  python pseudo_bert.py --m 3 --walk eps [--hidden 256 --layers 4 --epochs 6 --max-train-min 170]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

try:  # some cHARISMa nodes lack libsndfile; transformers then crashes importing audio utils we never use
    import soundfile  # noqa: F401
except (OSError, ImportError):
    sys.modules["soundfile"] = None

import numpy as np

from calib_trajectories import measure
from pseudolang import make_points, text_lengths, walk


def regenerate(m, kind, holes, n_points, n_texts, lengths_csv):
    data = make_points(m, n_points, holes)
    V = len(data)
    eps = 1.5 * (1 / V) ** (1 / m)
    rng = np.random.default_rng(2026 + m)
    lens, _, _ = text_lengths(lengths_csv, V, rng)
    texts = [walk(data, int(rng.choice(lens)), kind, np.random.default_rng(i * 8 + 2), eps) for i in range(n_texts)]
    return data, V, texts


def chunks(texts, L):
    out = []
    for t in texts:
        for s in range(0, len(t), L):
            c = t[s:s + L]
            if len(c) >= 8:
                out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--walk", required=True, choices=["dist", "eps"])
    ap.add_argument("--holes", type=int, default=4)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--n-texts", type=int, default=1000)
    ap.add_argument("--lengths-csv", default="russian_Prussner_wolframe100.csv")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-train-min", type=float, default=170)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--emb-dims", type=int, nargs="+", default=[2, 5, 10, 15])
    ap.add_argument("--ladders", type=int, default=2)
    ap.add_argument("--max-points", type=int, default=10000)
    ap.add_argument("--hidalgo-iter", type=int, default=5000)
    ap.add_argument("--out", default="pseudo_results")
    a = ap.parse_args()

    import torch
    from transformers import BertConfig, BertForMaskedLM
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    out = Path(a.out); out.mkdir(exist_ok=True)
    tag = f"cube{a.m}_h{a.holes}_{a.walk}"
    vec_file = out / f"{tag}_bert_vectors.npz"

    t0 = time.time()
    data, V, texts = regenerate(a.m, a.walk, a.holes, a.n_points, a.n_texts, a.lengths_csv)
    print(f"{tag}: V={V}, tokens={sum(map(len, texts))}, texts regenerated in {time.time() - t0:.0f}s", flush=True)
    PAD, MASK = V, V + 1
    seqs = chunks(texts, a.seq)
    meta = {"object": tag, "true_dim": a.m, "walk": a.walk, "V": V, "sequences": len(seqs), "hidden": a.hidden,
            "layers": a.layers, "heads": a.heads, "seq": a.seq, "batch": a.batch, "lr": a.lr}

    if vec_file.exists():
        Z = np.load(vec_file)
        H, meta_tr = Z["H"], json.loads(str(Z["meta"]))
        meta.update(meta_tr)
        print("vectors loaded from", vec_file, flush=True)
    else:
        cfg = BertConfig(vocab_size=V + 2, hidden_size=a.hidden, num_hidden_layers=a.layers, num_attention_heads=a.heads,
                         intermediate_size=4 * a.hidden, max_position_embeddings=a.seq, type_vocab_size=1,
                         pad_token_id=PAD)
        model = BertForMaskedLM(cfg)
        n_par = sum(p.numel() for p in model.parameters())

        def batchify(idx):
            L = max(len(seqs[i]) for i in idx)
            x = torch.full((len(idx), L), PAD, dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i])
            return x, (x != PAD).long()

        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(seqs) / a.batch)
        total = steps_per_epoch * a.epochs
        warm = max(1, int(0.06 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
        g = np.random.default_rng(1)
        hist, step, t_tr, stop = [], 0, time.time(), False
        model.train()
        for ep in range(a.epochs):
            order = g.permutation(len(seqs))
            run_loss, run_acc, nb = 0.0, 0.0, 0
            for b in range(0, len(order), a.batch):
                x, att = batchify(order[b:b + a.batch])
                labels = x.clone()
                sel = (torch.rand(x.shape) < 0.15) & (att == 1)
                labels[~sel] = -100
                r = torch.rand(x.shape)
                inp = x.clone()
                inp[sel & (r < 0.8)] = MASK
                rnd = sel & (r >= 0.8) & (r < 0.9)
                inp[rnd] = torch.randint(0, V, (int(rnd.sum()),))
                o = model(input_ids=inp, attention_mask=att, labels=labels)
                o.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                with torch.no_grad():
                    acc = (o.logits.argmax(-1)[sel] == x[sel]).float().mean().item()
                run_loss += o.loss.item(); run_acc += acc; nb += 1; step += 1
                if step % 50 == 0:
                    el = (time.time() - t_tr) / 60
                    print(f"ep {ep} step {step}/{total} loss {run_loss / nb:.3f} acc {run_acc / nb:.3f} "
                          f"{el:.1f} min ({el / step * total:.0f} min projected)", flush=True)
                if (time.time() - t_tr) / 60 > a.max_train_min:
                    stop = True
                    break
            hist.append({"epoch": ep, "loss": run_loss / max(nb, 1), "mlm_acc": run_acc / max(nb, 1)})
            print("epoch", hist[-1], flush=True)
            if stop:
                print("time limit reached, stopping training", flush=True)
                break
        meta_tr = {"params": n_par, "train_minutes": (time.time() - t_tr) / 60, "steps": step, "planned_steps": total,
                   "history": hist, "stopped_by_time": stop}
        meta.update(meta_tr)
        # contextual vectors: last layer, mean over all occurrences of each word
        model.eval()
        S = np.zeros((V, a.hidden)); C = np.zeros(V)
        t_inf = time.time()
        with torch.no_grad():
            for b in range(0, len(seqs), 128):
                idx = list(range(b, min(b + 128, len(seqs))))
                x, att = batchify(idx)
                h = model.bert(input_ids=x, attention_mask=att).last_hidden_state.numpy()
                xs, ms = x.numpy(), att.numpy().astype(bool)
                np.add.at(S, xs[ms], h[ms]); np.add.at(C, xs[ms], 1)
        H = np.full((V, a.hidden), np.nan)
        H[C > 0] = S[C > 0] / C[C > 0, None]
        meta["inference_minutes"] = (time.time() - t_inf) / 60
        meta_tr["inference_minutes"] = meta["inference_minutes"]
        np.savez_compressed(vec_file, H=H, meta=json.dumps(meta_tr))

    from sklearn.decomposition import PCA
    ok = ~np.isnan(H).any(1)
    pca = PCA(n_components=max(a.emb_dims), random_state=0).fit(H[ok])
    P = np.full((V, max(a.emb_dims)), np.nan); P[ok] = pca.transform(H[ok])
    meta["pca_explained"] = {d: float(pca.explained_variance_ratio_[:d].sum()) for d in a.emb_dims}
    (out / f"{tag}_bert_meta.json").write_text(json.dumps(meta, indent=1))
    print({k: v for k, v in meta.items() if k != "history"}, flush=True)

    embs = {("bert_pca", d): P[:, :d] for d in a.emb_dims}
    embs[("bert_full", a.hidden)] = H
    used = np.unique(np.concatenate(texts))
    big = np.array(sorted({(x, y) for t in texts for x, y in zip(t, t[1:])}), dtype=np.int64)
    mask = np.zeros(V, bool); mask[used] = True
    for (emb, d), W in embs.items():
        for n in (1, 2):
            f = out / f"{tag}_{emb}_d{d}_n{n}.json"
            if f.exists():
                continue
            good = ~np.isnan(W).any(1) & mask
            if n == 1:
                Xc = W[good]
            else:
                b = big[good[big[:, 0]] & good[big[:, 1]]]
                Xc = np.hstack([W[b[:, 0]], W[b[:, 1]]])
            t1 = time.time()
            r = measure(Xc, a.ladders, a.max_points, a.hidalgo_iter)
            r.update({"object": tag, "true_dim": a.m, "embedding": emb, "d": d, "ngram": n, "seconds": time.time() - t1})
            f.write_text(json.dumps(r))
            sw = r["schweinhart"]
            print(f"{tag:16s} {emb:9s} d={d:<3d} n={n} N={r['N']:6d} | Schw {sw['min']}-{sw['max']} | TwoNN {r['twonn']:.2f} "
                  f"| FisherS {r['fishers']} | Hidalgo {np.round(r['hidalgo']['d_k'], 2).tolist()} ({r['seconds']:.0f}s)",
                  flush=True)
    (out / f"{tag}_bert_done.json").write_text("{}")


if __name__ == "__main__":
    main()
