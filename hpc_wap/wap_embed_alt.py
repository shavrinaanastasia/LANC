"""Alternative word embeddings for the SAME lemma vocabulary and bigram list as the SVD run.

  CBOW   : word2vec CBOW (Gromov et al. 2024, Table 5 method), trained on the lemmatised
           sentences of the novel, separately for d = 5, 10, 15.
           window=5, negative=5, min_count=1, epochs=30, seed=1, workers=1 (deterministic).
  ruBERT : DeepPavlov/rubert-base-cased, last hidden layer (768). For every kept token occurrence
           the vectors of its sub-word pieces are averaged; a lemma's vector is the mean over all its
           occurrences in the novel (so <PROPN>, <NUM> are the mean of all names / numerals).
           Context = consecutive sentences of one chapter packed into windows of <= 510 pieces.
           PCA of the 17 900 lemma vectors gives the 5/10/15-dim versions (bigram = concatenation).

Output: wap_embeddings_alt.npz with vocabulary, bigrams (identical to wap_embeddings.npz),
cbow5, cbow10, cbow15, bert768 (float32), bert_pca15 (first 15 principal components).

Runs offline on the laptop (HF cache already holds rubert-base-cased).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from wap_preprocess import EXPECTED_SHA256, build, load_stop_words, read_text, split_chapters, tokenise  # noqa: E402

MODEL = "DeepPavlov/rubert-base-cased"


def cbow_vectors(docs, vocab, dims, log):  # used by wap_cbow_hse.py
    import gensim
    from gensim.models import Word2Vec

    sentences = [s for d in docs for s in d]
    major = int(gensim.__version__.split(".")[0])
    out = {}
    for d in dims:
        kw = dict(sg=0, window=5, negative=5, min_count=1, seed=1, workers=1)
        if major >= 4:
            m = Word2Vec(sentences, vector_size=d, epochs=30, **kw)
            wv = m.wv
        else:
            m = Word2Vec(sentences, size=d, iter=30, **kw)
            wv = m.wv
        out[d] = np.array([wv[w] for w in vocab], dtype=np.float64)
        log(f"  CBOW d={d} trained (gensim {gensim.__version__})")
    return out


def bert_vectors(chapters, docs, spans, vocab, log, batch=8):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    index = {w: i for i, w in enumerate(vocab)}
    H = model.config.hidden_size
    sums = np.zeros((len(vocab), H), dtype=np.float64)
    counts = np.zeros(len(vocab), dtype=np.int64)

    # Windows are grown token by token and measured on the REAL text slice (gaps included:
    # dropped French passages, punctuation, stop words), so no window is ever truncated.
    windows = []  # (chapter, [(lemma_id, start, stop), ...])
    for ci, (text, sents, offs) in enumerate(zip(chapters, docs, spans)):
        cur, cur_len, prev_e = [], 0, None
        for lemmas, so in zip(sents, offs):
            for lem, (s0, e0) in zip(lemmas, so):
                if cur:
                    add = len(tok.tokenize(text[prev_e:e0]))
                    if cur_len + add > 440:
                        windows.append((ci, cur))
                        cur, cur_len = [], 0
                if not cur:
                    add = len(tok.tokenize(text[s0:e0]))
                cur.append((index[lem], s0, e0))
                cur_len += add
                prev_e = e0
        if cur:
            windows.append((ci, cur))
    log(f"  {len(windows)} BERT windows")

    def encode(texts):
        enc = tok(texts, return_offsets_mapping=True, padding=True, truncation=True, max_length=512,
                  return_tensors="pt")
        offsets = enc.pop("offset_mapping").numpy()
        with torch.no_grad():
            return model(**enc).last_hidden_state.numpy(), offsets

    leftovers = []  # (lemma_id, surface) occurrences not matched inside their window
    t0 = time.time()
    for b in range(0, len(windows), batch):
        chunk = windows[b: b + batch]
        texts, metas = [], []
        for ci, items in chunk:
            a, z = items[0][1], items[-1][2]
            texts.append(chapters[ci][a:z])
            metas.append((ci, a, items))
        hid, offsets = encode(texts)
        for k, (ci, a, items) in enumerate(metas):
            om = offsets[k]
            valid = (om[:, 1] > om[:, 0])
            for lid, s, e in items:
                sel = valid & (om[:, 0] < e - a) & (om[:, 1] > s - a)
                if not sel.any():
                    leftovers.append((lid, chapters[ci][s:e]))
                    continue
                sums[lid] += hid[k][sel].mean(axis=0)
                counts[lid] += 1
        if (b // batch) % 25 == 0:
            done = min(b + batch, len(windows))
            el = time.time() - t0
            log(f"  BERT {done}/{len(windows)} windows, {el/60:.1f} min, eta {el/done*(len(windows)-done)/60:.1f} min")
    missed = len(leftovers)
    # safety net: an occurrence that still did not match is encoded on its own (no context)
    for b in range(0, len(leftovers), 64):
        part = leftovers[b: b + 64]
        hid, offsets = encode([t for _, t in part])
        for k, (lid, _) in enumerate(part):
            sel = offsets[k][:, 1] > offsets[k][:, 0]
            if sel.any():
                sums[lid] += hid[k][sel].mean(axis=0)
                counts[lid] += 1
    log(f"  unmatched occurrences re-encoded without context: {missed}")
    if (counts == 0).any():
        raise RuntimeError(f"{int((counts == 0).sum())} lemmas got no BERT vector even without context")
    return (sums / counts[:, None]).astype(np.float32), int(missed), int(counts.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, type=Path)
    ap.add_argument("--svd", default="wap_embeddings.npz", help="to check vocabulary/bigrams are identical")
    ap.add_argument("--out", default="wap_embeddings_alt.npz")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731

    text, digest = read_text(a.text)
    if digest != EXPECTED_SHA256:
        raise SystemExit("unexpected edition hash")
    chapters = split_chapters(text)
    stop, _ = load_stop_words()
    spans = []
    log("lemmatising (natasha)...")
    docs, _ = tokenise(chapters, stop, log=log, spans=spans)
    vocab, k, ent, u, sv, bigrams = build(docs)
    ref = np.load(a.svd)
    if list(ref["vocabulary"]) != vocab or not np.array_equal(ref["bigrams"], bigrams):
        raise SystemExit("vocabulary/bigrams differ from the SVD run")
    log(f"vocabulary {len(vocab)}, bigrams {len(bigrams)} -- identical to SVD run")

    # CBOW is trained on HSE (gensim there has the fast C routines; laptop gensim 3.8 does not):
    import gzip
    with gzip.open("wap_sentences.json.gz", "wt", encoding="utf-8") as fh:
        json.dump([s for d in docs for s in d], fh, ensure_ascii=False)
    log("lemma sentences saved to wap_sentences.json.gz (CBOW is trained on HSE)")
    log("ruBERT...")
    bert, missed, occ = bert_vectors(chapters, docs, spans, vocab, log)
    c = bert.astype(np.float64) - bert.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    pca = c @ vt[:15].T
    evr = (s ** 2 / (s ** 2).sum())[:15]

    np.savez_compressed(a.out, vocabulary=np.array(vocab), bigrams=bigrams, bert768=bert, bert_pca15=pca)
    man = {"model": MODEL, "layer": "last_hidden_state", "occurrences": occ, "occurrences_unmatched": missed,
           "pca_explained_variance_ratio_top15": evr.tolist(),
           "pca_cumulative_5_10_15": [float(evr[:5].sum()), float(evr[:10].sum()), float(evr[:15].sum())],
           "distinct_rows": {"bert768": int(np.unique(bert, axis=0).shape[0])}}
    Path(a.out.replace(".npz", "_manifest.json")).write_text(json.dumps(man, indent=2))
    log(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
