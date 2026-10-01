"""Frame-specific CBOW spaces from the short-dialogue corpus (run short-10turn-cbow-v1).

Stage 1 (env Google_Colab_GPU_2025, spaCy en_core_web_sm):
    python dialogue_cbow.py lemmatize
  - uses only (stimulus, seed) pairs present in ALL frames, so the three corpora have the same design;
  - every utterance -> spaCy sentences -> lemmas, lower-cased; PROPN -> <PROPN>, NUM -> <NUM>;
    punctuation, spaces and spaCy stop words dropped (same recipe as for War and Peace, English analogue);
  - writes lemmas_<frame>.json.gz (list of sentences) and lemma_stats.json.

Stage 2 (env Python/Anaconda, gensim 4.3):
    python dialogue_cbow.py train [--min-count 1]
  - CBOW exactly as for War and Peace: sg=0, window 5, negative 5, 30 epochs, seed 1, workers 1, d = 5/10/15;
  - writes cbow_<frame>_mc<k>.npz with vocabulary, bigrams (unique adjacent lemma pairs inside a sentence,
    both words in the vocabulary), word_freq, cbow5, cbow10, cbow15 -- the format read by
    ~/wap_run/wap_schweinhart.py --source cbow (words and bigrams, concatenation).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent / "results" / "short-10turn-cbow-v1"
OUT = RUN / "cbow"
FRAMES = ["competition", "cooperation", "neutral"]


def load_corpus(frame):
    rows = [json.loads(x) for x in (RUN / f"corpus_{frame}.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return {(r["stimulus_id"], r["seed"]): r["utterances"] for r in rows}


def lemmatize():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    corpora = {f: load_corpus(f) for f in FRAMES}
    common = set.intersection(*(set(c) for c in corpora.values()))
    OUT.mkdir(exist_ok=True)
    stats = {"common_dialogues_per_frame": len(common)}
    for f in FRAMES:
        texts = [u for k in sorted(common) for u in corpora[f][k]]
        sents = []
        for doc in nlp.pipe(texts, batch_size=256):
            for s in doc.sents:
                lem = []
                for t in s:
                    if t.is_punct or t.is_space or not any(ch.isalnum() for ch in t.text):
                        continue
                    if t.pos_ == "PROPN":
                        lem.append("<PROPN>")
                    elif t.pos_ == "NUM" or t.like_num:
                        lem.append("<NUM>")
                    elif not t.is_stop:
                        lem.append(t.lemma_.lower())
                if lem:
                    sents.append(lem)
        with gzip.open(OUT / f"lemmas_{f}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(sents, fh)
        freq = Counter(w for s in sents for w in s)
        stats[f] = {"sentences": len(sents), "tokens": sum(freq.values()), "types": len(freq),
                    "hapax_share": round(sum(1 for c in freq.values() if c == 1) / max(len(freq), 1), 3),
                    "top20": freq.most_common(20)}
        print(f, {k: v for k, v in stats[f].items() if k != "top20"}, flush=True)
    (OUT / "lemma_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def train(min_count):
    import numpy as np
    from gensim.models import Word2Vec
    for f in FRAMES:
        with gzip.open(OUT / f"lemmas_{f}.json.gz", "rt", encoding="utf-8") as fh:
            sents = json.load(fh)
        freq = Counter(w for s in sents for w in s)
        vocab = sorted(w for w, c in freq.items() if c >= min_count)
        index = {w: i for i, w in enumerate(vocab)}
        vecs = {}
        for d in (5, 10, 15):
            m = Word2Vec(sents, vector_size=d, sg=0, window=5, negative=5, min_count=min_count,
                         epochs=30, seed=1, workers=1)
            vecs[f"cbow{d}"] = np.array([m.wv[w] for w in vocab], dtype=np.float64)
        pairs = {(index[a], index[b]) for s in sents for a, b in zip(s, s[1:]) if a in index and b in index}
        bigrams = np.array(sorted(pairs), dtype=np.int64)
        np.savez_compressed(OUT / f"cbow_{f}_mc{min_count}.npz", vocabulary=np.array(vocab), bigrams=bigrams,
                            word_freq=np.array([freq[w] for w in vocab]), **vecs)
        print(f"{f} mc{min_count}: vocabulary {len(vocab)}, bigrams {len(bigrams)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lemmatize", "train"])
    ap.add_argument("--min-count", type=int, default=1)
    a = ap.parse_args()
    lemmatize() if a.stage == "lemmatize" else train(a.min_count)


if __name__ == "__main__":
    main()
