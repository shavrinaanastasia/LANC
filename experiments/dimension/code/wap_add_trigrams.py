"""Add the unique trigram list (adjacent kept lemmas inside one sentence) to an embeddings file.
Usage: python wap_add_trigrams.py in.npz wap_sentences.json.gz out.npz"""
import gzip
import json
import sys

import numpy as np

src, sents, dst = sys.argv[1:4]
data = dict(np.load(src))
index = {str(w): i for i, w in enumerate(data["vocabulary"])}
tri = set()
for s in json.load(gzip.open(sents, "rt", encoding="utf-8")):
    ids = [index[w] for w in s]
    tri.update(zip(ids, ids[1:], ids[2:]))
data["trigrams"] = np.array(sorted(tri), dtype=np.int32)
np.savez_compressed(dst, **data)
print("unique trigrams:", len(tri), "->", dst)
