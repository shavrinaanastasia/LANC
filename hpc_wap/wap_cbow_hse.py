"""Train CBOW (word2vec) on the lemma sentences on HSE and add cbow5/10/15 to the embeddings file.
Gromov et al. 2024 Table 5 method; window=5, negative=5, min_count=1, epochs=30, seed=1, workers=1."""
import gzip
import json
import sys

import numpy as np

from wap_embed_alt import cbow_vectors

src, sents, dst = sys.argv[1:4]
data = dict(np.load(src))
sentences = json.load(gzip.open(sents, "rt", encoding="utf-8"))
vocab = [str(v) for v in data["vocabulary"]]
missing = set(vocab) - {w for s in sentences for w in s}
assert not missing, f"{len(missing)} lemmas absent from sentences"
cb = cbow_vectors([sentences], vocab, [5, 10, 15], lambda m: print(m, flush=True))
for d in (5, 10, 15):
    data[f"cbow{d}"] = cb[d]
    print(f"cbow{d}: distinct rows {np.unique(cb[d], axis=0).shape[0]} of {len(vocab)}", flush=True)
np.savez_compressed(dst, **data)
print("saved", dst)
