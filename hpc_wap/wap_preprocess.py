"""War and Peace -> Gromov-style SVD word embeddings + unique bigram list.

Protocol (Gromov, Borodin, Yerbolova, Complexity 2024, sec. 4.1.1 and 6), adapted to one novel:
  documents        = chapters (Roman-numeral headings), since one novel is one "text";
                     editor's footnotes (translations of French) and [n] markers removed
  tokenisation     = natasha Segmenter (razdel) -> sentences / tokens
  class labels     = PROPN -> <PROPN>, NUM or digits -> <NUM>   ("proper nouns, numerals -> class labels")
  stop words       = POS in {ADP, CCONJ, SCONJ, PART, INTJ, PUNCT, SYM} + `stop-words` Russian list
  lemmatisation    = natasha NewsMorphTagger + MorphVocab
  kept tokens      = Cyrillic lemmas and the two class labels (French passages are dropped)
  matrix           = s_ij = (1 - eps_i) * k_ij / sum_i' k_i'j   (eq. 1-3 of the paper)
  embedding        = rows of U from the exact (dense LAPACK) SVD, first 15 components
  bigrams          = unique ordered pairs of adjacent kept tokens inside one sentence

Run:  python wap_preprocess.py --text war_and_peace_ru.txt --out wap_embeddings.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "14935bc6b407949ed8d375ee6e725370ed55ed3233205c20e7a2ac17c9e373c2"
CHAPTER_HEADING = re.compile(r"(?m)^[ \t]*[IVXLC]+[ \t]*\.?[ \t]*$")
CYRILLIC = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$")
DROP_POS = {"ADP", "CCONJ", "SCONJ", "PART", "INTJ", "PUNCT", "SYM"}
N_COMPONENTS = 15


NLTK_RUSSIAN = """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до
вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем
была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним
здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об
другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем
хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между""".split()


def load_stop_words():
    """stop-words package if present, else NLTK corpus, else the NLTK Russian list embedded above."""
    try:
        from stop_words import get_stop_words

        return {w.replace("ё", "е") for w in get_stop_words("ru")}, "stop-words package (ru)"
    except Exception:
        pass
    try:
        from nltk.corpus import stopwords

        return {w.replace("ё", "е") for w in stopwords.words("russian")}, "nltk corpus (russian)"
    except Exception:
        return set(NLTK_RUSSIAN), "embedded NLTK russian list"


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc), digest
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode text as UTF-8 or CP1251")


def split_chapters(text: str) -> list[str]:
    # drop the editor's footnotes (Russian translations of French passages) and their markers
    text = re.split(r"(?m)^\s*ПРИМЕЧАНИЯ\s*$", text)[0]
    text = re.sub(r"\[\d+\]", "", text)
    heads = list(CHAPTER_HEADING.finditer(text))
    chapters = []
    for a, b in zip(heads, heads[1:] + [None]):
        body = text[a.end(): b.start() if b else len(text)]
        # cut trailing volume/part headings that precede the next chapter
        body = re.split(r"(?mi)^\s*(?:ТОМ|ЧАСТЬ|ЭПИЛОГ)\b.*$", body)[0].strip()
        if len(body) > 200:
            chapters.append(body)
    return chapters


def tokenise(chapters, stop_words, log=print, spans=None):
    """If `spans` is a list, it receives per chapter a list of sentences of (start, stop) char offsets
    of every kept token, parallel to the returned lemmas (used by the BERT embedding script)."""
    import inspect

    if not hasattr(inspect, "getargspec"):  # pymorphy2 on Python >= 3.11
        from collections import namedtuple

        _A = namedtuple("ArgSpec", "args varargs keywords defaults")
        inspect.getargspec = lambda f: _A(*inspect.getfullargspec(f)[:4])
    from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

    seg, vocab = Segmenter(), MorphVocab()
    tagger = NewsMorphTagger(NewsEmbedding())
    docs, stats = [], Counter()
    for ci, chapter in enumerate(chapters):
        doc = Doc(chapter)
        doc.segment(seg)
        doc.tag_morph(tagger)
        sentences, sent_spans = [], []
        for sent in doc.sents:
            toks, offs = [], []
            for tok in sent.tokens:
                stats["tokens"] += 1
                if tok.pos in DROP_POS:
                    stats["dropped_pos"] += 1
                    continue
                if tok.pos == "NUM" or tok.text.isdigit():
                    toks.append("<NUM>")
                    offs.append((tok.start, tok.stop))
                    continue
                if tok.pos == "PROPN":
                    toks.append("<PROPN>")
                    offs.append((tok.start, tok.stop))
                    continue
                tok.lemmatize(vocab)
                lemma = (tok.lemma or tok.text).lower().replace("ё", "е")
                if not CYRILLIC.match(lemma):
                    stats["dropped_non_cyrillic"] += 1
                    continue
                if lemma in stop_words:
                    stats["dropped_stopword"] += 1
                    continue
                toks.append(lemma)
                offs.append((tok.start, tok.stop))
            if toks:
                sentences.append(toks)
                sent_spans.append(offs)
        docs.append(sentences)
        if spans is not None:
            spans.append(sent_spans)
        if (ci + 1) % 50 == 0:
            log(f"  lemmatised {ci + 1}/{len(chapters)} chapters")
    return docs, stats


def build(docs):
    vocab = sorted({t for d in docs for s in d for t in s})
    index = {w: i for i, w in enumerate(vocab)}
    M, L = len(vocab), len(docs)
    k = np.zeros((M, L), dtype=np.float64)
    bigrams = set()
    for j, d in enumerate(docs):
        for s in d:
            ids = [index[t] for t in s]
            for i in ids:
                k[i, j] += 1
            bigrams.update(zip(ids, ids[1:]))
    tau = k.sum(axis=1)
    p = k / tau[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(L)
    s = (1.0 - ent)[:, None] * k / k.sum(axis=0)[None, :]
    u, sv, _ = np.linalg.svd(s, full_matrices=False)
    u = u[:, :N_COMPONENTS]
    # deterministic sign convention: largest |entry| of each column is positive
    signs = np.sign(u[np.abs(u).argmax(axis=0), np.arange(u.shape[1])])
    u *= signs
    return vocab, k, ent, u, sv, np.array(sorted(bigrams), dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, type=Path)
    ap.add_argument("--out", default=Path("wap_embeddings.npz"), type=Path)
    ap.add_argument("--allow-other-edition", action="store_true")
    a = ap.parse_args()

    text, digest = read_text(a.text)
    print("sha256", digest)
    if digest != EXPECTED_SHA256 and not a.allow_other_edition:
        raise SystemExit("SHA-256 differs from the fixed edition; pass --allow-other-edition to override")
    chapters = split_chapters(text)
    print("chapters", len(chapters))

    stop, stop_source = load_stop_words()
    docs, st = tokenise(chapters, stop)
    vocab, k, ent, u, sv, bigrams = build(docs)
    rows_unique = np.unique(np.round(u, 12), axis=0).shape[0]
    manifest = {
        "protocol": "gromov2024-svd, chapters-as-documents adaptation (War and Peace)",
        "sha256": digest,
        "chapters": len(chapters),
        "token_stats": dict(st),
        "kept_tokens": int(k.sum()),
        "unique_words": len(vocab),
        "unique_bigrams": int(len(bigrams)),
        "distinct_word_embedding_rows_d15": int(rows_unique),
        "singular_values_top15": sv[:15].tolist(),
        "stop_word_list": stop_source,
        "stop_word_count": len(stop),
        "drop_pos": sorted(DROP_POS),
    }
    try:
        import natasha
        manifest["natasha_version"] = getattr(natasha, "__version__", "unknown")
    except Exception:
        pass
    np.savez_compressed(a.out, vocabulary=np.array(vocab), word_u=u, singular_values=sv,
                        entropy=ent, word_freq=k.sum(axis=1), bigrams=bigrams)
    Path(str(a.out).replace(".npz", "_manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
