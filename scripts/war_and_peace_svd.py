"""Build a Gromov-style chapter-by-term SVD representation for one fixed edition."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from scipy.sparse import csc_matrix
from sklearn.utils.extmath import randomized_svd
from stop_words import get_stop_words

ROMAN_CHAPTER = re.compile(r"(?m)^\s*[IVXLCDM]+\s*$")
RUSSIAN_WORD = re.compile(r"^[а-яё]+$")


def _chapters(text: str) -> list[str]:
    headings = list(ROMAN_CHAPTER.finditer(text))
    result = [
        text[start.end() : end.start() if end else len(text)].strip()
        for start, end in zip(headings, [*headings[1:], None], strict=False)
    ]
    return [chapter for chapter in result if chapter]


def _lemmatize(
    chapter: str, segmenter: Segmenter, tagger: NewsMorphTagger, vocab: MorphVocab
) -> list[str]:
    doc = Doc(chapter)
    doc.segment(segmenter)
    doc.tag_morph(tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(vocab)
        lemma = token.lemma.lower() if token.lemma else ""
        if RUSSIAN_WORD.fullmatch(lemma):
            lemmas.append(lemma)
    return lemmas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = args.text.read_text(encoding="utf-8")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.text.read_bytes()).hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError("Text hash does not match metadata")
    chapters = _chapters(text)
    if len(chapters) < 100:
        raise ValueError("Chapter segmentation did not yield at least 100 documents")

    embedding = NewsEmbedding()
    segmenter, tagger, vocab = Segmenter(), NewsMorphTagger(embedding), MorphVocab()
    stop_words = set(get_stop_words("ru"))
    documents = [
        [
            lemma
            for lemma in _lemmatize(chapter, segmenter, tagger, vocab)
            if lemma not in stop_words
        ]
        for chapter in chapters
    ]
    vocabulary = sorted({lemma for document in documents for lemma in document})
    term_index = {term: index for index, term in enumerate(vocabulary)}
    rows, columns, counts = [], [], []
    document_lengths = []
    for column, document in enumerate(documents):
        frequencies = Counter(document)
        document_lengths.append(sum(frequencies.values()))
        for term, count in frequencies.items():
            rows.append(term_index[term])
            columns.append(column)
            counts.append(count)
    raw = csc_matrix((counts, (rows, columns)), shape=(len(vocabulary), len(documents)))
    term_totals = np.asarray(raw.sum(axis=1)).ravel()
    probabilities = raw.multiply(1.0 / term_totals[:, None]).tocoo()
    entropy = np.bincount(
        probabilities.row,
        weights=-probabilities.data * np.log(probabilities.data),
        minlength=len(vocabulary),
    )
    entropy /= np.log(len(documents))
    weights = (1.0 - entropy)[rows] * np.asarray(counts) / np.asarray(document_lengths)[columns]
    weighted = csc_matrix((weights, (rows, columns)), shape=raw.shape)
    u, singular_values, _ = randomized_svd(weighted, n_components=15, random_state=0)

    bigrams = sorted(
        {
            (term_index[left], term_index[right])
            for document in documents
            for left, right in zip(document, document[1:], strict=False)
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "gromov_chapter_svd.npz",
        vocabulary=np.asarray(vocabulary),
        word_u=u.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        bigram_word_indices=np.asarray(bigrams, dtype=np.int32),
    )
    manifest = {
        "protocol": "war-and-peace-chapter-as-document-gromov-adaptation",
        "source": metadata,
        "chapter_count": len(documents),
        "vocabulary_size": len(vocabulary),
        "unique_bigram_count": len(bigrams),
        "embedding_dimensions": [5, 10, 15],
        "embedding_definition": "rows of U from entropy-weighted chapter-term SVD",
        "preprocessing": {"lemmatizer": "natasha==1.6.0", "stop_words": "stop-words==2018.7.23"},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
