"""Tokenization, BM25 and IRI helpers — pure Python, zero deps.

CJK is tokenized into character bi-grams (like Lucene's CJK analyzer) so Korean /
Chinese / Japanese label & passage search works with no morphological analyzer;
Latin text falls back to word tokens. BM25 Okapi gives ranked full-text retrieval
over both — the in-memory stand-in for a jena-text / Lucene index.
"""
from __future__ import annotations

import math
import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[가-힣぀-ヿ一-鿿]+")
# Characters not allowed unescaped in a Turtle IRIREF local name (plus whitespace).
_IRI_BAD = re.compile(r"[\s<>\"{}|^`\\()\[\],;:/?#%&*!+=~'·•«»‘’“”]")


def tokenize(text: str) -> list[str]:
    """Latin word tokens + CJK character bi-grams (unigram if length 1)."""
    text = (text or "").lower()
    toks = _WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def safe_uri(name: str) -> str:
    """A Turtle-safe IRI local name. Keeps Unicode letters/digits (incl. Korean),
    collapses everything else to ``_``. Returns ``"Unknown"`` if nothing survives."""
    name = unicodedata.normalize("NFC", (name or "").strip())
    if not name:
        return "Unknown"
    out = _IRI_BAD.sub("_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "Unknown"


def normalize_name(name: str) -> str:
    """NFC + whitespace collapse + strip stray quote/punctuation artefacts."""
    if not name:
        return ""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.strip("\"'.,;:")


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(self, docs_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for d in docs_tokens:
            c: dict[str, int] = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            self.tf.append(c)
            for t in c:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q_tokens: list[str], i: int) -> float:
        tf = self.tf[i]
        dl = self.dl[i] or 1
        s = 0.0
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]
