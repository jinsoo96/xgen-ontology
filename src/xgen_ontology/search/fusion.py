"""Fusion primitives — diversity (MMR), adaptive cut, relation ranking.

Pure functions, no deps. This is what makes fused evidence *non-redundant*: top-k
by relevance alone lets near-duplicate passages crowd out the decisive minority
(warnings, exceptions, reversals); MMR keeps both.
"""
from __future__ import annotations

from ..text import tokenize


def _toks(s: str) -> set[str]:
    return set(tokenize(s))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dynamic_cut(scored: list[tuple], *, ratio: float = 0.55, min_k: int = 4, max_k: int = 40) -> list:
    """Adaptive top-k: keep items scoring >= ratio * top_score (clamped to [min_k, max_k])."""
    if not scored:
        return []
    top = scored[0][1] or 1e-9
    keep = [it for it, sc in scored if sc >= ratio * top][:max_k]
    if len(keep) < min_k:
        keep = [it for it, _ in scored[:min_k]]
    return keep


def mmr(candidates: list[tuple], *, lam: float = 0.72, k: int = 16) -> list:
    """Maximal Marginal Relevance over (text, relevance); similarity = token Jaccard."""
    if not candidates:
        return []
    pool = [(text, rel, _toks(text)) for text, rel in candidates]
    rmax = max((r for _, r, _ in pool), default=1.0) or 1.0
    selected: list[tuple] = []
    chosen: list[set[str]] = []
    while pool and len(selected) < k:
        best_i, best_v = 0, -1e18
        for i, (text, rel, tks) in enumerate(pool):
            div = max((jaccard(tks, c) for c in chosen), default=0.0)
            v = lam * (rel / rmax) - (1 - lam) * div
            if v > best_v:
                best_v, best_i = v, i
        text, rel, _tks = pool.pop(best_i)
        selected.append((text, rel))
        chosen.append(_tks)
    return [t for t, _ in selected]


def rank_relations(triples: list[str], question: str, *, limit: int = 40) -> list[str]:
    """Dedup relation strings and rank by query-term hits."""
    q = set(tokenize(question))
    seen: set[str] = set()
    uniq: list[str] = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    uniq.sort(key=lambda t: -len(q & set(tokenize(t))))
    return uniq[:limit]
