"""Zero-infra in-memory backends — dict + BM25, pure Python (no DB, no numpy).

These run the full search algorithm with no infrastructure. The graph indexes node
labels with BM25 (CJK bi-grams, so Korean/CJK works with no analyzer), enumerates
class instances and walks 1-hop relations; the vector store uses cosine when
embeddings are present, else BM25 over chunk text.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from ..models import Chunk, Node
from ..text import BM25, tokenize

_ISA = {"instanceOf", "type", "subClassOf", "rdf:type"}


class InMemoryGraph:
    """Nodes + (s, p, o) id-triples, indexed for label search and traversal."""

    def __init__(self, nodes: list[Node], triples: list[tuple[str, str, str]]):
        self.nodes: dict[str, Node] = {n.id: n for n in nodes}
        self.triples = triples
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        self._members: dict[str, list[str]] = {}
        self._label_index: dict[str, str] = {}
        for n in nodes:
            self._label_index.setdefault(n.label, n.id)
        for s, p, o in triples:
            sl, ol = self._label(s), self._label(o)
            self._adj.setdefault(s, []).append((sl, p, ol))
            self._adj.setdefault(o, []).append((sl, p, ol))
            if p in _ISA:
                self._members.setdefault(o, []).append(s)
        self._ids = list(self.nodes.keys())
        self._bm25 = BM25([tokenize(self.nodes[i].label) for i in self._ids])

    def _label(self, nid: str) -> str:
        n = self.nodes.get(nid)
        return n.label if n else nid

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        return [(self.nodes[self._ids[i]], s) for i, s in self._bm25.search(query, limit=limit)]

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        ids = self._members.get(class_id, [])
        return [self.nodes[i] for i in ids[:limit] if i in self.nodes]

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        out = list(self._adj.get(node_id, []))[:limit]
        if hops > 1:
            seen = {node_id}
            frontier = {t[2] for t in out} | {t[0] for t in out}
            for _ in range(hops - 1):
                for lbl in list(frontier):
                    nid = self._label_index.get(lbl)
                    if nid and nid not in seen:
                        seen.add(nid)
                        out.extend(self._adj.get(nid, [])[:limit])
                frontier = set()
        return out[:limit]

    def count_class(self, class_id: str) -> int:
        return len(self._members.get(class_id, []))

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class InMemoryVector:
    """Passage store: cosine when embeddings + embedder present, else BM25 over text."""

    def __init__(self, chunks: list[Chunk], *, embedder: Optional[Callable[[str], list[float]]] = None):
        self.chunks = chunks
        self._by_id = {c.id: c for c in chunks}
        self.embedder = embedder
        self._use_vec = embedder is not None and bool(chunks) and all(c.embedding for c in chunks)
        if not self._use_vec:
            self._bm25 = BM25([tokenize(c.text) for c in chunks])

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        if self._use_vec:
            q = self.embedder(query)  # type: ignore[misc]
            scored = [(c, _cosine(q, c.embedding)) for c in self.chunks if c.embedding]
            scored.sort(key=lambda x: -x[1])
            return scored[:limit]
        return [(self.chunks[i], s) for i, s in self._bm25.search(query, limit=limit)]

    def fetch(self, ids: list[str]) -> list[Chunk]:
        return [self._by_id[i] for i in ids if i in self._by_id]


class InMemoryGraphSink:
    """A GraphSink that just keeps the uploaded Turtle (for tests / dry runs)."""

    def __init__(self):
        self.graphs: dict[Optional[str], str] = {}

    def upload_turtle(self, ttl: str, *, graph: Optional[str] = None, clear: bool = False) -> None:
        if clear or graph not in self.graphs:
            self.graphs[graph] = ttl
        else:
            self.graphs[graph] += "\n" + ttl
