"""Predicate governance — closed-IE style normalization of relation names.

Free-form extraction sprays near-synonym predicates ("belongs", "is part of",
"part-of"). This folds *surface variants* to one canonical form and anchors them
to the schema's declared object-property vocabulary, without discarding genuinely
different relations. The default suffix set strips Korean relational endings;
pass ``suffix_pattern=`` for another language (or ``None`` to disable).

Meaning-level synonymy that survives surface normalization (different spellings of
the same idea) is handled separately by the embedding dedup in :mod:`.dedup`.
"""
from __future__ import annotations

import re

from ..models import ObjectProperty, Relation

# Korean relational endings — removed only to merge surface variants of one form.
_DEFAULT_SUFFIX = re.compile(
    r"(되어있는것|되어있음|되어있다|되어있는|되어진|되어야|되어|되는|된다|되며|된|됨|"
    r"하는것|하는|하다|한다|하며|하고|"
    r"이다|이며|있는|있다|있음)$"
)


def normalize_predicate(pred: str, *, suffix_pattern: re.Pattern | None = _DEFAULT_SUFFIX) -> str:
    """Lowercase, strip separators, peel repeated relational endings -> a form key."""
    if not pred:
        return ""
    p = pred.strip().lower()
    p = re.sub(r"[\s_\-/]+", "", p)
    if suffix_pattern is not None:
        for _ in range(3):
            new = suffix_pattern.sub("", p)
            if new == p or not new:
                break
            p = new
    return p


def govern_predicates(
    relations: list[Relation],
    object_properties: list[ObjectProperty] | None,
    *,
    suffix_pattern: re.Pattern | None = _DEFAULT_SUFFIX,
) -> dict:
    """Fold predicate surface variants to one canonical name, in place.

    Returns stats ``{total, distinct_before, distinct_after, merged, anchored_to_schema}``."""
    if not relations:
        return {"total": 0, "distinct_before": 0, "distinct_after": 0, "merged": 0, "anchored_to_schema": 0}

    canon_by_norm: dict[str, str] = {}
    schema_norms: set[str] = set()
    for op in (object_properties or []):
        name = (op.name or "").strip()
        if not name:
            continue
        norm = normalize_predicate(name, suffix_pattern=suffix_pattern)
        if norm and norm not in canon_by_norm:
            canon_by_norm[norm] = name
            schema_norms.add(norm)

    for rel in relations:
        pred = (rel.predicate or "").strip()
        norm = normalize_predicate(pred, suffix_pattern=suffix_pattern)
        if not norm:
            continue
        if norm not in canon_by_norm:
            canon_by_norm[norm] = pred
        elif norm not in schema_norms and len(pred) < len(canon_by_norm[norm]):
            canon_by_norm[norm] = pred

    distinct_before = {(r.predicate or "").strip() for r in relations if (r.predicate or "").strip()}
    used: set[str] = set()
    anchored = 0
    for rel in relations:
        pred = (rel.predicate or "").strip()
        norm = normalize_predicate(pred, suffix_pattern=suffix_pattern)
        if not norm:
            continue
        canon = canon_by_norm.get(norm, pred)
        if canon != pred:
            rel.predicate = canon
        if norm in schema_norms:
            anchored += 1
        used.add(canon)

    return {
        "total": len(relations),
        "distinct_before": len(distinct_before),
        "distinct_after": len(used),
        "merged": max(0, len(distinct_before) - len(used)),
        "anchored_to_schema": anchored,
    }
