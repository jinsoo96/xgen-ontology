"""Class hierarchy cleaning + Semantic Context Synthesis (SCS).

* ``clean_hierarchy`` — keep only genuine is-a edges: a parent must be a *class*,
  not a property/relation name an extractor mislabeled as a parent ("being linked
  is not being a subclass"). Also drops self-loops and cycles.
* ``SCSGenerator`` — per class, aggregate **direct** properties (horizontal) and
  **inherited** properties down the is-a chain (vertical), and synthesize a short
  natural-language context summary (LLM if available, rule-based fallback).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..llm import invoke_json
from ..models import Concepts


def clean_hierarchy(concepts: Concepts) -> int:
    """Drop hierarchy edges whose parent isn't a real class (in place). Returns dropped count."""
    class_names = {c.name for c in concepts.classes if c.name}

    kept: list[tuple[str, str]] = []
    seen: set = set()
    dropped = 0
    # collect edges from both class_hierarchy and class.parent
    edges = list(concepts.class_hierarchy)
    for c in concepts.classes:
        if c.parent:
            edges.append((c.parent, c.name))
    for parent, child in edges:
        if not parent or not child or parent == child:
            dropped += 1
            continue
        if parent not in class_names:
            # a property masquerading as a parent, or an undefined node -> not is-a
            dropped += 1
            continue
        if (parent, child) in seen:
            continue
        seen.add((parent, child))
        kept.append((parent, child))

    # break cycles deterministically (drop the edge that closes a cycle)
    parent_of: dict[str, str] = {}
    final: list[tuple[str, str]] = []
    for parent, child in kept:
        if _creates_cycle(parent_of, child, parent):
            dropped += 1
            continue
        parent_of[child] = parent
        final.append((parent, child))

    concepts.class_hierarchy = final
    # normalize class.parent to match the cleaned hierarchy
    for c in concepts.classes:
        c.parent = parent_of.get(c.name)
    return dropped


def _creates_cycle(parent_of: dict[str, str], child: str, parent: str) -> bool:
    cur = parent
    seen = {child}
    while cur is not None:
        if cur in seen:
            return True
        seen.add(cur)
        cur = parent_of.get(cur)
    return False


class SCSGenerator:
    """Generate SCS context profiles (depth, direct + inherited properties, summary)."""

    def __init__(self, llm=None):
        self.llm = llm
        self.llm_calls = 0

    def generate_profiles(self, concepts: Concepts) -> list[dict]:
        classes = concepts.classes
        parent_map = self._parent_map(concepts)
        depth_map = self._depths(classes, parent_map)
        depth_groups: dict[int, list[str]] = defaultdict(list)
        for name, d in depth_map.items():
            depth_groups[d].append(name)

        domain_props = self._property_map(concepts)
        related = self._related_map(concepts)
        cache: dict[str, dict] = {}

        max_depth = max(depth_groups) if depth_groups else 0
        for depth in range(max_depth + 1):
            batch = []
            for name in depth_groups.get(depth, []):
                batch.append({
                    "class_name": name,
                    "depth": depth,
                    "direct_properties": domain_props.get(name, []),
                    "inherited_properties": self._inherited(name, parent_map, cache),
                    "related_classes": related.get(name, []),
                    "context_summary": "",
                })
            for profile, summary in zip(batch, self._summarize(batch)):
                profile["context_summary"] = summary
                cache[profile["class_name"]] = profile
        return list(cache.values())

    def _parent_map(self, concepts: Concepts) -> dict[str, Optional[str]]:
        pm: dict[str, Optional[str]] = {}
        for parent, child in concepts.class_hierarchy:
            pm[child] = parent
        for c in concepts.classes:
            if c.name and c.parent and c.name not in pm:
                pm[c.name] = c.parent
        for c in concepts.classes:
            pm.setdefault(c.name, None)
        return pm

    def _depths(self, classes, parent_map) -> dict[str, int]:
        depth: dict[str, int] = {}

        def get(name, visited=None):
            if name in depth:
                return depth[name]
            visited = visited or set()
            if name in visited:
                return 0
            visited.add(name)
            parent = parent_map.get(name)
            depth[name] = 0 if not parent else get(parent, visited) + 1
            return depth[name]

        for c in classes:
            if c.name:
                get(c.name)
        return depth

    def _property_map(self, concepts: Concepts) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for p in concepts.object_properties:
            if p.domain:
                out[p.domain].append({"name": p.name, "type": "ObjectProperty", "range": p.range})
        for p in concepts.datatype_properties:
            if p.domain:
                out[p.domain].append({"name": p.name, "type": "DatatypeProperty", "range": p.range})
        return out

    def _related_map(self, concepts: Concepts) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for p in concepts.object_properties:
            if p.domain and p.range:
                if p.range not in out[p.domain]:
                    out[p.domain].append(p.range)
                if p.domain not in out[p.range]:
                    out[p.range].append(p.domain)
        return out

    def _inherited(self, name, parent_map, cache) -> list[dict]:
        inherited: list[dict] = []
        visited: set = set()
        current = parent_map.get(name)
        depth_count = 1
        while current and current not in visited:
            visited.add(current)
            pp = cache.get(current)
            if pp:
                for prop in pp.get("direct_properties", []):
                    inherited.append({"name": prop["name"], "from": current, "depth": depth_count})
                for prop in pp.get("inherited_properties", []):
                    inherited.append({"name": prop["name"], "from": prop["from"],
                                      "depth": prop["depth"] + depth_count})
            current = parent_map.get(current)
            depth_count += 1
        return inherited

    def _summarize(self, profiles: list[dict]) -> list[str]:
        if not profiles:
            return []
        if self.llm is not None:
            desc = []
            for p in profiles:
                direct = ", ".join(d["name"] for d in p["direct_properties"]) or "none"
                inh = ", ".join(f"{i['name']}(from {i['from']})" for i in p["inherited_properties"]) or "none"
                rel = ", ".join(p["related_classes"]) or "none"
                desc.append(f"[{p['class_name']}] direct: {direct} / inherited: {inh} / related: {rel}")
            user = (
                "Summarize each ontology class below in 1-2 sentences "
                "(direct properties, inherited 'from X', related classes).\n\n"
                "## Class profiles\n" + "\n".join(desc) + "\n\n"
                '## Output (JSON only, same order)\n{"summaries": ["...", "..."]}'
            )
            self.llm_calls += 1
            result = invoke_json(self.llm, "You summarize ontology class context.", user)
            summaries = result.get("summaries")
            if isinstance(summaries, list) and len(summaries) == len(profiles):
                return [str(s) for s in summaries]
        return [self._rule_summary(p) for p in profiles]

    @staticmethod
    def _rule_summary(p: dict) -> str:
        name = p["class_name"]
        parts = []
        if p["direct_properties"]:
            parts.append(f"{name} has properties: " + ", ".join(d["name"] for d in p["direct_properties"]))
        if p["inherited_properties"]:
            src: dict[str, list[str]] = {}
            for i in p["inherited_properties"]:
                src.setdefault(i["from"], []).append(i["name"])
            for s, props in src.items():
                parts.append(f"inherits {', '.join(props)} from {s}")
        if p["related_classes"]:
            parts.append("related to " + ", ".join(p["related_classes"]))
        return (". ".join(parts) + ".") if parts else f"{name} class."
