"""Document -> ontology extraction (LLM).

For each batch of chunks the LLM returns schema (classes / object & datatype
properties / hierarchy) *and* instances (entities / relations / data values) in one
call, tagged back to their source chunk ids. A junk filter skips machine dumps
(base64 / degenerate tokens) before spending a call; merging is conservative
(reuse an existing class only for the *same* concept — synonym folding happens
later in :mod:`.dedup`). Predicate governance runs at the end.

The prompt is overridable for any language / domain via ``system_prompt`` /
``user_template``; the default is English-neutral.
"""
from __future__ import annotations

import base64
import re

from ..llm import invoke_json
from ..models import Class, Concepts, DataProperty, DataValue, Instance, ObjectProperty, Relation
from .govern import govern_predicates

_CJK = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏぀-ヿ一-鿿㐀-䶿]")

_SYSTEM = (
    "You are a knowledge-graph engineer. Extract schema and instances from a document.\n"
    "Principles:\n"
    "1. A class is a recurring type/category; an instance is a concrete individual.\n"
    "2. Extract every proper noun (organization, person, law, product, ...) as an instance.\n"
    "3. Extract every figure/date/ratio as a data value — losing numbers loses knowledge.\n"
    "4. Do not make generic words ('document', 'information', 'data') into classes.\n"
    "5. Extract only relations stated in the text. Do not infer.\n"
    "6. Never lump distinct concepts into one class; concrete individuals are instances, not classes."
)

_USER_TEMPLATE = """Extract an ontology schema and instances from the document.

## Domain: {domain}
## Document: {doc}
{schema_context}

## Document content
{text}
{source_instruction}
## Output (JSON only)
{{
  "classes": [{{"name": "...", "description": "...", "parent": "parent class or null"}}],
  "object_properties": [{{"name": "...", "domain": "...", "range": "..."}}],
  "datatype_properties": [{{"name": "...", "display_name": "...", "domain": "...", "range": "xsd:string|xsd:integer|xsd:decimal|xsd:date|xsd:boolean"}}],
  "entities": [{{"entity": "...", "class": "...", "type": "INSTANCE"{src_field}}}],
  "relations": [{{"subject": "...", "predicate": "...", "object": "...", "predicate_type": "ObjectProperty"{src_field}}}],
  "data_values": [{{"entity": "...", "property": "...", "value": "...", "value_type": "xsd:string"{src_field}}}]
}}"""


class DocumentExtractor:
    """Extract a typed ontology from chunked documents using an LLM."""

    def __init__(self, llm, *, domain: str = "", max_text_len: int = 10000,
                 system_prompt: str = _SYSTEM, user_template: str = _USER_TEMPLATE):
        self.llm = llm
        self.domain = domain
        self.max_text_len = max_text_len
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.llm_calls = 0

    def extract(
        self, documents: dict[str, list[dict]],
    ) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
        concepts = Concepts()
        seen_c: set[str] = set()
        seen_op: set[str] = set()
        seen_dp: set[str] = set()
        seen_h: set = set()
        instances: list[Instance] = []
        relations: list[Relation] = []
        data_values: list[DataValue] = []

        for doc, chunks in documents.items():
            for batch in self._batches(chunks):
                r = self._extract_batch(doc, batch)
                if not r:
                    continue
                dc, di, dr, dv = r
                for c in dc.classes:
                    if c.name and c.name not in seen_c:
                        concepts.classes.append(c)
                        seen_c.add(c.name)
                for op in dc.object_properties:
                    if op.name and op.name not in seen_op:
                        concepts.object_properties.append(op)
                        seen_op.add(op.name)
                for dp in dc.datatype_properties:
                    if dp.name and dp.name not in seen_dp:
                        concepts.datatype_properties.append(dp)
                        seen_dp.add(dp.name)
                for edge in dc.class_hierarchy:
                    if edge not in seen_h:
                        concepts.class_hierarchy.append(edge)
                        seen_h.add(edge)
                instances.extend(di)
                relations.extend(dr)
                data_values.extend(dv)

        govern_predicates(relations, concepts.object_properties)
        return concepts, instances, relations, data_values

    def _batches(self, chunks: list[dict]) -> list[list[dict]]:
        batches: list[list[dict]] = []
        cur: list[dict] = []
        cur_len = 0
        for ch in sorted(chunks, key=lambda c: c.get("chunk_index", 0)):
            tl = len(ch.get("chunk_text", ""))
            if cur and cur_len + tl > self.max_text_len:
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(ch)
            cur_len += tl
        if cur:
            batches.append(cur)
        return batches

    def _extract_batch(self, doc: str, batch: list[dict]):
        chunk_ids = [c.get("chunk_id", "") for c in batch]
        parts = []
        for c in batch:
            cid, text = c.get("chunk_id", ""), c.get("chunk_text", "")
            parts.append(f"[CHUNK:{cid}]\n{text}" if cid else text)
        combined = "\n\n".join(parts)
        if not combined.strip() or not _is_extractable(combined):
            return None
        text = combined[:16000]

        result = invoke_json(self.llm, self.system_prompt, self._user(doc, text, chunk_ids))
        self.llm_calls += 1
        if not result:
            return None

        fallback = [cid for cid in chunk_ids if cid]
        classes = [Class(name=c.get("name", ""), description=c.get("description", ""),
                         parent=c.get("parent") or None, source_chunks=c.get("source_chunks", fallback))
                   for c in result.get("classes", []) if c.get("name")]
        class_names = {c.name for c in classes}
        prop_names = ({p.get("name") for p in result.get("object_properties", []) if p.get("name")}
                      | {p.get("name") for p in result.get("datatype_properties", []) if p.get("name")})

        hierarchy: list[tuple[str, str]] = []
        for c in result.get("classes", []):
            name, parent = c.get("name"), c.get("parent")
            if not name or not parent or parent == name:
                continue
            if parent in prop_names and parent not in class_names:
                continue  # a property mislabeled as a parent -> not is-a
            hierarchy.append((parent, name))

        concepts = Concepts(
            classes=classes,
            object_properties=[ObjectProperty(name=p.get("name", ""), domain=p.get("domain", ""),
                                              range=p.get("range", ""))
                               for p in result.get("object_properties", []) if p.get("name")],
            datatype_properties=[DataProperty(name=p.get("name", ""), display_name=p.get("display_name", ""),
                                              domain=p.get("domain", ""), range=p.get("range", "xsd:string"))
                                 for p in result.get("datatype_properties", []) if p.get("name")],
            class_hierarchy=hierarchy,
        )
        instances = [Instance(name=e.get("entity", ""), class_name=e.get("class", ""),
                              source_chunks=e.get("source_chunks") or fallback)
                     for e in result.get("entities", []) if e.get("entity")]
        relations = [Relation(subject=r.get("subject", ""), predicate=r.get("predicate", ""),
                              object=r.get("object", ""), predicate_type=r.get("predicate_type", "ObjectProperty"),
                              source_chunks=r.get("source_chunks") or fallback)
                     for r in result.get("relations", []) if r.get("subject") and r.get("predicate")]
        data_values = [DataValue(entity=d.get("entity", ""), property=d.get("property", ""),
                                 value=d.get("value", ""), value_type=d.get("value_type", "xsd:string"),
                                 source_chunks=d.get("source_chunks") or fallback)
                       for d in result.get("data_values", []) if d.get("entity") and d.get("property")]
        return concepts, instances, relations, data_values

    def _user(self, doc: str, text: str, chunk_ids: list[str]) -> str:
        valid = [cid for cid in chunk_ids if cid]
        src_instruction = ""
        src_field = ""
        if valid:
            src_instruction = (f"\n- source_chunks: record the chunk_id each item came from. "
                               f"Available: {valid}\n")
            src_field = ', "source_chunks": ["chunk_id"]'
        return self.user_template.format(
            domain=self.domain or "auto-detect", doc=doc, schema_context="",
            text=text, source_instruction=src_instruction, src_field=src_field,
        )


def _looks_base64(s: str) -> bool:
    if len(s) < 100 or re.search(r"\s", s):
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", s) or len(s) % 4 != 0:
        return False
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def _is_extractable(text: str) -> bool:
    """False only for obvious machine junk; numbers/tables/short text/CJK pass."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _CJK.search(stripped):
        return True
    if len(stripped) >= 100 and not re.search(r"\s", stripped):
        uniq = len(set(stripped))
        if uniq >= 16 and _looks_base64(stripped):
            return False
        if uniq <= 8:
            return False
    if not any(c.isalnum() for c in stripped):
        return False
    return True
