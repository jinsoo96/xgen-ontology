"""Build a tiny graph from loose triples, then one-shot GraphRAG over it.

No API key needed — the default EchoLLM returns the fused evidence so you can see
exactly what the synthesis step receives. Pass your own LLM via CallableLLM.
"""
import sys

sys.path.insert(0, "src")

from xgen_ontology import CallableLLM, build_from_triples  # noqa: E402

onto = build_from_triples(
    [
        ("Widget", "instanceOf", "Product"),
        ("Gadget", "instanceOf", "Product"),
        ("Gizmo", "instanceOf", "Product"),
        ("Widget", "hasColor", "Red"),
        ("Gadget", "hasColor", "Blue"),
    ],
    chunks=[
        ("c1", "The Widget is a flagship red product.", ["Widget"]),
        ("c2", "The Gadget ships in blue.", ["Gadget"]),
    ],
)

# (a) zero-key run — EchoLLM echoes the fused evidence
res = onto.search("which products are red")
print("class_seed     :", res.class_seed)
print("relations used :", res.relations)
print("evidence_nodes :", res.evidence_nodes)

# (b) plug a real model
def my_llm(prompt, system=""):
    # call OpenAI / Anthropic / vLLM here; we just echo a canned answer
    return "Widget is the red product."

res = onto.search("which products are red", llm=CallableLLM(my_llm))
print("\nanswer         :", res.answer)
print("cited nodes    :", res.evidence_nodes)
