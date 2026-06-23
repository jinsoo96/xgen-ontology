from xgen_ontology import CallableLLM, build_from_csv, build_from_triples


def test_from_triples_class_enumeration_and_relations():
    onto = build_from_triples(
        [("Widget", "instanceOf", "Product"),
         ("Gadget", "instanceOf", "Product"),
         ("Widget", "hasColor", "Red")],
        chunks=[("c1", "Widget is a red product", ["Widget"])],
    )
    r = onto.search("list every Product")
    # class enumeration seed lists both instances
    assert "Widget" in r.class_seed and "Gadget" in r.class_seed
    # honest evidence_nodes are answer-cited only
    assert all(isinstance(n, str) for n in r.evidence_nodes)


def test_search_uses_injected_llm():
    captured = {}

    def fake(prompt, system=""):
        captured["prompt"] = prompt
        return "The answer mentions Widget."

    onto = build_from_triples([("Widget", "instanceOf", "Product"), ("Widget", "hasColor", "Red")])
    r = onto.search("what is Widget", llm=CallableLLM(fake))
    assert r.answer == "The answer mentions Widget."
    assert "Widget" in captured["prompt"]
    assert "Widget" in r.evidence_nodes


def test_build_then_search_csv():
    onto = build_from_csv({
        "products": "product_id,name,color_id\n1,Widget,10\n2,Gadget,20",
        "colors": "color_id,name\n10,Red\n20,Blue",
    })
    r = onto.search("colors")
    assert r.class_seed  # 'Colors' class enumerated
