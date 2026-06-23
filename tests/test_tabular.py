from xgen_ontology import build_from_csv
from xgen_ontology.build.tabular import analyze_tables


def _docs(csv_map):
    return {n: [{"chunk_id": f"{n}#0", "chunk_text": t, "chunk_index": 0}] for n, t in csv_map.items()}


def test_star_schema_fk_and_instances():
    onto = build_from_csv({
        "products": "product_id,name,color_id\n1,Widget,10\n2,Gadget,20",
        "colors": "color_id,name\n10,Red\n20,Blue",
    })
    assert {c.name for c in onto.concepts.classes} == {"Products", "Colors"}
    # FK products.color_id -> colors becomes an object property
    assert any(p.domain == "Products" and p.range == "Colors" for p in onto.concepts.object_properties)
    # FK value is resolved to the target instance label, not the raw id
    rels = {(r.subject, r.object) for r in onto.relations}
    assert ("Widget", "Red") in rels and ("Gadget", "Blue") in rels


def test_column_type_inference():
    schema = analyze_tables(_docs({"t.csv": "id,price,when\n1,9.5,2020-01-01\n2,8.0,2020-02-02"}))
    types = schema["tables"]["t.csv"]["column_types"]
    assert types["price"] == "xsd:decimal"
    assert types["when"] == "xsd:date"


def test_fact_table_kept_as_schema_only():
    # a junction/fact table (2 FKs, many rows) should not be instantiated
    rows = "\n".join(f"{i},{i%3+1},{i%2+1}" for i in range(300))
    onto = build_from_csv({
        "sales": "sale_id,product_id,color_id\n" + rows,
        "products": "product_id,name\n1,A\n2,B\n3,C",
        "colors": "color_id,name\n1,Red\n2,Blue",
    })
    inst_classes = {i.class_name for i in onto.instances}
    assert "Sales" not in inst_classes      # fact table: schema only
    assert "Products" in inst_classes and "Colors" in inst_classes
    assert any(c.name == "Sales" for c in onto.concepts.classes)  # class still declared
