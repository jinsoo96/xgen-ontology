# Changelog

## 0.2.0

- **Ingestion** so it works end-to-end from raw documents:
  - `parse` — `extract_text` / `load_documents`: txt/md/rst/json/html (zero-dep), csv/tsv kept
    as raw table text; pdf/docx/xlsx via the new `[files]` extra.
  - `chunk` — `chunk_text` / `chunk_document`: boundary-aware (paragraph→sentence→char) windows
    with overlap and stable chunk ids for provenance/search.
  - `build_from_files(paths)` and `build_from_text(text)`; raw prose is auto-chunked in the
    pipeline (tables are never chunked). `OntologyBuilder(chunk=, chunk_size=, chunk_overlap=)`.
- **License: MIT © jinsoo96** (was unset).

## 0.1.0

Initial extraction of the production ontology build + search logic as a
backend-agnostic library.

**Build** (documents/tables → a clean knowledge graph):
- `build_from_csv` / `build_from_csv_files` — deterministic table → ontology, no LLM:
  table→Class, FK→ObjectProperty (same-name / normalized-name / value-overlap), column→DataProperty,
  dimension rows→instances, large fact/junction tables kept schema-only.
- `build_from_documents` — LLM extraction (schema + instances per chunk batch, source-tagged),
  with a junk filter; mixes table + text inputs.
- Cleaning stages, each independently importable: `resolve_entities` (entity resolution),
  `govern_predicates` / `normalize_predicate` (predicate governance), `Deduplicator` +
  `cluster_by_cosine` (rule + LLM + embedding dedup), `clean_hierarchy` (genuine is-a only,
  cycle-breaking), `SCSGenerator` (property inheritance + context profiles).
- `review_quality` — completeness / integrity / grounding / shape score (in-memory, no SPARQL).
- `louvain_communities` / `detect_communities` — pure-Python Louvain clustering.
- `to_turtle` (zero-dep) and `to_owl_xml` (optional rdflib) emit.

**Search** (one-shot GraphRAG):
- `Ontology.search` / `GraphRAG` — fuse vector/lexical + graph label-linking + class
  enumeration + HippoRAG 1-hop with MMR diversity and adaptive top-k; single synthesis;
  honest `evidence_nodes`. Language-neutral default prompt (overridable).

**Backends**:
- Zero-infra `InMemoryGraph` / `InMemoryVector` / `InMemoryGraphSink` (BM25, CJK n-grams).
- `SparqlGraph` — read + write any SPARQL 1.1 store (Fuseki/GraphDB/Blazegraph/Virtuoso),
  stdlib-only; `fuseki(base, dataset)` convenience.

`dependencies = []` core; rdflib/kiwipiepy/qdrant-client are optional extras behind protocols.
`Ontology.from_triples` for the search-only path. pytest suite; build + search examples.
