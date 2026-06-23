"""xgen-ontology — backend-agnostic ontology / knowledge-graph toolkit.

Build a clean knowledge graph from documents or tables, then search it with
one-shot GraphRAG. Zero infra (pure-Python in-memory), zero lock-in (any SPARQL
store), zero hard deps in the core.

Quickstart — deterministic table -> ontology, no LLM, no infra::

    from xgen_ontology import build_from_csv
    onto = build_from_csv({
        "products": "id,name,color_id\\n1,Widget,10\\n2,Gadget,20",
        "colors":   "color_id,name\\n10,Red\\n20,Blue",
    })
    print(onto.stats())
    print(onto.search("what color is Widget").answer)   # EchoLLM by default
    onto.to_turtle()                                      # serialize to RDF
"""
from .build.community import detect_communities, louvain_communities
from .build.dedup import Deduplicator, cluster_by_cosine
from .build.emit import to_owl_xml, to_rdf_triples, to_turtle
from .build.govern import govern_predicates, normalize_predicate
from .build.hierarchy import SCSGenerator, clean_hierarchy
from .build.pipeline import OntologyBuilder
from .build.quality import review_quality
from .build.resolve import resolve_entities
from .build.tabular import analyze_tables, build_from_tables
from .backends.memory import InMemoryGraph, InMemoryGraphSink, InMemoryVector
from .backends.sparql import SparqlGraph, fuseki
from .facade import (build_from_csv, build_from_csv_files, build_from_documents,
                     build_from_triples, rows_to_csv)
from .llm import CallableLLM, EchoLLM
from .models import (BuildReport, Chunk, Class, Concepts, DataProperty, DataValue,
                     Instance, Node, ObjectProperty, RDFTriple, Relation, SearchResult)
from .ontology import Ontology
from .protocols import LLM, Embedder, GraphSink, GraphStore, Morphology, VectorStore
from .search.oneshot import GraphRAG
from .text import BM25, safe_uri, tokenize

__version__ = "0.1.0"

__all__ = [
    # facade
    "build_from_documents", "build_from_csv", "build_from_csv_files", "build_from_triples",
    "rows_to_csv", "OntologyBuilder", "Ontology",
    # search
    "GraphRAG",
    # build stages
    "analyze_tables", "build_from_tables", "resolve_entities", "Deduplicator",
    "cluster_by_cosine", "govern_predicates", "normalize_predicate", "clean_hierarchy",
    "SCSGenerator", "review_quality", "detect_communities", "louvain_communities",
    "to_rdf_triples", "to_turtle", "to_owl_xml",
    # backends
    "InMemoryGraph", "InMemoryVector", "InMemoryGraphSink", "SparqlGraph", "fuseki",
    # llm
    "EchoLLM", "CallableLLM",
    # models
    "Class", "ObjectProperty", "DataProperty", "Concepts", "Instance", "Relation",
    "DataValue", "Node", "Chunk", "RDFTriple", "SearchResult", "BuildReport",
    # protocols
    "LLM", "GraphStore", "VectorStore", "GraphSink", "Morphology", "Embedder",
    # text
    "BM25", "tokenize", "safe_uri",
    "__version__",
]
