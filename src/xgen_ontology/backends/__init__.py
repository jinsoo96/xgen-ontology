"""Swappable backends: zero-infra in-memory, and a generic SPARQL 1.1 adapter."""
from .memory import InMemoryGraph, InMemoryGraphSink, InMemoryVector
from .sparql import SparqlGraph

__all__ = ["InMemoryGraph", "InMemoryVector", "InMemoryGraphSink", "SparqlGraph"]
