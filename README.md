# RAG + Agentic Requirements Traceability

Automatically recover trace links between requirements and code using
semantic retrieval, a RAG/LLM verifier, and an agentic layer, evaluated
against a published benchmark (CoEST).

## Problem
Trace links connect requirements to the code that implements them. Creating
and maintaining them by hand is costly, and in practice they are often
incomplete or missing. This makes impact analysis, compliance, and
maintenance harder. Automating trace-link recovery addresses this gap.

## Methods (planned)
1. IR baseline (TF-IDF / vector space)
2. Semantic retrieval (nomic-embed-text + ChromaDB via Ollama)
3. RAG + local LLM verification (Llama 3.2 3B)
4. Agentic layer (Proposer + Verifier)

## Status
Phase 0 — scope & data.