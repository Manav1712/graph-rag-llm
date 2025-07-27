# GraphRAG QA System Implementation Guide
## Two Implementation Approaches

---

## Table of Contents

### [PART 1: FULL TWO-AGENT SYSTEM](#part-1-full-two-agent-system)
1. [Project Purpose & Overview](#project-purpose--overview)
2. [System Requirements & Setup](#system-requirements--setup)
3. [Repository Structure & Conventions](#repository-structure--conventions)
4. [Key Concepts: RAG, Knowledge Graphs, Community Detection, Agents](#key-concepts)
5. [End-to-End Pipeline: Overview](#end-to-end-pipeline-overview)
6. [Phase 1: Environment, Data, and Preprocessing](#phase-1-environment-data-and-preprocessing-1–15-h)
7. [Phase 2: Knowledge Graph Construction & Community Detection](#phase-2-knowledge-graph-construction--community-detection-2–25-h)
8. [Phase 3: Community Summarization](#phase-3-community-summarization-1-h)
9. [Phase 4: Baseline Vector RAG Implementation](#phase-4-baseline-vector-rag-implementation-05–1-h)
10. [Phase 5: Global GraphRAG Search Implementation](#phase-5-global-graphrag-search-implementation-15–2-h)
11. [Phase 6: Orchestrator Agent Logic](#phase-6-orchestrator-agent-logic-05–1-h)
12. [Phase 7: Evaluation Harness](#phase-7-evaluation-harness-2–25-h)
13. [Phase 8: Logging, Cost Tracking, and Basic Guardrails](#phase-8-logging-cost-tracking-and-basic-guardrails-05-h)
14. [Phase 9: Visualization (Optional/Stretch)](#phase-9-visualization-optionalstretch-05–1-h)
15. [Phase 10: Packaging, Documentation, and Next Steps](#phase-10-packaging-documentation-and-next-steps-05–1-h)
16. [Time Estimates, Tips, and Pitfalls](#time-estimates-tips-and-pitfalls)

### [PART 2: SIMPLIFIED SINGLE-AGENT SYSTEM](#part-2-simplified-single-agent-system)
1. [Simplified Project Overview](#simplified-project-overview)
2. [Simplified System Requirements](#simplified-system-requirements)
3. [Simplified Repository Structure](#simplified-repository-structure)
4. [Simplified Implementation Phases](#simplified-implementation-phases)
5. [Simplified Time Estimates](#simplified-time-estimates)

---

# PART 1: FULL TWO-AGENT SYSTEM

## 1. Project Purpose & Overview

**Goal:**  
- Build a modular, reproducible QA system over Physics 2e and College Algebra content, leveraging state-of-the-art GraphRAG (Graph-augmented Retrieval-Augmented Generation) and agent orchestration.
- Demonstrate rapid adoption of advanced retrieval, automatic knowledge graph construction, and agent workflows by comparing global, community-based retrieval against vanilla vector-based RAG for both local and synthesis-style queries.

**Deliverable:**  
- Modular Python project (not just a single notebook)
- Runs start-to-finish in ≤ 16 hours
- Documents all decisions and enables future extension

---

## 2. System Requirements & Setup

**Hardware:**  
- Standard laptop or cloud VM (≥8 GB RAM, ≥30 GB disk)
- (Optional) GPU for faster embedding, not essential

**Software/Dependencies:**  
- Python 3.10+
- `openai` (OpenAI Python SDK)
- `graphrag` (GraphRAG official OSS library)
- `sentence-transformers`
- `lancedb`
- `pandas`, `pyarrow`
- `networkx` (optional, for visualization)
- `matplotlib`, `pyvis` (optional)
- `langchain` or similar for chunking
- `tiktoken`
- `python-dotenv`
- `loguru`
- (Optional) `graspologic` (if hand-running Leiden)
- `jupyterlab` or `notebook`
- `pytest` (optional, for unit testing)

**External Resources:**  
- OpenAI API key
- Downloaded PDFs of OpenStax Physics 2e and College Algebra
- (Optional) HuggingFace account if using a local LLM

**Environment setup:**
```bash
# Clone the project repo
git clone <repo_url>
cd <project_root>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 3. Repository Structure & Conventions

```
project_root/
  src/
    indexer.py          # Chunking, extraction, graph, community detection
    summarizer.py       # Community summary LLM calls
    vector_rag.py       # Baseline embedding-based retrieval
    graph_rag.py        # GraphRAG global search and interfaces
    agent.py            # Orchestrator (routes queries to search mode)
    eval.py             # LLM-as-judge, table output, metrics
    utils.py            # Logging, token counting, helper functions
  data/
    raw/                # PDFs, source docs
    chunks/             # Preprocessed text chunks
    kg/                 # Parquet or .gpickle graph files
    communities/        # Community assignment, summaries
    vector_db/          # LanceDB index files
    outputs/            # Query results, evaluation logs
  config/
    settings.yaml       # Model names, token budgets, chunk size
    prompts/            # YAML/text prompt templates
  notebooks/
    demo.ipynb          # Top-level workflow for testing
  docs/
    architecture.png    # System/data flow diagram
    README.md           # Project overview, quickstart
  requirements.txt
  .env.example
```

All module imports are relative (e.g., from src.indexer import ...)

Data artifacts versioned if small, otherwise .gitignore large outputs

README.md contains step-by-step quickstart and troubleshooting

## 4. Key Concepts: RAG, Knowledge Graphs, Community Detection, Agents

**RAG (Retrieval-Augmented Generation)**
Uses embedding-based similarity to pull context for LLM question-answering

Baseline: split docs → embed → store → retrieve via nearest neighbors

**Knowledge Graphs (KG)**
Graph where nodes = entities/concepts, edges = relationships

Built from LLM extraction prompts or hand-curation

**Community Detection (Leiden)**
Partition graph into "communities" of closely connected nodes

In GraphRAG, used for scalable, hierarchical summarization and retrieval

**Agents**
Modular "workers" (Python classes or scripts) that handle specific tasks

For this project:
- Indexer Agent: offline, builds graph/summaries
- Query Orchestrator Agent: online, routes queries and composes final answer

## 5. End-to-End Pipeline: Overview

**Indexing/Offline:**
- Chunk PDFs
- Extract entities & relationships via LLM
- Build and persist KG
- Detect communities (Leiden, level 0; multi-level stretch)
- Summarize each community using LLM

**Online/Query-time:**
- Receive user query
- Orchestrator Agent decides on baseline or GraphRAG
- Run search (vector, global)
- Generate answer with citations
- (Optional) LLM-based evaluation of answer quality

## 6. Phase 1: Environment, Data, and Preprocessing (1–1.5 h)

Get OpenAI API key, save to .env

Download PDFs, save to data/raw/

Chunk PDFs

Use langchain.text_splitter or a custom chunker

Target: 600 tokens/chunk, 100 token overlap

Save chunks to data/chunks/physics.parquet etc.

**Tips:**
- Remove tables/images for cleaner input
- Store metadata (source file, page number, section)

**Code sketch:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_pdf(pdf_path, out_path, chunk_size=600, overlap=100):
    # Load, preprocess, split, and save to Parquet
    ...
```

## 7. Phase 2: Knowledge Graph Construction & Community Detection (2–2.5 h)

**Entity/Relationship Extraction**

For each chunk, LLM prompt:
"Extract all physics concepts, entities, formulas, relationships, and units in JSON triples."

Aggregate results into a node/edge table

**KG Assembly**

Build a directed graph (in-memory or write directly to Parquet)

Schema:
- Nodes: entity_id, name, type, source_chunk
- Edges: source_entity, target_entity, relation, evidence_chunk

**Community Detection**

Use graphrag API:
```python
graphrag.index.build_index(chunks=..., out_dir=...)
```

Save community membership to Parquet (data/communities/)

## 8. Phase 3: Community Summarization (1 h)

For each community, generate a "community report" with a single LLM call:
"Summarize the following community of physics entities, listing all key relationships and claims..."

Save summaries to Parquet:
- community_id, summary_text, level, nodes_included

This step is fully automated in graphrag, or you can manually call GPT-4 for a few communities as a fallback.

## 9. Phase 4: Baseline Vector RAG Implementation (0.5–1 h)

Embed all text chunks

Use sentence-transformers/all-MiniLM-L6-v2 for fast, cheap embeddings

Save to LanceDB index in data/vector_db/

**Search Function**

Given a query, embed and retrieve top-k chunks via similarity

Pass to GPT-4 with a simple QA prompt:
"Answer this question using only the provided context. Cite all supporting text."

## 10. Phase 5: Global GraphRAG Search Implementation (1.5–2 h)

**Search Function**

Given a query, load all (or selected) community summaries (from Parquet)

Shuffle or batch summaries for diversity

Use map-reduce pattern:
- Map: LLM answers Q for each summary, rates confidence/relevance
- Reduce: aggregate top-scoring snippets, feed to LLM for synthesis

**graphrag API**
```python
graphrag.query.global_search(query=..., summaries=..., config=...)
```

Set token limits and concurrency in config/settings.yaml

Capture all intermediate LLM outputs for logging and debugging

## 11. Phase 6: Orchestrator Agent Logic (0.5–1 h)

**Heuristic:**

If query length >12 or contains "summarize/compare/explain", route to GraphRAG global search

Else, use baseline vector RAG

**Class skeleton:**
```python
class OrchestratorAgent:
    def __init__(self, ...):
        ...
    def route_query(self, query):
        if self.is_global_query(query):
            return self.graphrag_global(query)
        else:
            return self.vector_baseline(query)
```

**Logging:**

Log which path was chosen and why, for every query

## 12. Phase 7: Evaluation Harness (2–2.5 h)

LLM-as-judge evaluation

Compare GraphRAG vs vector RAG performance

Generate metrics and tables

## 13. Phase 8: Logging, Cost Tracking, and Basic Guardrails (0.5 h)

Implement comprehensive logging

Track API costs

Add basic input validation

## 14. Phase 9: Visualization (Optional/Stretch, 0.5–1 h)

**Graph Visualization**

Use networkx and matplotlib to draw communities

Highlight nodes/edges cited in top answers

**Community Structure**

Plot histogram of community sizes

Visualize answer provenance (which communities contributed to which Qs)

## 15. Phase 10: Packaging, Documentation, and Next Steps (0.5–1 h)

**README**

Quickstart instructions

Pipeline overview and architecture diagram

List known issues and next steps

**Docs**

architecture.png: high-level flowchart

evaluation.md: win-rate table, brief observations

**Stretch/Future**

DRIFT/local search mode

Multi-level community hierarchy

SQLite/Neo4j storage

RL agent selection

## 16. Time Estimates, Tips, and Pitfalls

| Phase | Base Estimate | Common Pitfalls |
|-------|---------------|-----------------|
| Env & Data | 1 h | Chunk sizes too small/large—test first |
| KG Build | 2.5 h | Extraction hallucination—QA 10 chunks first |
| Communities | 1 h | Over-splitting—tune Leiden resolution |
| Summaries | 1 h | LLM rate limits, summarization drift |
| Vector RAG | 1 h | Embeddings mismatch, wrong model |
| Global Search | 2 h | Map step OOM (reduce tokens or batch) |
| Orchestrator | 1 h | Routing too brittle—tweak heuristics |
| Eval | 2.5 h | LLM-judge variance, double-run edge cases |
| Logging/Guard | 0.5 h | Missed moderation, keep checks simple |
| Viz/Docs | 1 h | Time sink—deprioritize if under time |

**Total Base: 10–11 h; Stretch/Future: up to 16 h.**

---

# PART 2: SIMPLIFIED SINGLE-AGENT SYSTEM

## Simplified Project Overview

**Goal:**  
- Build a focused GraphRAG QA system with emphasis on knowledge graph quality
- Single agent that can use both GraphRAG and vector RAG methods
- Simplified orchestration with robust knowledge graph foundation

**Deliverable:**  
- Working GraphRAG system with single agent
- Knowledge graph construction as primary focus
- Basic comparison between GraphRAG and vector RAG
- Runs start-to-finish in ≤ 10 hours

## Simplified System Requirements

**Hardware:**  
- Standard laptop or cloud VM (≥8 GB RAM, ≥30 GB disk)

**Software/Dependencies:**  
- Python 3.10+
- `openai` (OpenAI Python SDK)
- `graphrag` (GraphRAG official OSS library)
- `sentence-transformers`
- `lancedb`
- `pandas`, `pyarrow`
- `networkx`
- `langchain` or similar for chunking
- `tiktoken`
- `python-dotenv`
- `loguru`

**External Resources:**  
- OpenAI API key
- Downloaded PDFs of OpenStax Physics 2e and College Algebra

## Simplified Repository Structure

```
project_root/
  src/
    indexer.py          # Knowledge graph construction (FOCUS)
    graph_rag.py        # GraphRAG search implementation
    vector_rag.py       # Baseline vector RAG
    agent.py            # Single agent with routing logic
    utils.py            # Logging, helper functions
  data/
    raw/                # PDFs, source docs
    chunks/             # Preprocessed text chunks
    kg/                 # Knowledge graph files
    communities/        # Community assignment, summaries
    vector_db/          # LanceDB index files
    outputs/            # Query results
  config/
    settings.yaml       # Model names, token budgets
    prompts/            # Prompt templates
  notebooks/
    demo.ipynb          # Working demo
  requirements.txt
  .env.example
```

## Simplified Implementation Phases

### Phase 1: Knowledge Graph Foundation (4 hours) - PRIMARY FOCUS
**Entity/Relationship Extraction**
- Robust LLM-based extraction with validation
- Multiple extraction passes for different entity types
- Quality checks and cleaning

**Graph Construction**
- Build directed graph with proper schema
- Graph validation and integrity checks
- Efficient storage and persistence

**Community Detection**
- Tune Leiden parameters for meaningful communities
- Validate community quality
- Handle edge cases (isolated nodes, etc.)

**Community Summarization**
- Generate comprehensive community summaries
- Quality validation of summaries

### Phase 2: Vector RAG Baseline (1 hour)
- Simple embedding-based retrieval
- Basic similarity search
- Simple QA with context

### Phase 3: GraphRAG Implementation (2 hours)
- Global search over community summaries
- Map-reduce pattern (simplified if needed)
- Integration with knowledge graph

### Phase 4: Single Agent (2 hours)
- Agent that can use both GraphRAG and vector RAG
- Simple routing logic based on query type
- Basic logging and transparency

### Phase 5: Demo & Testing (1 hour)
- Working end-to-end system
- Comparison examples
- Basic evaluation

## Simplified Time Estimates

| Phase | Estimate | Focus Area |
|-------|----------|------------|
| Knowledge Graph | 4 h | **PRIMARY FOCUS** |
| Vector RAG | 1 h | Baseline implementation |
| GraphRAG | 2 h | Core GraphRAG functionality |
| Single Agent | 2 h | Simple routing logic |
| Demo & Testing | 1 h | End-to-end validation |

**Total: 10 hours**

## Key Simplifications

### **Single Agent Design:**
```python
class GraphRAGAgent:
    def __init__(self):
        self.vector_rag = VectorRAG()
        self.graph_rag = GraphRAG()
        self.knowledge_graph = KnowledgeGraph()
    
    def answer_query(self, query):
        # Simple decision logic
        if self.is_complex_query(query):
            return self.graph_rag.search(query)
        else:
            return self.vector_rag.search(query)
    
    def is_complex_query(self, query):
        # Simple heuristics
        return len(query.split()) > 10 or any(word in query.lower() 
                for word in ['compare', 'explain', 'summarize', 'relationship'])
```

### **Removed Components:**
- Complex multi-agent orchestration
- Advanced evaluation harness
- Visualization (optional)
- Cost tracking
- Advanced guardrails
- Multi-level community hierarchy

### **Focus Areas:**
- **Knowledge graph quality** - robust entity extraction and graph construction
- **Community detection** - meaningful community identification
- **GraphRAG functionality** - working global search
- **Simple agent logic** - basic but effective routing

This simplified approach ensures you deliver the core GraphRAG innovation while keeping the scope manageable and focusing on the knowledge graph foundation.

