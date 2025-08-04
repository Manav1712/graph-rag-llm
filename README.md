# GraphRAG Knowledge Graph Search: From Physics Textbooks to Intelligent Q&A

## Project Overview

This project demonstrates the power of **Microsoft's GraphRAG framework** to transform traditional physics textbooks into an intelligent, queryable knowledge graph. The system takes OpenStax Physics 2e and College Algebra textbooks and creates a sophisticated search system that can answer complex questions through three different search strategies.

### What Was Built

A **three-tier search system** that can answer questions about physics and mathematics with remarkable accuracy:

- **Drift Search**: Iterative refinement for complex, multi-step queries
- **Local Search**: Precise entity and relationship-focused answers  
- **Global Search**: Holistic understanding across the entire corpus

### The Results Speak for Themselves

The system achieved an **average score of 7.2/10** across 45 evaluations, with some responses scoring as high as **9/10**. Here are some real examples of what the system can do:

## Sample Query Results

| Query Type | Question | Key Response Points | Score | Evaluation |
|------------|----------|-------------------|-------|------------|
| **Drift Search** | "What is the relationship between water and pressure in fluid mechanics?" | • Density and pressure relationships<br>• Buoyancy principles with wood/brass examples<br>• Static fluid behavior<br>• Measurement techniques<br>• Environmental implications | 8/10 | "Excellent domain expertise and comprehensive coverage" |
| **Drift Search** | "How do vectors and scalar products work together in physics?" | • Vector fundamentals (magnitude + direction)<br>• Scalar product mathematical definition<br>• Real-world applications (work calculations)<br>• Orthogonality principles | 9/10 | "Masterful integration of theory and application" |
| **Drift Search** | "Tell me about Blaise Pascal and his contributions to fluid mechanics" | • Pascal's Law formulation<br>• Fluid pressure experiments<br>• Hydrostatics contributions<br>• Legacy and impact | 7/10 | "Solid coverage with good historical context" |
| **Local Search** | "What is the difference between gauge pressure and absolute pressure?" | • Gauge pressure vs atmospheric reference<br>• Absolute pressure vs vacuum reference<br>• Measurement applications<br>• Key differences in values and uses | 8/10 | "Clear technical explanation with practical examples" |
| **Global Search** | "What are the main themes in physics?" | • Fundamental principles overview<br>• Cross-concept relationships<br>• Thematic connections<br>• Domain-wide patterns | 7/10 | "Good thematic analysis across the corpus" |

## Technical Architecture

### Knowledge Graph Construction
The system extracted **90+ entities** and **98 relationships** from the physics textbooks, creating a rich knowledge graph that captures:

- **Core Physics Concepts**: Vectors, scalar products, pressure, density
- **Mathematical Relationships**: Orthogonal vectors, cross products
- **Real-world Applications**: Buoyancy, fluid dynamics, measurement devices
- **Historical Context**: Blaise Pascal's contributions

### Search Engine Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Drift Search  │    │   Local Search  │    │  Global Search  │
│                 │    │                 │    │                 │
│ • Iterative     │    │ • Entity-focused│    │ • Thematic      │
│ • Multi-step    │    │ • Relationship  │    │ • Community     │
│ • Refinement    │    │ • Precise facts │    │ • Cross-domain  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Knowledge Graph│
                    │                 │
                    │ • 90+ Entities  │
                    │ • 98 Relations  │
                    │ • 156 Text Units│
                    └─────────────────┘
```

## Getting Started: Replicate This Project

### Prerequisites
- Python 3.11+
- OpenAI API key
- 8GB+ RAM (for processing large documents)
- Some disk space
- Around $5 to replicate the KG with a corpus of ~300,000 words

### Step 1: Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd graph-rag-llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Step 2: Prepare Your Documents

Place your PDF documents in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy your PDF files here
cp your_physics_textbook.pdf data/raw/
cp your_math_textbook.pdf data/raw/
```

### Step 3: Run GraphRAG Indexing

```bash
# Install GraphRAG if not already installed
pip install graphrag

# Run the indexing process
python -m graphrag.index \
    --input-dir data/raw \
    --output-dir graphrag_workspace/output \
    --settings graphrag_workspace/settings.yaml
```

This process will:
- Extract text from PDFs
- Identify entities and relationships
- Create community clusters
- Generate embeddings
- Build the knowledge graph

### Step 4: Test the Search System

```bash
# Test Drift Search (complex queries)
python src/drift_search.py --query "What is the relationship between water and pressure?"

# Test Local Search (specific facts)
python src/local_search.py --query "Tell me about Blaise Pascal"

# Test Global Search (thematic questions)
python src/global_search.py --query "What are the main themes in physics?"
```

### Step 5: Visualize Your Knowledge Graph

```bash
# Generate visualization data
python load_data.py

# Serve the interactive visualization
python serve_graph.py --port 8000
```

Open `http://localhost:8000/knowledge_graph.html` to explore your knowledge graph interactively.

## Evaluation Results

The system was evaluated across multiple dimensions:

### Overall Performance
- **Average Score**: 7.2/10
- **Best Score**: 9/10 (Density analysis)
- **Total Evaluations**: 45 queries
- **Success Rate**: 93% (42/45 successful responses)

### Category Breakdown
- **Relationship Queries**: 30 evaluations, avg 7.1/10
- **Community Reports**: 3 evaluations, avg 8.0/10  
- **Entity Quality**: 12 evaluations, avg 7.4/10

### Key Strengths Identified
- Clear domain focus on physics and mechanics  
- Well-structured community detection  
- High-quality entity descriptions  
- Logical relationship mappings  

### Areas for Improvement
- Relationship specificity could be enhanced  
- Cross-domain connections need strengthening  
- Entity coverage could be expanded  

## Customization Guide

### Adding New Document Types

1. **Modify the settings file** (`graphrag_workspace/settings.yaml`):
```yaml
entity_extraction:
  model: "gpt-4"
  max_tokens: 4000
  temperature: 0.0

community_detection:
  algorithm: "louvain"
  resolution: 1.0
```

2. **Adjust search parameters** in each search module:
```python
# In src/drift_search.py
drift_params = DRIFTSearchConfig(
    temperature=0,
    max_tokens=12_000,
    primer_folds=1,
    drift_k_followups=3,
    n_depth=3,
)
```

### Extending the Knowledge Graph

Add new entity types by modifying the extraction prompts in `graphrag_workspace/prompts/`:

```python
# Example: Add chemical entities
chemical_entities = [
    "molecules", "atoms", "compounds", "reactions",
    "catalysts", "enzymes", "polymers"
]
```

## Troubleshooting

### Common Issues

**Problem**: "OpenAI API key not found"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your-key-here"
```

**Problem**: "Required file not found"
```bash
# Solution: Run GraphRAG indexing first
python -m graphrag.index --input-dir data/raw --output-dir graphrag_workspace/output
```

**Problem**: Memory errors during processing
```bash
# Solution: Reduce batch size in settings.yaml
processing:
  batch_size: 4  # Reduce from default 8
```

### Performance Optimization

For large documents (>1000 pages):
- Increase `max_tokens` in settings
- Use `gpt-4-turbo-preview` for better performance
- Process documents in smaller chunks

## Future Enhancements

### Planned Improvements
- Multi-language support
- Real-time document updates
- Advanced visualization features
- Integration with external databases
- Mobile-friendly interface

### Research Directions
- Cross-document relationship discovery
- Temporal knowledge graph evolution
- Multi-modal entity extraction (images, diagrams)
- Collaborative knowledge graph building

## Contributing

Contributions are welcome! Please contact **manav172022@gmail.com** to discuss potential contributions, feature requests, or collaboration opportunities.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

---

**Ready to build your own intelligent knowledge graph?** Start with the installation guide above and transform your documents into a queryable knowledge base!

For questions or support, please contact **manav172022@gmail.com**.
