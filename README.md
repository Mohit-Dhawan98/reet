# REET: Relevance-Engineered Efficient Tokens

**Entity-level semantic compression for LLM context windows**

## Overview

REET is a research project exploring a novel approach to context compression: **entity-level semantic extraction** rather than token or sentence-level filtering.

### The Problem

Current context compression methods operate at:
- **Token level** (LLMLingua-2, LongLLMLingua, QUITO-X) - Drop individual tokens
- **Sentence level** (EXIT, RECOMP) - Select/remove sentences

None operate at the **semantic/entity level** - understanding *what matters* rather than *what words to keep*.

### Our Approach

```
Traditional:  "John Smith, CEO of Acme Corp founded in 1985"
              → "John Smith CEO Acme Corp 1985" (token drops)

REET:         "John Smith, CEO of Acme Corp founded in 1985"
              → {"Acme Corp": ["founded 1985", "CEO: John Smith"]} (entity-facts)
```

REET extracts entities and their associated facts, producing a structured representation that:
1. Preserves semantic relationships
2. Enables better compression ratios
3. Supports both task-agnostic and query-aware modes

## Features

### Two Operating Modes

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| **Task-Agnostic** | Context only | All important entities + facts | General summarization |
| **Query-Aware** | Context + Query | Relevant entities + facts | QA, retrieval |

### Baselines Included

- Truncation (naive baseline)
- LLMLingua-2 (token classification)
- LongLLMLingua (contrastive perplexity)
- GLiNER2 baseline (entity extraction with generic schema)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reet.git
cd reet

# Install dependencies
pip install -e .

# Optional: Install LLMLingua for baseline comparisons
pip install llmlingua
```

## Quick Start

```python
from reet.benchmarks.baselines import CompressorRegistry

# Get a compressor
compressor = CompressorRegistry.get("truncation")

# Compress context
result = compressor.compress(
    context="Your long context here...",
    query="Optional query for query-aware compression",
    target_ratio=0.5  # Keep 50% of tokens
)

print(f"Compressed: {result.compressed_text}")
print(f"Ratio: {result.compression_ratio:.2%}")
```

## Project Structure

```
reet/
├── benchmarks/
│   ├── baselines/       # Compression baselines
│   │   ├── base.py      # BaseCompressor, CompressorRegistry
│   │   ├── truncation.py
│   │   └── llmlingua.py
│   └── scbench/         # SCBench evaluation
├── data/                # Dataset loaders (planned)
└── models/              # REET-Extractor (planned)

scripts/
└── run_scbench.py       # Benchmark runner

docs/
└── RESEARCH_NOTES.md    # Detailed research findings
```

## Benchmarks

### Datasets

| Phase | Dataset | Type |
|-------|---------|------|
| 1 | NaturalQuestions, TriviaQA, HotpotQA, SQuAD | Single-turn QA |
| 2 | LongBench, ZeroSCROLLS | Task-agnostic |
| 3 | SCBench, CoQA | Multi-turn |

### Running Evaluations

```bash
# Quick test
python scripts/run_scbench.py --quick

# Full evaluation
python scripts/run_scbench.py --dataset scbench_kv --limit 50 --compressors truncation
```

## Research Status

This is an active research project. See:
- [VISION.md](VISION.md) - Problem statement and goals
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical design
- [BENCHMARKS.md](BENCHMARKS.md) - Baselines and datasets
- [ROADMAP.md](ROADMAP.md) - Development phases
- [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) - Detailed research findings

## References

### Key Papers

- [LLMLingua-2](https://arxiv.org/abs/2403.12968) - Task-agnostic token compression
- [LongLLMLingua](https://arxiv.org/abs/2310.06839) - Query-aware compression
- [QUITO-X](https://arxiv.org/abs/2408.10497) - Current SOTA on QA tasks
- [EXIT](https://arxiv.org/abs/2412.12559) - Entity-preserving sentence selection
- [GLiNER2](https://arxiv.org/abs/2507.18546) - Schema-driven entity extraction

## License

MIT
