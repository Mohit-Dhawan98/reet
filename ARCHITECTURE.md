# REET Architecture

## Overview

REET-Extractor is a GLiNER-inspired model for entity-level context compression with two operating modes.

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      REET-Extractor                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Formulation:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Mode 1: [P] extract entities [SEP] [Context]            │   │
│  │ Mode 2: [P] for query [SEP] [Query] [SEP] [Context]     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Transformer Encoder                         │   │
│  │              (DeBERTa-v3-base / ModernBERT)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│              ┌────────────┴────────────┐                       │
│              ▼                         ▼                        │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │  Entity Span     │      │   Fact Span      │                │
│  │  Detection Head  │      │  Detection Head  │                │
│  └──────────────────┘      └──────────────────┘                │
│              │                         │                        │
│              └────────────┬────────────┘                       │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Entity-Fact Linking                         │   │
│  │         (Associate facts with their entities)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  Output: {"Entity1": ["fact1", "fact2"], "Entity2": [...]}     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Input Formulation

Similar to GLiNER2, we use a unified input format with special tokens:

| Token | Purpose |
|-------|---------|
| `[P]` | Task prompt prefix |
| `[SEP]` | Separator |
| `[E]` | Entity marker (optional) |
| `[F]` | Fact marker (optional) |

**Task-Agnostic Mode:**
```
[P] Extract all important entities and facts [SEP] {context}
```

**Query-Aware Mode:**
```
[P] Extract entities and facts relevant to: {query} [SEP] {context}
```

### 2. Encoder Backbone

**Options:**
- DeBERTa-v3-base (304M params) - Best balance of size/performance
- ModernBERT-base (149M params) - Faster, recent architecture
- DeBERTa-v3-large (434M params) - Higher quality, slower

**Why not GLiNER2's backbone?**
- GLiNER2 uses 205M custom encoder trained on NER
- We need query-encoding capability GLiNER2 lacks
- DeBERTa has better cross-attention for query-context interaction

### 3. Span Detection Heads

#### Entity Span Head
Identifies entity mentions in text:
- Input: Token embeddings
- Output: BIO tags or span scores
- Architecture: 2-layer MLP + CRF (optional)

#### Fact Span Head
Identifies fact phrases associated with entities:
- Input: Token embeddings + entity representations
- Output: Span scores conditioned on entity
- Architecture: Cross-attention + MLP

### 4. Entity-Fact Linking

Associates detected facts with their corresponding entities:
- Proximity-based heuristics (same sentence)
- Learned linking scores
- Dependency parsing (optional enhancement)

### 5. Output Formatting

Convert entity-fact structure to compressed text:

```python
# Option A: Structured format
{"John Smith": ["CEO of Acme Corp"], "Acme Corp": ["founded 1985"]}

# Option B: Linearized format
"John Smith: CEO of Acme Corp. Acme Corp: founded 1985."

# Option C: Markdown format
"- **John Smith**: CEO of Acme Corp\n- **Acme Corp**: founded 1985"
```

## Training

### Data Requirements

| Mode | Source | Labels |
|------|--------|--------|
| Task-Agnostic | Wikipedia, News | GPT-4 extracts all entities/facts |
| Query-Aware | NQ, TriviaQA, HotpotQA | GPT-4 extracts query-relevant entities/facts |

### Label Generation Pipeline

```python
def generate_labels(context: str, query: str | None) -> dict:
    """Use GPT-4 to generate entity-fact labels."""

    if query is None:
        # Task-agnostic mode
        prompt = f"""Extract all important entities and their facts from:

{context}

Output JSON: {{"entity": ["fact1", "fact2", ...], ...}}"""
    else:
        # Query-aware mode
        prompt = f"""Extract entities and facts needed to answer:
Query: {query}

Context: {context}

Output JSON with only relevant entities/facts."""

    return gpt4_call(prompt)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR (backbone) | 1e-5 |
| LR (heads) | 2e-5 |
| Batch size | 16 |
| Max length | 2048 tokens |
| Epochs | 5 |
| Loss | BCE for spans + Cross-entropy for linking |

### Mixed-Mode Training

Train single model on both modes:
- 50% task-agnostic samples
- 50% query-aware samples
- Mode indicated by input format

## Inference

### Pipeline

```python
def compress(context: str, query: str | None = None) -> CompressedResult:
    # 1. Format input
    if query:
        input_text = f"[P] for query [SEP] {query} [SEP] {context}"
    else:
        input_text = f"[P] extract entities [SEP] {context}"

    # 2. Encode
    embeddings = encoder(input_text)

    # 3. Detect spans
    entity_spans = entity_head(embeddings)
    fact_spans = fact_head(embeddings, entity_spans)

    # 4. Link and format
    entity_facts = link(entity_spans, fact_spans)
    compressed = format_output(entity_facts)

    return CompressedResult(
        compressed_text=compressed,
        original_tokens=len(tokenize(context)),
        compressed_tokens=len(tokenize(compressed)),
        compression_ratio=len(compressed) / len(context)
    )
```

### Complexity

| Operation | Complexity |
|-----------|-----------|
| Encoding | O(n) where n = context length |
| Span detection | O(n) |
| Linking | O(e * f) where e = entities, f = facts |
| **Total** | O(n) - linear in context length |

## Comparison with GLiNER2

| Aspect | GLiNER2 | REET-Extractor |
|--------|---------|----------------|
| Schema | Required (user-provided) | None (auto-extracts) |
| Query support | No | Yes (core feature) |
| Output | Entity spans by type | Entities + their facts |
| Use case | NER | Compression |
| Architecture | Custom 205M encoder | DeBERTa + custom heads |

## Alternative Architectures Considered

### Option B: Seq2Seq (T5-based)

```
Input:  [Query?] Context
Output: JSON of entity-facts
```

**Pros:** Simpler, can generate novel phrasings
**Cons:** Slower inference, hallucination risk

### Option C: Retriever-style

```
Embed entities → Embed query → Retrieve top-k relevant
```

**Pros:** Fast at inference
**Cons:** Requires pre-extracted entity index

### Why We Chose Option A (GLiNER-style)

1. Single forward pass extraction
2. No hallucination (extractive only)
3. Explicit span identification (interpretable)
4. Natural fit for both modes
