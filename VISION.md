# REET Vision: Entity-Level Semantic Compression

## The Problem

LLMs have limited context windows. As applications grow more complex (multi-document QA, long conversations, RAG pipelines), we need to compress context while preserving the information that matters.

### Current Approaches Fall Short

| Approach | Method | Limitation |
|----------|--------|------------|
| **Token-level** | Drop low-importance tokens | Loses semantic coherence |
| **Sentence-level** | Select/remove sentences | Too coarse, includes noise |
| **Soft compression** | Learn latent representations | Requires model fine-tuning |

**The fundamental question these methods ask:**
> "Which tokens/sentences should I keep?"

**The question we should ask:**
> "What entities and facts matter?"

## Our Vision

### Entity-Level Semantic Compression

Instead of filtering text, **extract the semantic structure**:

```
Input Context:
"John Smith, the CEO of Acme Corporation which was founded in 1985
in San Francisco, announced yesterday that the company's Q3 revenue
reached $5.2 million, representing a 15% increase over last year."

Query: "When was Acme founded?"

Token-level (LongLLMLingua):
"John Smith CEO Acme Corporation founded 1985 San Francisco
announced company Q3 revenue $5.2 million 15% increase"

Sentence-level (EXIT):
"John Smith, the CEO of Acme Corporation which was founded in 1985
in San Francisco, announced yesterday..."

REET Entity-Level:
{
  "Acme Corporation": [
    "founded in 1985",
    "headquartered in San Francisco",
    "CEO: John Smith",
    "Q3 revenue: $5.2M (+15% YoY)"
  ]
}

For query "When was Acme founded?":
{
  "Acme Corporation": ["founded in 1985"]
}
```

### Two Operating Modes

#### Mode 1: Task-Agnostic
Given any text, extract all important entities and their facts.

**Use cases:**
- General document summarization
- Knowledge base construction
- Pre-processing for downstream tasks

#### Mode 2: Query-Aware
Given text + query, extract only entities/facts relevant to answering the query.

**Use cases:**
- Question answering
- Retrieval-augmented generation
- Conversational AI

### Why Entity-Level?

1. **Semantic Preservation**: Entities and their relationships capture meaning better than token bags
2. **Higher Compression**: Focus on what matters, not word selection
3. **Structured Output**: Can be easily formatted, filtered, or further processed
4. **Interpretability**: Clear what was kept and why

## The Gap in Current Research

After surveying the SOTA landscape:

| Method | Level | Preserves Entities? | Preserves Relations? |
|--------|-------|--------------------|--------------------|
| LLMLingua-2 | Token | Partially | No |
| LongLLMLingua | Token | Partially | No |
| QUITO-X | Word | Partially | No |
| EXIT | Sentence | Yes (explicitly) | Implicitly |
| RECOMP | Sentence | Implicitly | Implicitly |
| **REET** | **Entity** | **Explicitly** | **Explicitly** |

EXIT comes closest with its entity-preserving objective, but still operates at sentence granularity.

**No existing method does entity-level extraction for compression.**

## Research Questions

1. Can entity-level compression achieve better accuracy at high compression ratios?
2. How should entity-facts be formatted for optimal LLM consumption?
3. Can a single model handle both task-agnostic and query-aware extraction?
4. How does entity-level compression perform on multi-hop reasoning tasks?
5. What's the trade-off between extraction accuracy and inference speed?

## Success Metrics

| Metric | Target |
|--------|--------|
| EM/F1 on QA | Match or exceed QUITO-X at same compression ratio |
| Compression ratio | Achieve 5-10x compression with <5% accuracy drop |
| Inference speed | <100ms per context on CPU |
| Multi-turn degradation | <10% accuracy drop over 10 turns (SCBench) |

## Non-Goals (For Now)

- Soft compression (ICAE-style) - requires model modification
- Abstractive compression - introduces hallucination risk
- Real-time streaming compression - focus on batch first
- Multi-modal contexts - text only initially
