# REET: Research Discoveries & New Direction

## Executive Summary

After deep research into context compression SOTA, we've pivoted from "multi-turn query chaining" to a more fundamental approach: **entity-level semantic compression**.

---

## Key Discoveries

### 1. Complete SOTA Landscape

#### Hard Compression (Token/Sentence Selection)

| Method | Level | Query-Aware | Approach | Paper |
|--------|-------|-------------|----------|-------|
| **LLMLingua-2** | Token | No | Token classification (BERT) | [arXiv](https://arxiv.org/abs/2403.12968) |
| **LongLLMLingua** | Token | Yes | Contrastive perplexity | [arXiv](https://arxiv.org/abs/2310.06839) |
| **Selective Context** | Token | No | Self-information pruning | 2023 |
| **QUITO** | Token | Yes | Attention-based filtering | [arXiv](https://arxiv.org/abs/2408.00274) |
| **QUITO-X** | Word | Yes | T5 cross-attention + IB theory | [arXiv](https://arxiv.org/abs/2408.10497) |
| **EXIT** | Sentence | Yes | Sentence classification (Gemma) | [arXiv](https://arxiv.org/abs/2412.12559) |
| **RECOMP-Ext** | Sentence | Yes | Contriever embeddings | [arXiv](https://arxiv.org/abs/2310.04408) |
| **RECOMP-Abs** | Abstractive | Yes | T5 summarization | [arXiv](https://arxiv.org/abs/2310.04408) |
| **CPC** | Sentence | Yes | Context-aware sentence encoder | 2025 |
| **EHPC** | Token | No | Evaluator heads (training-free) | 2025 |

#### Soft Compression (Learned Representations)

| Method | Approach | Compression | Paper |
|--------|----------|-------------|-------|
| **ICAE** | In-context autoencoder | 4x-8x | [ICLR 2024](https://arxiv.org/abs/2307.06945) |
| **500xCompressor** | KV-based ICAE improvement | Higher ratios | 2025 |
| **AutoCompressors** | Learned summary tokens | Variable | 2023 |

**Key Insight:** All methods operate at token or sentence level. None do semantic/entity-level compression.

### 2. The Gap We Identified

**Current approaches ask:** "Which tokens/sentences to keep?"
**Our approach asks:** "What entities and facts matter?"

```
Example:
Input:  "John Smith, CEO of Acme Corp founded in 1985, announced revenue $5M"
Query:  "When was the company founded?"

LongLLMLingua: "John Smith CEO Acme Corp 1985 announced revenue"  (token drops)
EXIT:          "John Smith, CEO of Acme Corp founded in 1985"     (sentence kept)
OUR VISION:    {"Acme Corp": ["founded 1985"]}                    (entity-fact)
```

### 3. Complete Dataset Landscape

#### Single-Turn QA Datasets (Task-Aware)

| Dataset | Type | Size | Used By | Has Compression Labels? |
|---------|------|------|---------|------------------------|
| **NaturalQuestions (NQ)** | Single-hop QA | 307K | LongLLMLingua, RECOMP, EXIT, QUITO | ❌ Answer spans only |
| **TriviaQA** | Single-hop QA | 650K | RECOMP, EXIT | ❌ Answers only |
| **HotpotQA** | Multi-hop QA | 113K | RECOMP, EXIT, QUITO-X | ✅ Sentence-level supporting facts |
| **2WikiMultiHopQA** | Multi-hop QA | 200K | EXIT, QUITO-X | ✅ Supporting facts |
| **MuSiQue** | Multi-hop QA | 40K | LongLLMLingua, EXIT, QUITO-X | ✅ Supporting facts |
| **SQuAD** | Reading comprehension | 100K | QUITO-X | ❌ Answer spans only |
| **CoQA** | Conversational QA | 127K | QUITO-X | ❌ Answers only |
| **Quoref** | Coreference QA | 24K | QUITO-X | ❌ Answers only |
| **DROP** | Discrete reasoning | 96K | QUITO-X | ❌ Answers only |
| **ASQA** | Ambiguous QA | 6K | QUITO | ❌ |

#### Task-Agnostic / Long Context Benchmarks

| Dataset | Type | Avg Tokens | Used By |
|---------|------|------------|---------|
| **LongBench** | Multi-task (6 categories) | 10K | LLMLingua-2, LongLLMLingua |
| **ZeroSCROLLS** | Multi-task (10 tasks) | 10K | LLMLingua-2, LongLLMLingua |
| **LooGLE** | Long-context understanding | 24K | LongLLMLingua |
| **MeetingBank** | Meeting summarization | Variable | LLMLingua-2 |
| **WikiText-103** | Language modeling | Variable | RECOMP |

#### Reasoning Benchmarks

| Dataset | Type | Used By |
|---------|------|---------|
| **GSM8K** | Math reasoning | LLMLingua-2 |
| **BBH** | Big-Bench Hard | LLMLingua-2 |

#### Domain-Specific

| Dataset | Domain | Used By |
|---------|--------|---------|
| **BioASQ** | Biomedical QA | EXIT |
| **COVID-QA** | COVID research | EXIT |

---

### 4. Datasets for Our Experiments

**Phase 1: Single-Turn QA (Priority)**

| Dataset | Why | Task Type |
|---------|-----|-----------|
| **NaturalQuestions** | Most common benchmark, used by all SOTA | Task-aware |
| **TriviaQA** | Large scale, diverse | Task-aware |
| **HotpotQA** | Has supporting facts (can derive labels) | Task-aware, Multi-hop |
| **SQuAD** | Standard reading comprehension | Task-aware |

**Phase 2: Task-Agnostic**

| Dataset | Why |
|---------|-----|
| **LongBench** | Standard long-context benchmark |
| **ZeroSCROLLS** | Multi-task evaluation |

**Phase 3: Multi-Turn (Later)**

| Dataset | Why |
|---------|-----|
| **SCBench** | Multi-turn degradation testing |
| **CoQA** | Conversational QA |

---

### 5. Dataset Reality for Our Vision

| Dataset | Has | Missing for Entity-Level |
|---------|-----|------------------------|
| HotpotQA | Sentence-level supporting facts | Entity extraction needed |
| 2WikiMultiHopQA | Supporting facts | Entity extraction needed |
| MuSiQue | Supporting facts | Entity extraction needed |
| NQ/TriviaQA/SQuAD | Answer spans only | Full label creation needed |

**No dataset exists with entity-level compression labels.** We need to create our own.

### 6. GLiNER2 Deep Dive

#### Architecture Overview

GLiNER2 uses a **205M parameter transformer encoder** with unified input formulation:

```
[Task Prompt] ⊕ [SEP] ⊕ [Input Text]
```

**Special Tokens:**
- `[P]` (Prompt): Task specifications
- `[E]` (Entity): Precedes entity types
- `[C]` (Child): Hierarchical attributes
- `[L]` (Label): Classification options

#### How Entity Extraction Works

```
1. Input: "John Smith is CEO of Acme Corp" + Schema: [PERSON, ORGANIZATION]

2. Entity Embeddings: [E] tokens → entity_embedding for each type

3. Span Representations: All possible text spans → span_embedding
   - "John" → embed1
   - "John Smith" → embed2
   - "Acme" → embed3
   - "Acme Corp" → embed4
   - etc.

4. Matching: score(span_i, entity_j) = similarity(span_embed, entity_embed)

5. Output: Spans with score > 0.5 threshold
```

#### Training Data (254K examples)

| Source | Count | Type |
|--------|-------|------|
| News | 74K | Real-world |
| Wikipedia | 18K | Real-world |
| Law | 20K | Real-world |
| PubMed | 16K | Real-world |
| ArXiv | 7K | Real-world |
| Synthetic | 119K | GPT-4o generated |

**All annotations by GPT-4o** using task-specific prompts.

#### Training Config
- Optimizer: AdamW (differential LR)
- Backbone: 1×10⁻⁵, Task layers: 2×10⁻⁵
- Epochs: 5
- Context: 2,048 tokens
- Runs on CPU (no GPU required)

#### Key Limitation for Us
- **Requires predefined schema** - can't auto-generate
- **Not query-aware** - extracts based on schema, not relevance to query

---

### 7. REET Model Vision: Custom GLiNER-style Architecture

#### Two Modes

| Mode | Input | Output |
|------|-------|--------|
| **Task-Agnostic** | Context only | All important entities + facts |
| **Query-Aware** | Context + Query | Only query-relevant entities + facts |

#### Proposed Architecture: REET-Extractor

```
┌─────────────────────────────────────────────────────────────┐
│                    REET-Extractor                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mode 1 (Task-Agnostic):                                    │
│  [P] Extract important entities [SEP] [Context]             │
│                     ↓                                       │
│  Output: {"John Smith": ["CEO", "Acme Corp"], ...}          │
│                                                             │
│  Mode 2 (Query-Aware):                                      │
│  [P] Extract for query [SEP] [Query] [SEP] [Context]        │
│                     ↓                                       │
│  Output: Only entities relevant to query                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Differences from GLiNER2

| Aspect | GLiNER2 | REET-Extractor |
|--------|---------|----------------|
| Schema | User-provided | Auto-generated or none |
| Query | Not supported | Core feature |
| Output | Entities by type | Entities + their facts |
| Goal | NER | Compression-ready output |

#### Training Data Creation

**For Task-Agnostic Mode:**
```python
{
  "context": "John Smith, CEO of Acme Corp founded in 1985...",
  "output": {
    "John Smith": ["CEO of Acme Corp"],
    "Acme Corp": ["founded in 1985", "CEO is John Smith"]
  }
}
# Labels from GPT-4: "Extract all important entities and their facts"
```

**For Query-Aware Mode:**
```python
{
  "context": "John Smith, CEO of Acme Corp founded in 1985...",
  "query": "When was the company founded?",
  "output": {
    "Acme Corp": ["founded in 1985"]
  }
}
# Labels from GPT-4: "Extract only entities/facts needed to answer query"
```

#### Architecture Options

1. **Modified GLiNER (Preferred)**
   - Add query encoding branch
   - Condition span scoring on query embedding
   - Output: entity + associated fact spans

2. **T5 Seq2Seq**
   - Input: `[Query?] Context`
   - Output: JSON of entity-facts
   - Simpler but slower inference

3. **DeBERTa + Dual Span Extraction**
   - One head for entity spans
   - One head for fact spans
   - Cross-attention between query and context

---

## Proposed Approach

### Phase 1: Quick Baseline (No Training)

Use GLiNER2 with generic schema to test if entity-based compression is viable:

```python
schema = ["PERSON", "ORGANIZATION", "DATE", "LOCATION", "EVENT", "NUMBER", "FACT"]
entities = gliner2.extract(context, schema)
compressed = format_as_text(entities)
# Benchmark against EXIT, LLMLingua2, LongLLMLingua, QUITO-X
```

**Goal:** Validate entity-based compression works before investing in training.

### Phase 2: Dataset Curation

Create training data for REET-Extractor:

```python
# Task-Agnostic samples (from any text corpus):
{
  "context": "...",
  "mode": "agnostic",
  "output": gpt4_extract_all_entities_facts(context)
}

# Query-Aware samples (from QA datasets):
{
  "context": "...",
  "query": "...",
  "mode": "query_aware",
  "output": gpt4_extract_relevant_entities_facts(context, query)
}
```

**Data Sources:**
- HotpotQA, NQ, TriviaQA (query-aware)
- Wikipedia, News articles (task-agnostic)
- Target: ~100K examples (similar to GLiNER2 scale)

### Phase 3: Train REET-Extractor

Train a single model that handles BOTH modes:

```
Mode 1: [P] summarize [SEP] [Context] → All entity-facts
Mode 2: [P] for query [SEP] [Query] [SEP] [Context] → Relevant entity-facts
```

**Architecture:** Modified GLiNER with:
- Query encoding branch (optional, for mode 2)
- Entity span detection
- Fact span detection (linked to entities)
- Single forward pass output

**Training:**
- Mixed batches of task-agnostic + query-aware samples
- Loss: Binary cross-entropy for span selection
- Differential LR (backbone: 1e-5, heads: 2e-5)

### Phase 4: Multi-turn Extension

Once single-turn works, extend to multi-turn:
- Accumulate entities across turns
- Track entity importance over conversation
- Test on SCBench for degradation

---

## Files to Clean Up

### Remove (Outdated/Redundant)
- `reet/benchmarks/baselines/reet_v0.py` - Query-chaining approach (superseded)
- `reet/benchmarks/baselines/recomp.py` - If exists, remove (not needed for now)
- Any test scripts in `scripts/` that test old approach

### Keep
- `reet/benchmarks/baselines/base.py` - Base compressor class
- `reet/benchmarks/baselines/llmlingua.py` - LLMLingua wrappers (for comparison)
- `reet/benchmarks/baselines/truncation.py` - Simple baseline
- `reet/benchmarks/scbench/` - Evaluation infrastructure
- `scripts/run_scbench.py` - Benchmark runner

### Create New
- `reet/baselines/gliner_baseline.py` - GLiNER2 generic schema baseline
- `reet/data/` - Dataset curation pipeline
- `docs/RESEARCH_NOTES.md` - This document (permanent)

---

## Benchmarks to Run

### Baselines to Compare Against

| Baseline | Level | Query-Aware | Priority | Why |
|----------|-------|-------------|----------|-----|
| **LLMLingua-2** | Token | No | P0 | Task-agnostic SOTA |
| **LongLLMLingua** | Token | Yes | P0 | Query-aware baseline |
| **QUITO-X** | Word | Yes | P0 | Current SOTA on QA (outperforms all) |
| **EXIT** | Sentence | Yes | P1 | Entity-preserving, closest to our vision |
| **RECOMP-Ext** | Sentence | Yes | P1 | Extractive baseline |
| **RECOMP-Abs** | Abstractive | Yes | P2 | Abstractive baseline |
| **Selective Context** | Token | No | P2 | Classic baseline |

### Datasets by Phase

**Phase 1: Single-Turn QA (Task-Aware)**
| Dataset | Size | Why |
|---------|------|-----|
| NaturalQuestions | 307K | Standard benchmark, used by all |
| TriviaQA | 650K | Large scale |
| HotpotQA | 113K | Multi-hop, has supporting facts |
| SQuAD | 100K | Reading comprehension standard |

**Phase 2: Task-Agnostic**
| Dataset | Why |
|---------|-----|
| LongBench | Multi-task long-context |
| ZeroSCROLLS | Standard benchmark |

**Phase 3: Multi-Turn (Later)**
| Dataset | Why |
|---------|-----|
| SCBench | Multi-turn degradation |
| CoQA | Conversational QA |

### Metrics
- EM (Exact Match)
- F1
- Compression ratio
- Inference time

---

## Key References

### Compression Methods
- [LLMLingua-2](https://arxiv.org/abs/2403.12968) - Token classification, task-agnostic
- [LongLLMLingua](https://arxiv.org/abs/2310.06839) - Contrastive perplexity, query-aware
- [QUITO](https://arxiv.org/abs/2408.00274) - Attention-based filtering
- [QUITO-X](https://arxiv.org/abs/2408.10497) - T5 cross-attention, current SOTA
- [EXIT](https://arxiv.org/abs/2412.12559) - Sentence-level, entity-preserving
- [RECOMP](https://arxiv.org/abs/2310.04408) - Extractive/abstractive
- [ICAE](https://arxiv.org/abs/2307.06945) - In-context autoencoder (soft compression)
- [Selective Context](https://arxiv.org/abs/2310.06201) - Self-information pruning

### Entity Extraction
- [GLiNER](https://arxiv.org/abs/2311.08526) - Zero-shot NER (NAACL 2024)
- [GLiNER2](https://arxiv.org/abs/2507.18546) - Schema-driven extraction

### Datasets
- [NaturalQuestions](https://ai.google.com/research/NaturalQuestions) - Google QA
- [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) - Large-scale QA
- [HotpotQA](https://hotpotqa.github.io/) - Multi-hop with supporting facts
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) - Reading comprehension
- [LongBench](https://github.com/THUDM/LongBench) - Long-context benchmark
- [SCBench](https://arxiv.org/abs/2412.10319) - Multi-turn benchmark

---

## Next Steps (Priority Order)

### Immediate (This Session)
1. **Clean up codebase** - Remove old reet_v0.py, update imports
2. **Document research** - Save this plan as permanent docs/RESEARCH_NOTES.md

### Phase 1: GLiNER2 Baseline
3. **Implement GLiNER2 baseline** - Generic schema extraction
4. **Set up evaluation** - NQ, TriviaQA, HotpotQA loaders
5. **Run comparison** - vs LLMLingua2, LongLLMLingua, QUITO-X, EXIT
6. **Analyze results** - Is entity-based compression viable?

### Phase 2: If Baseline Works
7. **Dataset curation pipeline** - GPT-4 labeling for entity relevance
8. **Train relevance model** - DeBERTa/T5 for query→entity filtering
9. **Iterate and improve**

### Phase 3: Multi-Turn (Later)
10. **Extend to multi-turn** - SCBench evaluation
11. **Entity accumulation** - Track entities across turns

---

## Open Questions

1. What generic schema works best across domains?
2. How to format entity-facts as compressed text for LLM consumption?
3. Best model for relevance filtering (DeBERTa vs T5 vs Cross-encoder)?
4. How much compression can entity-level achieve vs sentence-level?
5. QUITO-X claims SOTA - should we prioritize comparing against it first?
6. Should we also implement soft compression baselines (ICAE) for comparison?
