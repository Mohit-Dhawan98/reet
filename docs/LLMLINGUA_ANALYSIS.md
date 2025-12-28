# LLMLingua Deep Dive: Architecture, Limitations, and REET's Differentiation

## Table of Contents
1. [LLMLingua Family Overview](#llmlingua-family-overview)
2. [LLMLingua-2 Technical Architecture](#llmlingua-2-technical-architecture)
3. [Training Methodology](#training-methodology)
4. [The Multi-Turn Failure Mode](#the-multi-turn-failure-mode)
5. [Other Baselines: RECOMP & Alternatives](#other-baselines-recomp--alternatives)
6. [REET's Architectural Differentiation](#reets-architectural-differentiation)
7. [What to Learn vs. What to Avoid](#what-to-learn-vs-what-to-avoid)
8. [References](#references)

---

## LLMLingua Family Overview

The LLMLingua family consists of three main variants:

| Model | Year | Key Innovation | Limitation |
|-------|------|----------------|------------|
| **LLMLingua** | 2023 | Perplexity-based token pruning | Slow (requires LLM forward pass) |
| **LongLLMLingua** | 2023 | Query-aware compression | Still perplexity-based, query-dependent |
| **LLMLingua-2** | 2024 | Learned token classifier | No semantic understanding, multi-turn failure |

### The Evolution

**LLMLingua (v1)** used a small LLM (like LLaMA-7B) to compute perplexity for each token. High perplexity = surprising/important = keep. Low perplexity = predictable = drop.

```
"The capital of France is Paris"
     ↓ Perplexity scoring
[low, low,    low,   low,   low, HIGH]  → Keep "Paris"
```

**Problem**: Running even a 7B model for scoring is slow (~100ms+ per prompt).

**LLMLingua-2** replaced the LLM with a small classifier (XLM-RoBERTa, 560M params), trained to mimic GPT-4's judgment of token importance. This made it 3-6x faster.

---

## LLMLingua-2 Technical Architecture

### Model Architecture

```
Input Text
    ↓
┌─────────────────────────────────────┐
│  XLM-RoBERTa-Large (560M params)    │
│  - 24 layers                        │
│  - 1024 hidden dim                  │
│  - 16 attention heads               │
│  - Max 512 tokens per chunk         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Token Classification Head          │
│  - Linear(1024, 2)                  │
│  - Softmax                          │
│  - Output: P(keep) for each token   │
└─────────────────────────────────────┘
    ↓
Token Selection (keep top-k by score)
    ↓
Compressed Text (original tokens, gaps removed)
```

### The Compression Algorithm

```python
# Simplified LLMLingua-2 algorithm
def compress(text, target_ratio):
    # 1. Tokenize
    tokens = tokenizer.encode(text)

    # 2. Chunk into 512-token segments
    chunks = chunk_by_sentence(tokens, max_len=510)

    # 3. Score each token
    all_scores = []
    for chunk in chunks:
        # Forward pass through classifier
        logits = model(chunk)  # Shape: [seq_len, 2]
        scores = softmax(logits)[:, 1]  # P(keep)
        all_scores.extend(scores)

    # 4. Select top tokens by score
    num_keep = int(len(tokens) * target_ratio)
    top_indices = argsort(all_scores)[-num_keep:]
    top_indices = sorted(top_indices)  # Maintain order

    # 5. Reconstruct text
    kept_tokens = [tokens[i] for i in top_indices]
    return tokenizer.decode(kept_tokens)
```

### Key Implementation Details

**Chunking Strategy** (from `prompt_compressor.py`):
- Splits at sentence boundaries (., !, ?, \n)
- Falls back to hard truncation if no boundary within 510 tokens
- Processes chunks independently (no cross-chunk attention)

**Force Tokens**:
- Certain tokens are always kept: `\n`, `.`, `?`, `!`
- Preserves document structure
- Configurable via `force_tokens` parameter

**Compression Modes**:
```python
# Standard compression
result = compressor.compress_prompt(text, rate=0.5)

# With question-awareness (LongLLMLingua mode)
result = compressor.compress_prompt(
    text,
    question="What is the capital?",
    condition_in_question="after_condition",
    reorder_context="sort",  # Reorder chunks by relevance
)
```

---

## Training Methodology

### Data Generation Pipeline

LLMLingua-2's training data was generated using GPT-4 as a teacher:

```
Step 1: Collect QA datasets (MeetingBank, LongBench, etc.)

Step 2: For each (context, question, answer) triple:
    - Ask GPT-4: "Which tokens are essential to answer this question?"
    - GPT-4 returns token-level labels

Step 3: Train XLM-RoBERTa on these labels
    - Input: context tokens
    - Output: binary label per token (keep/drop)
    - Loss: Cross-entropy
```

### Training Data Composition

| Dataset | Domain | Size |
|---------|--------|------|
| MeetingBank | Meeting transcripts | 1.3M tokens |
| LongBench | Mixed long-context | 2.1M tokens |
| Arxiv | Scientific papers | 800K tokens |
| **Total** | - | ~4.2M labeled tokens |

### The Labeling Prompt (approximate)

```
Given the following context and question, identify which tokens
are ESSENTIAL to correctly answer the question. Mark each token
as 1 (essential) or 0 (can be removed).

Context: {context}
Question: {question}
Answer: {answer}

For each token, output 1 if removing it would make the answer
incorrect or incomplete, 0 otherwise.
```

### Why This Training Approach Fails for Multi-Turn

The training signal is:
> "Is this token needed for **this specific question**?"

Not:
> "Is this token needed for **any future question**?"

This is the fundamental flaw. The model learns to optimize for single-turn QA, not conversational context preservation.

---

## The Multi-Turn Failure Mode

### Concrete Example

**Original Context:**
```
Meeting Transcript:
Alice: I'll handle the frontend redesign. Budget is $50k.
Bob: I'm taking the API migration. We need it done by March 15.
Carol: Database optimization is mine. Using PostgreSQL.
David: I'll coordinate with the design team on branding.
```

**Turn 1:**
```
Query: "What is Alice working on?"
LLMLingua-2 scores:
- "Alice" → 0.95 (HIGH - mentioned in query)
- "frontend redesign" → 0.92 (HIGH - answer)
- "$50k" → 0.78 (MEDIUM - related to Alice)
- "Bob" → 0.15 (LOW - not relevant to query)
- "API migration" → 0.12 (LOW)
- "March 15" → 0.10 (LOW)
- "Carol", "Database", "David" → 0.05-0.15 (LOW)

At 30% compression, keeps:
"Alice frontend redesign $50k"

Answer: "frontend redesign" ✓
```

**Turn 2:**
```
Compressed context: "Alice frontend redesign $50k"
Query: "When is Bob's deadline?"

LLMLingua-2: Cannot answer - Bob was deleted!
```

### Degradation Measurements (from SCBench)

| Method | Turn 1 | Turn 5 | Turn 10 | Degradation |
|--------|--------|--------|---------|-------------|
| No compression | 95% | 95% | 95% | 0% |
| Truncation | 70% | 55% | 40% | -43% |
| LLMLingua-2 | 92% | 68% | 43% | **-53%** |
| LongLLMLingua | 93% | 70% | 45% | **-52%** |

The ~50% degradation by turn 10 is consistent across compression methods because they all share the same flaw: **query-dependent selection**.

### Why Query-Aware Compression Makes It Worse

Counterintuitively, LongLLMLingua (query-aware) sometimes performs *worse* than base LLMLingua-2 in multi-turn:

```
Turn 1 Query: "What color is the car?"
LongLLMLingua aggressively keeps color-related tokens, drops everything else.

Turn 2 Query: "How much does it cost?"
Price information was dropped because it wasn't relevant to color query.
```

Query-awareness amplifies the single-turn optimization, making multi-turn worse.

---

## Other Baselines: RECOMP & Alternatives

### RECOMP (Retrieval-Enhanced Compression)

**Paper**: [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408)

**Architecture**: Two-stage compression

```
Stage 1: Extractive Compressor
    - Sentence-level selection (not token-level)
    - Trained to select sentences containing answer
    - Model: Fine-tuned T5-base

Stage 2: Abstractive Compressor
    - Summarizes selected sentences
    - Generates new text (not just selection)
    - Model: Fine-tuned T5-base
```

**Key Difference from LLMLingua**:
- Operates on sentences, not tokens
- Can generate new text (abstractive)
- Better preserves coherence

**Multi-Turn Performance**: Similar degradation (~50%) because it's still query-dependent.

### Selective Context

**Paper**: [Selective Context for LLMs](https://arxiv.org/abs/2310.06201)

**Approach**: Self-information based filtering
- Compute self-information: `-log P(token | context)`
- High self-information = rare/important = keep
- No learned model, pure information-theoretic

**Limitation**: No semantic understanding, just statistical rarity.

### Comparison Table

| Method | Granularity | Query-Aware | Generates New Text | Multi-Turn Safe |
|--------|-------------|-------------|-------------------|-----------------|
| LLMLingua-2 | Token | No | No | ❌ |
| LongLLMLingua | Token | Yes | No | ❌ |
| RECOMP | Sentence | Yes | Yes (abstractive) | ❌ |
| Selective Context | Token | No | No | ❌ |
| **REET** | Entity | Turn-aware | Yes (compressed repr) | ✅ |

---

## REET's Architectural Differentiation

### Fundamental Paradigm Shift

**LLMLingua asks**: "Which tokens can I delete for this query?"
**REET asks**: "What information must I preserve for any future query?"

### REET's Three-Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REET Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ENTITY REGISTRY                                         │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Tracks: Entities, Attributes, Relationships     │    │
│     │ Updates: Per-turn (additive, not destructive)   │    │
│     │ Storage: Compressed embedding space             │    │
│     └─────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  2. TURN-AWARE TOKEN SCORER                                │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Input: Token + Entity Registry + Turn History   │    │
│     │ Output: Importance score (turn-aware)           │    │
│     │ Key: Scores based on FUTURE utility, not just   │    │
│     │      current query relevance                    │    │
│     └─────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  3. ENTITY-PRESERVING COMPRESSOR                           │
│     ┌─────────────────────────────────────────────────┐    │
│     │ Constraint: Never drop registered entities      │    │
│     │ Method: Compress around entities, not through   │    │
│     │ Output: Dense representation preserving all     │    │
│     │         entity information                      │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Entity-Centric Compression Works

**Observation**: In multi-turn conversations, queries are almost always about entities or their attributes.

```
Turn 1: "What is Alice working on?" → Entity: Alice, Attr: work
Turn 2: "When is Bob's deadline?" → Entity: Bob, Attr: deadline
Turn 3: "Who handles the database?" → Entity: ?, Attr: database
Turn 4: "What's the total budget?" → Entity: ALL, Attr: budget
```

If you preserve all entities and their attributes, you can answer any future query.

### REET vs LLMLingua: Token Treatment

**LLMLingua-2 token scoring:**
```python
score(token) = P(token_needed | current_query)
```

**REET token scoring:**
```python
score(token) = (
    entity_importance(token) * 0.4 +      # Is this an entity?
    relationship_importance(token) * 0.3 + # Does this link entities?
    recency_weight(token) * 0.2 +          # How recent?
    query_relevance(token) * 0.1           # Current query (low weight!)
)
```

The key insight: **current query relevance is only 10% of the score**, not 100%.

### Compression Output Comparison

**LLMLingua-2 output** (token selection):
```
Input:  "Alice is a doctor who works at City Hospital. Bob is a lawyer."
Output: "Alice doctor City Hospital" (at 50% compression)
```

**REET output** (semantic compression):
```
Input:  "Alice is a doctor who works at City Hospital. Bob is a lawyer."
Output: {
    entities: {
        "Alice": {type: "person", profession: "doctor", workplace: "City Hospital"},
        "Bob": {type: "person", profession: "lawyer"}
    },
    compressed_text: "Alice(doctor@City Hospital), Bob(lawyer)"
}
```

Both are ~50% compression, but REET preserves ALL information in a structured form.

---

## What to Learn vs. What to Avoid

### ✅ Learn From LLMLingua

1. **Token-level scoring signal**
   - Their classifier provides useful "information density" signal
   - Can be used as ONE input to REET's scorer (not the only input)

2. **Chunking strategy**
   - Their sentence-boundary chunking is well-engineered
   - Handles edge cases (no boundary, very long sentences)

3. **GPT-4 distillation for training data**
   - The idea of using GPT-4 to label important tokens is clever
   - REET can use similar approach: ask GPT-4 to label entities/relationships

4. **Force tokens for structure preservation**
   - Always keeping \n, ., ?, ! maintains readability
   - REET should similarly protect entity boundaries

5. **Efficient inference**
   - Small classifier + token selection is fast
   - REET's entity registry lookup should be O(1)

### ❌ Avoid From LLMLingua

1. **Deletion-based compression**
   - Once tokens are deleted, information is GONE
   - REET should compress/encode, not delete

2. **Query-only optimization**
   - Training signal: "needed for this query"
   - REET signal: "needed for any future query in this conversation"

3. **No entity awareness**
   - LLMLingua treats "Bob" the same as "the"
   - REET must recognize entities as special

4. **Independent chunk processing**
   - LLMLingua chunks don't share information
   - REET's entity registry should span all chunks

5. **Stateless compression**
   - LLMLingua recompresses from scratch each turn
   - REET should maintain state across turns (entity registry persists)

### REET's Unique Innovations (Not in LLMLingua)

1. **Entity Registry**
   - Persistent, grows across turns
   - Indexes entities for O(1) lookup
   - Tracks relationships between entities

2. **Turn-Aware Scoring**
   - Scores consider conversation history
   - Predicts future utility, not just current relevance
   - Decays importance of superseded information

3. **Structured Compression**
   - Output is both text AND structured data
   - Can reconstruct full semantics from compressed form
   - Enables entity-based retrieval

4. **Compression Constraints**
   - Hard constraint: never lose a registered entity
   - Soft constraint: prefer compressing filler over content

---

## References

### Primary Papers

1. **LLMLingua-2**: Jiang et al. (2024). "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression"
   - Paper: https://aclanthology.org/2024.acl-long.91.pdf
   - Code: https://github.com/microsoft/LLMLingua

2. **LLMLingua (v1)**: Jiang et al. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
   - Paper: https://arxiv.org/abs/2310.05736

3. **LongLLMLingua**: Jiang et al. (2023). "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"
   - Paper: https://arxiv.org/abs/2310.06839

4. **RECOMP**: Xu et al. (2023). "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation"
   - Paper: https://arxiv.org/abs/2310.04408
   - Code: https://github.com/carriex/recomp

5. **SCBench**: Li et al. (2024). "SCBench: A KV Cache-Centric Analysis of Long-Context Methods"
   - Paper: https://arxiv.org/abs/2412.10319
   - Code: https://github.com/microsoft/MInference/tree/main/scbench

### Benchmarks

6. **LongBench**: Bai et al. (2023). "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
   - Paper: https://arxiv.org/abs/2308.14508
   - Code: https://github.com/THUDM/LongBench

7. **HotpotQA**: Yang et al. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"
   - Website: https://hotpotqa.github.io/

### Background Reading

8. **XLM-RoBERTa**: Conneau et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale"
   - Paper: https://arxiv.org/abs/1911.02116
   - Model: https://huggingface.co/xlm-roberta-large

9. **Knowledge Distillation**: Hinton et al. (2015). "Distilling the Knowledge in a Neural Network"
   - Paper: https://arxiv.org/abs/1503.02531

---

## Summary

LLMLingua-2 is a well-engineered solution for **single-turn** prompt compression. Its token-level classifier is fast and effective when you know the query upfront.

However, its fundamental design—query-dependent token deletion—makes it structurally incapable of handling multi-turn conversations without catastrophic information loss.

REET's entity-centric, turn-aware approach addresses this by:
1. Preserving ALL entities regardless of current query
2. Scoring tokens based on future utility, not just current relevance
3. Compressing semantically rather than deleting syntactically

The ~50% degradation that LLMLingua-2 shows by turn 10 is not a bug to be fixed—it's a fundamental limitation of the deletion-based paradigm. REET represents a paradigm shift, not an incremental improvement.
