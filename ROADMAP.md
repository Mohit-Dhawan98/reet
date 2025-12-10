# REET: Research Roadmap

## Overview

**Goal**: Build the first learned compression that works in multi-turn conversations.

**Timeline**: 16 weeks to publishable results

**Primary Metric**: SCBench Turn-10 retention (target: >90%, vs 43-48% for baselines)

---

## Success Metrics by Phase

| Metric | MVP (W1-6) | Parity (W7-12) | Leadership (W13-16) |
|--------|------------|----------------|---------------------|
| Single-turn (LongBench) | >93% | >95% | >95% |
| Multi-turn T10 (SCBench) | >80% | >88% | **>90%** |
| Latency | <100ms | <75ms | **<50ms** |
| Entity retention | >90% | >95% | >98% |

---

## Phase 1: Baselines & Infrastructure (Weeks 1-4)

### Goal
Establish reproducible evaluation and quantify the multi-turn gap.

### Week 1-2: Setup

**Project Infrastructure**
- [ ] GitHub repository with CI/CD
- [ ] Package structure (`reet/`)
- [ ] Dependencies (torch, transformers, tiktoken)
- [ ] Pre-commit hooks (ruff, mypy)

**Evaluation Harness**
- [ ] SCBench integration (PRIORITY - multi-turn)
- [ ] LongBench evaluation script
- [ ] NoLiMa evaluation script
- [ ] HotpotQA evaluation script

```python
# reet/benchmarks/scbench/evaluate.py
def evaluate_multiturn(compressor, model, n_turns=10):
    """
    Key evaluation: same compressed context, multiple queries.
    Measures degradation over turns.
    """
    for scenario in scbench.load_scenarios():
        compressed = compressor.compress(scenario.context)

        scores_by_turn = []
        for turn, query in enumerate(scenario.queries[:n_turns]):
            response = model.generate(compressed + query)
            score = evaluate(response, scenario.answers[turn])
            scores_by_turn.append(score)

        yield {
            "t1": scores_by_turn[0],
            "t5": scores_by_turn[4],
            "t10": scores_by_turn[9],
            "degradation": scores_by_turn[0] - scores_by_turn[9]
        }
```

### Week 3-4: Baseline Measurements

**Run All Baselines**
- [ ] Simple truncation (last-k tokens)
- [ ] LLMLingua-2 on all benchmarks
- [ ] LongLLMLingua (question-aware variant)
- [ ] RECOMP extractive + abstractive

**Document the Multi-Turn Gap**

| Benchmark | Method | Turn 1 | Turn 5 | Turn 10 | Notes |
|-----------|--------|--------|--------|---------|-------|
| SCBench | Truncation | ? | ? | ? | Baseline |
| SCBench | LLMLingua-2 | ? | ? | ? | Primary competitor |
| SCBench | RECOMP | ? | ? | ? | Quality baseline |

**Basic Rule-Based REET**
- [ ] Slot budget system
- [ ] Simple heuristic strategies
- [ ] Measure: how far can heuristics get?

### Deliverables
- [ ] `reet/benchmarks/` with evaluation scripts
- [ ] `results/baselines/` with documented numbers
- [ ] Baseline comparison table showing 50%+ multi-turn drop
- [ ] Basic `reet/` package structure

### Exit Criteria
- Can reproduce LLMLingua-2 numbers (±2%)
- Have clear evidence of multi-turn degradation
- SCBench evaluation working

---

## Phase 2: Token Importance Model (Weeks 5-8)

### Goal
Train a token scorer on conversation data that maintains importance across turns.

### Week 5-6: Multi-Turn Training Data

**Data Collection**
- [ ] Download conversation datasets (ShareGPT, LMSYS)
- [ ] Download HotpotQA, LongBench for task-specific data
- [ ] Clean and preprocess

**Multi-Turn Importance Labels**

```python
# Key innovation: label importance based on LATER turns
def generate_multiturn_labels(conversation, model="gpt-4o"):
    """
    For each turn, label which tokens from previous turns
    were important for generating the answer.

    Different from single-turn: importance compounds over turns.
    """
    labels = []
    for turn_idx in range(len(conversation)):
        # Context: all previous turns
        context = conversation[:turn_idx]
        query = conversation[turn_idx].query

        # Use frontier model to identify important context
        importance = analyze_attention(model, context, query)

        # Key: boost tokens that were important in earlier turns
        for prev_idx in range(turn_idx):
            if labels[prev_idx]:
                importance = compound_importance(importance, labels[prev_idx])

        labels.append(importance)

    return labels
```

**Training Data Generation**
- [ ] Generate 30K conversation samples with multi-turn labels
- [ ] Include entity annotations
- [ ] Estimated cost: $150-300 in API calls

### Week 7-8: Model Training

**Model Architecture**
```python
class TurnAwareTokenScorer(nn.Module):
    """
    Token scorer that takes turn history into account.
    """
    def __init__(self, base="distilbert-base-uncased"):
        self.encoder = AutoModel.from_pretrained(base)
        self.turn_embedding = nn.Embedding(MAX_TURNS, HIDDEN)
        self.importance_head = nn.Linear(HIDDEN, 1)

    def forward(self, tokens, turn_positions, entity_mask=None):
        # Encode tokens
        hidden = self.encoder(tokens).last_hidden_state

        # Add turn position information
        turn_emb = self.turn_embedding(turn_positions)
        hidden = hidden + turn_emb

        # Boost entities if mask provided
        if entity_mask is not None:
            hidden = hidden * (1 + entity_mask * ENTITY_BOOST)

        # Score importance
        scores = self.importance_head(hidden).sigmoid()
        return scores
```

**Training**
- [ ] Train on Mac MPS or Colab
- [ ] Hyperparameter search: lr, batch_size, epochs
- [ ] Track with Weights & Biases

**Key Ablations**
- [ ] Single-turn vs multi-turn training data
- [ ] With vs without turn position embedding
- [ ] With vs without entity boosting
- [ ] Turn history window: 0, 3, 5, 10

### Deliverables
- [ ] `reet/models/token_scorer.py`
- [ ] Trained weights in `checkpoints/`
- [ ] Training script in `scripts/train_scorer.py`
- [ ] Ablation results table

### Exit Criteria
- Token scorer shows <10% drop at turn 5 (vs 30%+ for baselines)
- Inference <50ms
- Clear ablation showing multi-turn training helps

---

## Phase 3: Entity-Preserving Compressor (Weeks 9-12)

### Goal
Train abstractive compressor with explicit entity preservation loss.

### Week 9-10: Training Data

**Compression Pair Generation**
```python
def generate_compression_pairs(context, ratios=[0.3, 0.5, 0.7]):
    """
    Generate (full, compressed) pairs with entity annotations.
    """
    entities = extract_entities(context)

    for ratio in ratios:
        compressed = gpt4_compress(context, ratio)

        yield {
            "full": context,
            "compressed": compressed,
            "ratio": ratio,
            "entities": entities,
            "preserved": check_entities(compressed, entities)
        }
```

**Training Data**
- [ ] Generate 20K compression pairs
- [ ] Include entity preservation labels
- [ ] Estimated cost: $200-400 in API calls

### Week 11-12: Model Training

**Entity Preservation Loss**
```python
def compute_loss(model, batch, entity_weight=0.3):
    """
    Combined loss: language modeling + entity preservation.

    Key: heavily penalize losing entities that were referenced
    in previous turns.
    """
    # Standard LM loss
    lm_loss = model(**batch).loss

    # Entity preservation loss
    generated = model.generate(batch["input_ids"])
    generated_text = tokenizer.batch_decode(generated)

    entity_loss = 0
    for gen, entities, turn_refs in zip(
        generated_text,
        batch["entities"],
        batch["turn_references"]
    ):
        # Higher penalty for entities referenced in more turns
        for entity in entities:
            weight = 1.0 + 0.2 * turn_refs.get(entity, 0)
            if entity.lower() not in gen.lower():
                entity_loss += weight

    return lm_loss + entity_weight * entity_loss
```

**Training**
- [ ] Fine-tune T5-small or BART-small
- [ ] Entity preservation loss
- [ ] Ablate entity_weight: 0.0, 0.1, 0.3, 0.5

**Key Ablations**
- [ ] T5-small vs BART-small
- [ ] Entity loss weight
- [ ] With vs without turn reference weighting

### Deliverables
- [ ] `reet/models/compressor.py`
- [ ] Trained weights in `checkpoints/`
- [ ] Entity retention metrics

### Exit Criteria
- Entity retention >95% at 50% compression
- <2 EM loss on HotpotQA at 6% compression
- Inference <100ms

---

## Phase 4: Integration & Final Benchmarks (Weeks 13-14)

### Goal
Combine models, run full benchmarks, document results.

### Pipeline Integration

```python
class REET:
    """
    Full REET compression pipeline.
    """
    def __init__(self, token_scorer, compressor, entity_tracker):
        self.token_scorer = token_scorer
        self.compressor = compressor
        self.entity_tracker = entity_tracker

    @classmethod
    def from_pretrained(cls, name="reet-base"):
        """Load from HuggingFace."""
        scorer = TurnAwareTokenScorer.from_pretrained(f"reet/{name}-scorer")
        compressor = EntityCompressor.from_pretrained(f"reet/{name}-compressor")
        return cls(scorer, compressor, EntityTracker())

    def compress(self, context, query, turn_history=None, target_ratio=0.5):
        """
        Compress with multi-turn awareness.

        Key: turn_history allows maintaining state across calls.
        """
        # Update entity tracking
        entities = self.entity_tracker.update(context, query)

        # Score tokens with turn awareness
        scores = self.token_scorer(
            context,
            turn_history=turn_history,
            entity_mask=entities.mask
        )

        # Select top tokens
        top_tokens = select_by_score(context, scores, ratio=0.7)

        # Generate compressed version
        compressed = self.compressor.compress(
            top_tokens,
            target_ratio=target_ratio,
            preserve_entities=entities.names
        )

        return CompressionResult(
            text=compressed,
            entities_preserved=entities.check(compressed),
            compression_ratio=len(compressed) / len(context)
        )
```

### Full Benchmark Suite

**SCBench (Primary)**
- [ ] All 12 tasks
- [ ] 10 turns each
- [ ] Compare against all baselines

**LongBench**
- [ ] All 21 tasks
- [ ] 30%, 50%, 70% compression

**HotpotQA**
- [ ] Full test set
- [ ] 6% compression (match RECOMP)

**NoLiMa**
- [ ] All context lengths
- [ ] 50% compression

### Results Tables

**Multi-Turn (Primary Result)**
| Method | Turn 1 | Turn 5 | Turn 10 | Degradation |
|--------|--------|--------|---------|-------------|
| LLMLingua-2 | 92% | 68% | 43% | -53% |
| SnapKV | 95% | 72% | 48% | -49% |
| **REET** | 95% | 93% | **91%** | **-4%** |

**Single-Turn**
| Method | LongBench 50% | HotpotQA 6% | Latency |
|--------|---------------|-------------|---------|
| LLMLingua-2 | 95% | -2 EM | 100ms |
| RECOMP | 93% | -3 EM | 500ms |
| **REET** | 95% | **-1 EM** | **<50ms** |

### Deliverables
- [ ] `reet/pipeline.py` - full integrated pipeline
- [ ] `results/final/` - all benchmark results
- [ ] Comparison plots and tables

### Exit Criteria
- SCBench T10: >90% (vs 43-48% baselines)
- LongBench: >95% at 50% (match baselines)
- Latency: <100ms total

---

## Phase 5: Release (Weeks 15-16)

### Goal
Package, publish, document, launch.

### PyPI Package
- [ ] Final pyproject.toml
- [ ] Build and test installation
- [ ] Publish: `pip install reet`

### HuggingFace Hub
- [ ] Upload token scorer model
- [ ] Upload compressor model
- [ ] Model cards with usage examples

### Documentation
- [ ] README with quick start
- [ ] API documentation
- [ ] Benchmark reproduction guide

### Launch
- [ ] Technical blog post
- [ ] Twitter/X thread
- [ ] Reddit (r/MachineLearning, r/LocalLLaMA)
- [ ] Hacker News

### Deliverables
- [ ] `pip install reet` works
- [ ] Models on HuggingFace
- [ ] Published benchmark results
- [ ] Blog post / arXiv preprint

### Exit Criteria
- Package installable and working
- 50+ GitHub stars in first week
- At least one external reproduction attempt

---

## Future: Knowledge Graph Memory (v2)

### Goal
Add temporal knowledge graph for long-running agents.

### Planned Features
- Entity/relation extraction from context
- Temporal KG storage (inspired by Zep)
- KG-to-context retrieval
- Fact invalidation over time

### Inspiration
- [Zep](https://arxiv.org/abs/2501.13956) - Temporal KG for agents
- [A-MEM](https://arxiv.org/abs/2502.12110) - Zettelkasten-style memory
- [Graphiti](https://github.com/getzep/graphiti) - Graph memory

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Multi-turn training data doesn't help | Medium | High | Try different labeling approaches, use more frontier model calls |
| Can't beat baselines on single-turn | Low | Medium | Focus on multi-turn differentiation, accept single-turn parity |
| Latency too high | Medium | Medium | Try smaller models, quantization, token-scorer-only mode |
| Entity tracking doesn't reduce drift | Medium | High | Ablate different entity weighting schemes, try reference detection |
| GPU access needed | Low | Low | Start on Mac MPS, use Colab/Lambda as needed |

---

## Resource Requirements

### Compute
- Development: Mac M1/M2 (MPS) sufficient
- Training: Colab Pro or 1x A100 for ~10 hours
- Evaluation: CPU sufficient

### API Costs
- Training data generation: $300-600
- Evaluation runs: $100-200
- **Total estimated: $500-1000**

### Time
- Full-time: 16 weeks
- Part-time: 24-32 weeks

---

## References

### Key Papers
- LLMLingua-2: [ACL 2024](https://aclanthology.org/2024.acl-long.91.pdf)
- SCBench: [arXiv:2412.10319](https://arxiv.org/abs/2412.10319)
- RECOMP: [ICLR 2024](https://arxiv.org/abs/2310.04408)
- DynamicKV: [arXiv:2412.14838](https://arxiv.org/abs/2412.14838)

### Benchmarks
- SCBench: https://github.com/microsoft/SCBench
- LongBench: https://github.com/THUDM/LongBench
- NoLiMa: https://github.com/adobe-research/NoLiMa
- HotpotQA: https://hotpotqa.github.io/
