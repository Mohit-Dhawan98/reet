# REET Development Roadmap

## Phase 0: Foundation (Complete)

- [x] Project setup and structure
- [x] Base compressor interface (`BaseCompressor`, `CompressorRegistry`)
- [x] Truncation baseline
- [x] LLMLingua wrappers (LLMLingua-2, LongLLMLingua)
- [x] SCBench evaluation infrastructure
- [x] Research documentation

## Phase 1: GLiNER2 Baseline

**Goal:** Validate entity-based compression viability before training custom model.

### Tasks

- [ ] Implement GLiNER2 wrapper with generic schema
- [ ] Define generic extraction schema
  ```python
  schema = ["PERSON", "ORGANIZATION", "DATE", "LOCATION",
            "EVENT", "NUMBER", "FACT", "ATTRIBUTE"]
  ```
- [ ] Entity-to-text formatting
  ```python
  def format_entities(entities: dict) -> str:
      """Convert entity dict to compressed text."""
      lines = []
      for entity, facts in entities.items():
          lines.append(f"{entity}: {', '.join(facts)}")
      return ". ".join(lines)
  ```
- [ ] Integrate into CompressorRegistry
- [ ] Test on sample contexts

### Success Criteria

- GLiNER2 baseline produces coherent entity extractions
- Compressed output is valid for LLM consumption
- Compression ratio achievable: 3-5x

## Phase 2: Evaluation Infrastructure

**Goal:** Set up comprehensive evaluation on standard QA datasets.

### Tasks

- [ ] NaturalQuestions loader
  ```python
  class NQDataLoader:
      def load(self, split="dev", limit=None) -> list[QASample]:
          ...
  ```
- [ ] TriviaQA loader
- [ ] HotpotQA loader (with supporting facts)
- [ ] SQuAD loader
- [ ] Unified evaluation interface
  ```python
  class QAEvaluator:
      def evaluate(self, samples, compressor, ratio) -> EvalResult:
          ...
  ```
- [ ] Metrics: EM, F1, compression ratio

### Success Criteria

- Can load and evaluate on all 4 datasets
- Results format matches published baselines
- Reproducible evaluation runs

## Phase 3: Baseline Comparisons

**Goal:** Establish performance baselines for all SOTA methods.

### Tasks

- [ ] Run truncation baseline on all datasets
- [ ] Run LLMLingua-2 baseline
- [ ] Run LongLLMLingua baseline
- [ ] Run GLiNER2 baseline
- [ ] Implement QUITO-X wrapper (if code available)
- [ ] Implement EXIT wrapper (if code available)
- [ ] Generate comparison tables

### Expected Output

| Method | NQ EM | TriviaQA EM | HotpotQA F1 | Avg Compression |
|--------|-------|-------------|-------------|-----------------|
| Truncation | X% | X% | X% | 0.5 |
| LLMLingua-2 | X% | X% | X% | 0.5 |
| LongLLMLingua | X% | X% | X% | 0.5 |
| GLiNER2 | X% | X% | X% | varies |

### Success Criteria

- Complete comparison table for all methods
- Identify where entity-based approach wins/loses
- Clear understanding of baseline performance

## Phase 4: Dataset Curation

**Goal:** Create training data for REET-Extractor.

### Tasks

- [ ] Design labeling prompts for GPT-4
  ```
  Task-Agnostic: "Extract all important entities and their facts"
  Query-Aware: "Extract only entities/facts needed to answer: {query}"
  ```
- [ ] Build labeling pipeline
  ```python
  class LabelGenerator:
      def generate_task_agnostic(self, context: str) -> dict:
          ...
      def generate_query_aware(self, context: str, query: str) -> dict:
          ...
  ```
- [ ] Generate task-agnostic samples (Wikipedia, News)
- [ ] Generate query-aware samples (NQ, TriviaQA, HotpotQA)
- [ ] Quality validation
- [ ] Target: ~100K labeled examples

### Data Format

```json
{
  "context": "John Smith, CEO of Acme Corp...",
  "query": "When was Acme founded?",  // null for task-agnostic
  "mode": "query_aware",  // or "task_agnostic"
  "entities": {
    "Acme Corp": ["founded in 1985", "CEO: John Smith"]
  }
}
```

### Success Criteria

- 100K+ labeled examples
- High agreement with human evaluation (sample check)
- Balanced task-agnostic / query-aware split

## Phase 5: REET-Extractor Training

**Goal:** Train custom entity-level extraction model.

### Tasks

- [ ] Implement model architecture
  - DeBERTa backbone
  - Entity span head
  - Fact span head
  - Entity-fact linking
- [ ] Training loop with mixed-mode batches
- [ ] Evaluation during training
- [ ] Hyperparameter tuning
- [ ] Model selection

### Architecture Details

```python
class REETExtractor(nn.Module):
    def __init__(self, backbone="microsoft/deberta-v3-base"):
        self.encoder = AutoModel.from_pretrained(backbone)
        self.entity_head = SpanDetectionHead(...)
        self.fact_head = SpanDetectionHead(...)
        self.linker = EntityFactLinker(...)

    def forward(self, input_ids, attention_mask, mode="query_aware"):
        embeddings = self.encoder(input_ids, attention_mask)
        entities = self.entity_head(embeddings)
        facts = self.fact_head(embeddings, entities)
        linked = self.linker(entities, facts)
        return linked
```

### Success Criteria

- Model converges with decreasing loss
- Extraction quality matches/exceeds GLiNER2 baseline
- Supports both modes in single model

## Phase 6: Evaluation & Iteration

**Goal:** Comprehensive evaluation and improvements.

### Tasks

- [ ] Full evaluation on all datasets
- [ ] Comparison with all baselines
- [ ] Error analysis
- [ ] Ablation studies
  - Mode mixing ratio
  - Backbone choice
  - Head architecture
- [ ] Iteration based on findings

### Success Criteria

- Match or exceed QUITO-X on QA tasks
- 5x+ compression with <5% accuracy drop
- Clear understanding of method strengths/weaknesses

## Phase 7: Multi-Turn Extension

**Goal:** Extend to multi-turn conversation compression.

### Tasks

- [ ] Entity accumulation across turns
- [ ] Turn-aware importance scoring
- [ ] SCBench evaluation
- [ ] CoQA evaluation
- [ ] Multi-turn optimizations

### Success Criteria

- <10% degradation over 10 turns on SCBench
- Competitive with baselines on CoQA

## Timeline

| Phase | Estimated Effort |
|-------|------------------|
| Phase 1 | Quick (GLiNER2 wrapper) |
| Phase 2 | Medium (dataset loaders) |
| Phase 3 | Medium (running experiments) |
| Phase 4 | Longer (100K labels) |
| Phase 5 | Medium (training) |
| Phase 6 | Medium (evaluation) |
| Phase 7 | Medium (multi-turn) |

## Current Status

**Active:** Phase 1 - Implementing GLiNER2 baseline

## Open Questions

1. What generic schema works best across domains?
2. Best output format for LLM consumption?
3. How to handle coreference in entity extraction?
4. Should facts be extractive or can they be slightly rephrased?
5. How to handle very long contexts (>2048 tokens)?
