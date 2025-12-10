# REET: Benchmarking Strategy

## The Core Thesis

**Single-turn compression is solved. Multi-turn is broken.**

Every existing compression method (LLMLingua-2, SnapKV, RECOMP) drops **50%+ performance** by turn 10 in multi-turn conversations. REET is the first learned compression trained on conversation dynamics.

---

## The Three Battles

### Battle 1: Single-Turn Accuracy (Table Stakes)

This is baseline territory. Must match, not necessarily beat.

| Method | Benchmark | Performance | Notes |
|--------|-----------|-------------|-------|
| **LLMLingua-2** | GSM8K | 98.5% retention at 20x | SOTA for single-turn |
| **LongLLMLingua** | QA tasks | Better than base LLMLingua | Question-aware variant |
| **RECOMP** | HotpotQA | 2-3 EM loss at 6% tokens | Extractive + abstractive |
| **Selective Context** | LongBench | 75-85% retention | Attention-based |

**REET Target**: >95% single-turn retention at 50% compression (match LLMLingua-2)

---

### Battle 2: Multi-Turn Robustness (KEY DIFFERENTIATOR)

This is where REET wins. SCBench data shows catastrophic degradation:

| Method | Turn 1 | Turn 5 | Turn 10 | Degradation |
|--------|--------|--------|---------|-------------|
| **LLMLingua-2** | 92% | 68% | 43% | **-53%** |
| **SnapKV** | 95% | 72% | 48% | **-49%** |
| **H2O** | 90% | 65% | 41% | **-54%** |
| **StreamingLLM** | 88% | 60% | 38% | **-57%** |
| **REET Target** | 95% | 93% | >90% | **<5%** |

**Why competitors fail multi-turn:**
1. Trained on single contexts, not conversations
2. No entity tracking across turns
3. Important tokens from turn 1 get dropped by turn 5
4. No memory of what was referenced before

**REET's solution:**
- Train on conversation pairs
- Entity persistence scoring
- Turn-aware importance (referenced tokens stay important)

---

### Battle 3: Latency (Operational Requirement)

Must be fast enough for real-time agent use.

| Method | Latency | Category |
|--------|---------|----------|
| **LLMLingua-2** | ~100ms | Beat this |
| **RECOMP abstractive** | ~500ms | Much slower |
| **DynamicKV** | Runtime (inference) | Different category |
| **Simple truncation** | <10ms | Floor |
| **REET Target** | <50ms scoring | **2x faster than LLMLingua** |

**Latency breakdown target:**

| Component | Target | Notes |
|-----------|--------|-------|
| Token Scorer | <30ms | Must beat LLMLingua |
| Compressor | <50ms | Optional, adds quality |
| Slot Budget | <5ms | Rule-based overhead |
| **Total** | <100ms | Real-time viable |

---

## Baseline Methods

### Prompt Compression Methods

These are direct competitors—same problem, same approach:

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **LLMLingua-2** | Fine-tuned BERT token selection | Fast, high single-turn accuracy | Multi-turn degrades, no entity awareness |
| **LongLLMLingua** | Question-aware LLMLingua | Better for QA tasks | Still single-turn focused |
| **RECOMP Extractive** | Sentence selection by embedding | Good accuracy | Slower, sentence-level granularity |
| **RECOMP Abstractive** | Generate compressed summary | Best quality | Slow (~500ms), expensive |
| **Selective Context** | Attention-based token selection | Simple | 75-85% retention only |

### KV Cache Methods

These are **complementary**, not competitive—they work at inference time, not context preparation:

| Method | Approach | Performance | Notes |
|--------|----------|-------------|-------|
| **DynamicKV** | Adaptive KV cache eviction | 100% at 6.9% tokens | Inference-time only |
| **SnapKV** | Attention-pattern-based eviction | 3.6x speedup, 36% memory | Can't use for context prep |
| **FastGen** | KV cache compression | Similar to SnapKV | ICLR 2024 |
| **H2O** | Heavy Hitter Oracle | Keeps important KV pairs | Degrades multi-turn |

**Key insight**: KV cache methods show extreme compression is possible (6.9% tokens!), but they can't be used to pre-compress context before sending to API. REET fills this gap.

### Agent Frameworks

Different problem (task completion vs compression):

| Method | Approach | Results |
|--------|----------|---------|
| **ACE** | Evolving playbooks | 59.4% AppWorld |
| **IBM CUGA** | Similar to ACE | Comparable |
| **LangGraph** | trim_messages, SummarizationNode | Rule-based, loses info |

---

## Primary Benchmarks

### SCBench (PRIORITY 1 - Multi-Turn)

**Why it matters**: Only benchmark specifically testing compression across multiple turns.

| Aspect | Details |
|--------|---------|
| **Paper** | arXiv:2412.10319 |
| **Tasks** | 12 tasks testing KV-cache lifecycle |
| **Key metric** | Performance degradation per turn |
| **What it tests** | String retrieval, semantic retrieval, global info, multi-task |

**Evaluation protocol:**
```python
def evaluate_scbench(compressor, model, n_turns=10):
    """
    Test compression robustness across multiple turns.

    Key: Same compressed context used for N follow-up queries.
    Measures how much performance degrades over turns.
    """
    results = []
    for scenario in scbench.scenarios:
        # Compress shared context ONCE
        compressed = compressor.compress(scenario.context)

        turn_scores = []
        for turn, query in enumerate(scenario.queries[:n_turns]):
            response = model.query(compressed, query)
            score = evaluate(response, scenario.answers[turn])
            turn_scores.append(score)

        results.append({
            "turn_1": turn_scores[0],
            "turn_5": turn_scores[4] if len(turn_scores) > 4 else None,
            "turn_10": turn_scores[9] if len(turn_scores) > 9 else None,
            "degradation": turn_scores[0] - turn_scores[-1]
        })

    return aggregate(results)
```

**Success criteria:**
- Turn 1: >95% (match baselines)
- Turn 10: >90% (beat baselines by 40%+)
- Degradation: <5% (vs 50%+ for competitors)

---

### LongBench (Single-Turn Baseline)

**Why it matters**: Standard benchmark, ensures we don't regress on single-turn.

| Aspect | Details |
|--------|---------|
| **Paper** | arXiv:2308.14508 (ACL 2024) |
| **Tasks** | 21 tasks across 6 categories |
| **Categories** | Single-doc QA, multi-doc QA, summarization, few-shot, code, synthetic |

**Evaluation protocol:**
```python
def evaluate_longbench(compressor, model, ratios=[0.3, 0.5, 0.7]):
    """
    Standard compression evaluation on LongBench.
    """
    results = []
    for task in longbench.tasks:
        for ratio in ratios:
            for sample in task.samples:
                compressed = compressor.compress(
                    sample.context,
                    query=sample.question,
                    target_ratio=ratio
                )
                response = model.query(compressed, sample.question)
                score = evaluate(response, sample.answer, task.metric)

                results.append({
                    "task": task.name,
                    "ratio": ratio,
                    "score": score,
                    "original_tokens": len(sample.context),
                    "compressed_tokens": len(compressed)
                })

    return aggregate_by_task_and_ratio(results)
```

**Success criteria:**
- 50% compression: >95% retention
- 30% compression: >90% retention
- Beat LLMLingua-2 on at least half the tasks

---

### NoLiMa (Semantic Understanding)

**Why it matters**: Tests if compression preserves meaning, not just keywords.

| Aspect | Details |
|--------|---------|
| **Paper** | arXiv:2502.05167 (ICML 2025) |
| **Key feature** | No lexical overlap between question and answer |
| **What it tests** | True semantic understanding after compression |

**Why NoLiMa is hard:**
- Can't just keep tokens that match the question
- Must understand which content is semantically relevant
- Pattern matching fails completely

**Success criteria:**
- 32K context: >85% (GPT-4o gets 69.7%)
- Must beat LLMLingua-2 significantly (it relies on lexical overlap)

---

### HotpotQA (Multi-Hop Reasoning)

**Why it matters**: Tests entity preservation across reasoning chains.

| Aspect | Details |
|--------|---------|
| **Task** | Multi-hop question answering |
| **Key challenge** | Must preserve entities needed for reasoning |
| **RECOMP baseline** | 2-3 EM loss at 6% compression |

**Success criteria:**
- <2 EM loss at 6% compression
- Entity retention: >98%

---

## Success Criteria (Phased)

### MVP (Month 1-2)

Minimum viable results to continue the project.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Single-turn (LongBench) | >93% retention at 50% | Match existing methods |
| Multi-turn (SCBench T5) | <10% drop | Show multi-turn promise |
| Latency | <100ms | Usable in practice |
| Entity retention | >90% | Basic entity preservation |

**Exit criteria:** If we can't hit >85% single-turn, revisit approach.

---

### Parity (Month 3-4)

Match SOTA on single-turn, beat on multi-turn.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Single-turn (LongBench) | >95% retention at 50% | Match LLMLingua-2 |
| Multi-turn (SCBench T10) | <10% drop | Clear differentiation |
| HotpotQA | <2 EM loss at 6% | Beat RECOMP |
| Latency | <75ms | Faster than LLMLingua-2 |
| Entity retention | >95% | Strong entity preservation |

**Exit criteria:** If multi-turn isn't significantly better than baselines, pivot approach.

---

### Leadership (Month 5-6)

Best multi-turn compression, competitive single-turn.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Single-turn (LongBench) | >95% retention | Within 2% of SOTA |
| Multi-turn (SCBench T10) | >90% absolute | **40%+ better than any baseline** |
| NoLiMa 32K | >85% | Semantic understanding |
| Latency | <50ms scoring | **2x faster than LLMLingua-2** |
| Entity retention | >98% | Near-perfect entity preservation |

**Publication criteria:** Clear wins on multi-turn, competitive single-turn, open weights.

---

## Ablation Experiments

### Multi-Turn Training Ablations

| Experiment | Hypothesis | Metric |
|------------|------------|--------|
| Single-turn only vs multi-turn training data | Multi-turn data critical | SCBench T10 |
| Turn history window (0, 3, 5, 10 turns) | More history helps, diminishing returns | SCBench degradation |
| Entity tracking on/off | Entities reduce drift | Entity retention at T10 |
| Referenced-token bonus (0, 0.1, 0.3) | Previously referenced tokens matter | SCBench T10 |

### Entity Preservation Ablations

| Experiment | Hypothesis | Metric |
|------------|------------|--------|
| Entity loss weight (0, 0.1, 0.3, 0.5) | 0.3 is sweet spot | HotpotQA EM |
| Entity types (all vs named only) | Named entities most critical | F1 on entity preservation |
| Entity tracking across turns | Drift detection helps | SCBench multi-hop tasks |

### Latency Ablations

| Experiment | Hypothesis | Metric |
|------------|------------|--------|
| Base model (DistilBERT vs TinyBERT vs MiniLM) | Smaller faster, accuracy tradeoff | Latency vs retention |
| Quantization (FP32 vs FP16 vs INT8) | INT8 viable with <1% loss | Latency, retention |
| Token scorer only vs full pipeline | Scorer sufficient for most cases | Retention vs latency |

---

## Evaluation Infrastructure

### Directory Structure

```
reet/
├── benchmarks/
│   ├── scbench/
│   │   ├── evaluate.py      # Main evaluation script
│   │   ├── data.py          # Data loading
│   │   └── metrics.py       # Multi-turn metrics
│   ├── longbench/
│   │   ├── evaluate.py
│   │   ├── data.py
│   │   └── tasks/           # Per-task configs
│   ├── nolima/
│   │   ├── evaluate.py
│   │   └── data.py
│   ├── hotpotqa/
│   │   ├── evaluate.py
│   │   └── data.py
│   └── baselines/
│       ├── llmlingua.py     # LLMLingua-2 wrapper
│       ├── longllmlingua.py # Question-aware variant
│       ├── recomp.py        # RECOMP wrapper
│       ├── truncation.py    # Simple baseline
│       └── compare.py       # Run all baselines
```

### Running Benchmarks

```bash
# Quick validation (subset)
python -m reet.benchmarks.scbench --quick --model gpt-4o-mini

# Full SCBench (multi-turn focus)
python -m reet.benchmarks.scbench --n-turns 10 --output results/scbench/

# Full LongBench
python -m reet.benchmarks.longbench --output results/longbench/

# Compare against all baselines
python -m reet.benchmarks.compare --benchmarks scbench,longbench --output results/comparison/

# Generate report
python -m reet.benchmarks.report --input results/ --output report.md
```

### CI Integration

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  quick-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: pip install -e .[benchmarks]

      - name: SCBench Quick
        run: python -m reet.benchmarks.scbench --quick

      - name: Check Multi-Turn Retention
        run: |
          python -c "
          import json
          results = json.load(open('results/scbench_quick.json'))
          t10_retention = results['turn_10_avg']
          assert t10_retention > 0.85, f'Turn 10 retention {t10_retention} below 85%'
          print(f'Turn 10 retention: {t10_retention:.1%}')
          "
```

---

## Results Presentation

### Primary Results Table (for README)

```markdown
## Benchmark Results

### Multi-Turn Robustness (SCBench) - PRIMARY METRIC

| Method | Turn 1 | Turn 5 | Turn 10 | Degradation |
|--------|--------|--------|---------|-------------|
| LLMLingua-2 | 92% | 68% | 43% | -53% |
| SnapKV | 95% | 72% | 48% | -49% |
| **REET** | **95%** | **93%** | **91%** | **-4%** |

### Single-Turn Compression (LongBench)

| Method | 50% Compression | 30% Compression | Latency |
|--------|-----------------|-----------------|---------|
| LLMLingua-2 | 95.2% | 89.1% | 100ms |
| RECOMP | 93.8% | 86.5% | 500ms |
| **REET** | **95.5%** | **91.2%** | **<50ms** |

*Tested with GPT-4o-mini. Full results: [benchmarks/results/](./benchmarks/results/)*
```

### Visualization

```python
# Generate multi-turn degradation plot
import matplotlib.pyplot as plt

def plot_multiturn_degradation(results):
    """Show how methods degrade over turns."""
    fig, ax = plt.subplots(figsize=(10, 6))

    turns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for method, scores in results.items():
        ax.plot(turns, scores, label=method, marker='o')

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Multi-Turn Compression Robustness")
    ax.legend()
    ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90% target')
    ax.set_ylim(30, 100)

    plt.savefig("results/multiturn_degradation.png")
```

---

## References

### Benchmark Papers
- SCBench: [arXiv:2412.10319](https://arxiv.org/abs/2412.10319)
- LongBench: [arXiv:2308.14508](https://arxiv.org/abs/2308.14508) (ACL 2024)
- NoLiMa: [arXiv:2502.05167](https://arxiv.org/abs/2502.05167) (ICML 2025)
- HotpotQA: [EMNLP 2018](https://hotpotqa.github.io/)

### Baseline Papers
- LLMLingua-2: [ACL 2024](https://aclanthology.org/2024.acl-long.91.pdf)
- RECOMP: [ICLR 2024](https://arxiv.org/abs/2310.04408)
- DynamicKV: [arXiv:2412.14838](https://arxiv.org/abs/2412.14838)
- SnapKV: [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)

### Repositories
- SCBench: https://github.com/microsoft/SCBench
- LongBench: https://github.com/THUDM/LongBench
- NoLiMa: https://github.com/adobe-research/NoLiMa
- LLMLingua: https://github.com/microsoft/LLMLingua
