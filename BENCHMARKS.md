# REET Benchmarks

## Baseline Methods

### Hard Compression (Our Focus)

| Method | Level | Query-Aware | Approach | Priority |
|--------|-------|-------------|----------|----------|
| **Truncation** | Token | No | Keep first N tokens | P0 (baseline) |
| **LLMLingua-2** | Token | No | BERT token classification | P0 |
| **LongLLMLingua** | Token | Yes | Contrastive perplexity | P0 |
| **QUITO-X** | Word | Yes | T5 cross-attention + IB | P0 (SOTA) |
| **EXIT** | Sentence | Yes | Gemma sentence classification | P1 |
| **RECOMP-Ext** | Sentence | Yes | Contriever embeddings | P1 |
| **Selective Context** | Token | No | Self-information pruning | P2 |
| **RECOMP-Abs** | Abstractive | Yes | T5 summarization | P2 |

### Soft Compression (Future Work)

| Method | Approach | Compression |
|--------|----------|-------------|
| ICAE | In-context autoencoder | 4x-8x |
| 500xCompressor | KV-based ICAE | Higher |
| AutoCompressors | Learned summary tokens | Variable |

## Datasets

### Phase 1: Single-Turn QA (Task-Aware)

| Dataset | Type | Size | Has Labels | Notes |
|---------|------|------|------------|-------|
| **NaturalQuestions** | Single-hop QA | 307K | Answer spans | Most common benchmark |
| **TriviaQA** | Single-hop QA | 650K | Answers | Large scale, diverse |
| **HotpotQA** | Multi-hop QA | 113K | Supporting facts | Can derive entity labels |
| **SQuAD** | Reading comprehension | 100K | Answer spans | Standard benchmark |
| **2WikiMultiHopQA** | Multi-hop QA | 200K | Supporting facts | Multi-hop reasoning |
| **MuSiQue** | Multi-hop QA | 40K | Supporting facts | Complex reasoning |

### Phase 2: Task-Agnostic

| Dataset | Type | Avg Tokens | Notes |
|---------|------|------------|-------|
| **LongBench** | Multi-task (6 categories) | 10K | Standard long-context |
| **ZeroSCROLLS** | Multi-task (10 tasks) | 10K | Zero-shot evaluation |
| **LooGLE** | Long-context understanding | 24K | Very long contexts |

### Phase 3: Multi-Turn

| Dataset | Type | Notes |
|---------|------|-------|
| **SCBench** | Multi-turn QA | Tests degradation over turns |
| **CoQA** | Conversational QA | 127K examples |

### Dataset Details

#### NaturalQuestions
- **Source:** Google
- **Format:** Question + Wikipedia passage + answer span
- **Split:** Train 307K, Dev 8K
- **Used by:** LongLLMLingua, RECOMP, EXIT, QUITO

#### TriviaQA
- **Source:** University of Washington
- **Format:** Question + evidence documents + answer
- **Split:** Train 138K, Dev 18K (with documents: 650K)
- **Used by:** RECOMP, EXIT

#### HotpotQA
- **Source:** CMU + Stanford
- **Format:** Question + 10 paragraphs + answer + supporting facts
- **Why important:** Has sentence-level supporting fact annotations
- **Used by:** RECOMP, EXIT, QUITO-X

#### SQuAD
- **Source:** Stanford
- **Format:** Question + passage + answer span
- **Split:** Train 87K, Dev 10K
- **Used by:** QUITO-X

#### LongBench
- **Categories:** Single-doc QA, Multi-doc QA, Summarization, Few-shot, Synthetic, Code
- **Average length:** ~10K tokens
- **Used by:** LLMLingua-2, LongLLMLingua

#### SCBench
- **Focus:** Multi-turn context accumulation
- **Measures:** Accuracy degradation over turns
- **Tasks:** KV retrieval, multi-needle, variable tracking
- **Used by:** Our multi-turn evaluation

## Evaluation Metrics

### Primary Metrics

| Metric | Description | Used For |
|--------|-------------|----------|
| **Exact Match (EM)** | Exact string match | QA tasks |
| **F1 Score** | Token-level overlap | QA tasks |
| **Compression Ratio** | compressed_len / original_len | All tasks |
| **Accuracy @ Ratio** | EM/F1 at specific compression | Comparison |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Degradation** | (uncompressed_acc - compressed_acc) / uncompressed_acc |
| **Inference Time** | ms per context |
| **Multi-turn Drop** | Accuracy drop from turn 1 to turn N |

## Experimental Setup

### Compression Ratios to Test

```
ratios = [0.7, 0.5, 0.3, 0.2, 0.1]  # Keep 70%, 50%, 30%, 20%, 10%
```

### LLM Backends

| Provider | Models | Notes |
|----------|--------|-------|
| OpenAI | gpt-4o, gpt-4o-mini | Primary |
| Anthropic | claude-3.5-sonnet | Secondary |

### Evaluation Protocol

1. Load dataset (N samples)
2. For each compressor:
   - For each compression ratio:
     - Compress all contexts
     - Query LLM with compressed context
     - Compute EM/F1
     - Record compression stats
3. Compare across methods

## Running Benchmarks

### Quick Test

```bash
python scripts/run_scbench.py --quick
```

### Single Dataset

```bash
python scripts/run_scbench.py \
    --dataset scbench_kv \
    --limit 100 \
    --compressors truncation,llmlingua2 \
    --ratios 0.5,0.3 \
    --output results/scbench_kv.json
```

### Full Evaluation

```bash
python scripts/run_scbench.py \
    --all \
    --limit 500 \
    --compressors truncation,llmlingua2,longllmlingua \
    --ratios 0.7,0.5,0.3,0.2 \
    --output results/full_eval.json
```

## Expected Results

Based on published papers:

| Method | NQ EM @ 0.5 | TriviaQA EM @ 0.5 | HotpotQA F1 @ 0.5 |
|--------|-------------|-------------------|-------------------|
| No compression | ~45% | ~65% | ~60% |
| Truncation | ~30% | ~45% | ~40% |
| LLMLingua-2 | ~40% | ~58% | ~52% |
| LongLLMLingua | ~42% | ~60% | ~55% |
| QUITO-X | ~44% | ~63% | ~58% |
| EXIT | ~43% | ~62% | ~57% |

**Our target:** Match or exceed QUITO-X at same compression ratio.

## Baseline Implementation Status

| Baseline | Status | Location |
|----------|--------|----------|
| Truncation | Done | `reet/benchmarks/baselines/truncation.py` |
| LLMLingua-2 | Done | `reet/benchmarks/baselines/llmlingua.py` |
| LongLLMLingua | Done | `reet/benchmarks/baselines/llmlingua.py` |
| QUITO-X | Planned | - |
| EXIT | Planned | - |
| GLiNER2 baseline | Planned | - |
| REET-Extractor | Planned | - |
