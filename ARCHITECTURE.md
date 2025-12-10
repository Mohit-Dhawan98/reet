# REET: Technical Architecture

## Core Concept

REET is the first learned compression designed for **multi-turn conversations**. The key insight: token importance should **compound over turns**, not reset.

While LLMLingua-2 and other methods treat each context independently, REET tracks what matters across an entire conversation.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Conversation Input                         │
│  Turn 1: [context + query + response]                           │
│  Turn 2: [context + query + response]                           │
│  ...                                                            │
│  Turn N: [context + query]  ← Current turn                      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 1: Entity Registry (Persistent)               │
│                                                                  │
│  Track entities across all turns:                               │
│  - Which entities were mentioned when                           │
│  - Which entities were referenced in answers                    │
│  - Entity co-occurrence and relationships                       │
│                                                                  │
│  Latency: <5ms (lookup only)                                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│         Stage 2: Turn-Aware Token Scorer (<30ms)                 │
│                                                                  │
│  Input: [context] + [turn history] + [entity mask]              │
│  Model: DistilBERT/TinyBERT fine-tuned (~66M params)            │
│  Output: Per-token importance scores that COMPOUND              │
│                                                                  │
│  Key innovation: tokens important in earlier turns get boosted  │
│  Training: Multi-turn conversation pairs with cross-turn labels │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│       Stage 3: Entity-Preserving Compressor (<50ms)              │
│                                                                  │
│  Input: High-importance tokens + entity preservation list       │
│  Model: T5-small/BART-small (~60M params)                       │
│  Output: Compressed context with entities guaranteed            │
│                                                                  │
│  Key innovation: Entity persistence loss + turn reference weight │
│  Training: (full, compressed) pairs with entity annotations     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Compressed Output                          │
│  - Context ready for LLM                                        │
│  - Entities from all turns preserved                            │
│  - <5% degradation even at turn 10                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Components

### 1. Turn-Aware Token Scorer

The key innovation: score importance considering **what was important before**.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TurnAwareTokenScorer(nn.Module):
    """
    Token importance scorer with multi-turn awareness.

    Key differences from LLMLingua-2:
    1. Turn position embeddings
    2. Entity mask boosting
    3. Compounding importance from previous turns
    """

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        max_turns: int = 20,
        entity_boost: float = 0.3
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        hidden_size = self.encoder.config.hidden_size

        # Turn position embedding
        self.turn_embedding = nn.Embedding(max_turns, hidden_size)

        # Importance prediction head
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        self.entity_boost = entity_boost

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        turn_positions: torch.Tensor = None,
        entity_mask: torch.Tensor = None,
        previous_importance: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Score token importance with turn awareness.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            turn_positions: Which turn each token is from [batch, seq_len]
            entity_mask: Binary mask for entity tokens [batch, seq_len]
            previous_importance: Importance scores from previous turns [batch, seq_len]

        Returns:
            Importance scores [batch, seq_len] in range [0, 1]
        """
        # Encode tokens
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Add turn position information
        if turn_positions is not None:
            turn_emb = self.turn_embedding(turn_positions)
            hidden_states = hidden_states + turn_emb

        # Predict base importance
        scores = self.importance_head(hidden_states).squeeze(-1)  # [batch, seq_len]

        # Boost entities
        if entity_mask is not None:
            scores = scores + entity_mask * self.entity_boost

        # Compound with previous importance (key innovation)
        if previous_importance is not None:
            # Tokens that were important before stay important
            scores = scores + previous_importance * 0.5
            scores = torch.clamp(scores, 0, 1)

        return scores

    def score_conversation(
        self,
        turns: list[dict],
        current_query: str
    ) -> tuple[torch.Tensor, dict]:
        """
        Score tokens across an entire conversation.

        Args:
            turns: List of {"content": str, "role": str} dicts
            current_query: The current query to optimize for

        Returns:
            Tuple of (importance_scores, metadata)
        """
        # Build full context with turn markers
        full_text = ""
        turn_positions = []

        for turn_idx, turn in enumerate(turns):
            start_pos = len(full_text)
            full_text += turn["content"] + " "
            end_pos = len(full_text)
            turn_positions.append((start_pos, end_pos, turn_idx))

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )

        # Map turn positions to token positions
        token_turns = self._map_turns_to_tokens(
            inputs, turn_positions
        )

        # Get entity mask
        entity_mask = self._get_entity_mask(full_text, inputs)

        # Score
        with torch.no_grad():
            scores = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
                turn_positions=token_turns,
                entity_mask=entity_mask
            )

        return scores, {
            "total_tokens": inputs["input_ids"].shape[1],
            "entities_found": entity_mask.sum().item()
        }
```

### 2. Entity Registry

Tracks entities across the entire conversation.

```python
from dataclasses import dataclass, field
from typing import Optional
import spacy

@dataclass
class EntityMention:
    """A single mention of an entity."""
    text: str
    turn: int
    position: tuple[int, int]  # start, end in turn
    was_in_answer: bool = False

@dataclass
class TrackedEntity:
    """An entity tracked across turns."""
    canonical_name: str
    mentions: list[EntityMention] = field(default_factory=list)
    answer_references: int = 0  # How many times referenced in answers

    @property
    def importance_weight(self) -> float:
        """Higher weight for entities referenced more in answers."""
        base = 1.0
        answer_bonus = 0.2 * self.answer_references
        recency_bonus = 0.1 * (len(self.mentions) > 0)
        return base + answer_bonus + recency_bonus


class EntityRegistry:
    """
    Track entities across a conversation.

    Key insight: entities referenced in answers are more important
    than entities just mentioned in context.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.entities: dict[str, TrackedEntity] = {}
        self.current_turn = 0

    def update(self, content: str, is_answer: bool = False) -> list[str]:
        """
        Extract and track entities from new content.

        Args:
            content: Text to extract entities from
            is_answer: Whether this is an LLM response (weights higher)

        Returns:
            List of entity names found
        """
        doc = self.nlp(content)
        found = []

        for ent in doc.ents:
            canonical = ent.text.lower().strip()

            if canonical not in self.entities:
                self.entities[canonical] = TrackedEntity(
                    canonical_name=canonical
                )

            mention = EntityMention(
                text=ent.text,
                turn=self.current_turn,
                position=(ent.start_char, ent.end_char),
                was_in_answer=is_answer
            )
            self.entities[canonical].mentions.append(mention)

            if is_answer:
                self.entities[canonical].answer_references += 1

            found.append(canonical)

        return found

    def advance_turn(self):
        """Move to next turn."""
        self.current_turn += 1

    def get_importance_weights(self) -> dict[str, float]:
        """Get importance weight for each tracked entity."""
        return {
            name: entity.importance_weight
            for name, entity in self.entities.items()
        }

    def get_must_preserve(self, threshold: float = 1.5) -> list[str]:
        """Get entities that must be preserved (high importance)."""
        return [
            name for name, entity in self.entities.items()
            if entity.importance_weight >= threshold
        ]

    def create_entity_mask(
        self,
        text: str,
        tokenizer
    ) -> torch.Tensor:
        """Create binary mask marking entity tokens."""
        tokens = tokenizer(text, return_offsets_mapping=True)
        mask = torch.zeros(len(tokens["input_ids"]))

        for name, entity in self.entities.items():
            # Find all occurrences in text
            start = 0
            while True:
                pos = text.lower().find(name, start)
                if pos == -1:
                    break

                # Mark tokens that overlap with this entity
                for idx, (token_start, token_end) in enumerate(tokens["offset_mapping"]):
                    if token_start < pos + len(name) and token_end > pos:
                        mask[idx] = entity.importance_weight
                start = pos + 1

        return mask
```

### 3. Entity-Preserving Compressor

Compresses while guaranteeing entity preservation.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class EntityPreservingCompressor(nn.Module):
    """
    Abstractive compressor with entity preservation guarantee.

    Key innovation: loss function that heavily penalizes
    losing entities, especially those referenced in answers.
    """

    def __init__(
        self,
        base_model: str = "t5-small",
        entity_loss_weight: float = 0.3
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.entity_loss_weight = entity_loss_weight

    def compress(
        self,
        context: str,
        target_ratio: float = 0.5,
        must_preserve: list[str] = None,
        entity_weights: dict[str, float] = None
    ) -> str:
        """
        Compress context while preserving entities.

        Args:
            context: Text to compress
            target_ratio: Target compression ratio
            must_preserve: Entities that MUST appear in output
            entity_weights: Importance weight per entity

        Returns:
            Compressed text
        """
        # Prepare input with compression instruction
        target_length = int(len(context.split()) * target_ratio)
        input_text = f"compress to {target_length} words: {context}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # Generate compressed version
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=target_length + 50,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        compressed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Verify and fix entity preservation
        if must_preserve:
            compressed = self._ensure_entities(
                compressed, must_preserve, context, entity_weights
            )

        return compressed

    def _ensure_entities(
        self,
        compressed: str,
        must_preserve: list[str],
        original: str,
        weights: dict[str, float] = None
    ) -> str:
        """Ensure critical entities are preserved."""
        missing = []
        for entity in must_preserve:
            if entity.lower() not in compressed.lower():
                weight = weights.get(entity, 1.0) if weights else 1.0
                missing.append((entity, weight))

        if missing:
            # Sort by weight (most important first)
            missing.sort(key=lambda x: x[1], reverse=True)

            # Extract context around missing entities from original
            supplements = []
            for entity, weight in missing[:5]:  # Max 5 supplements
                context_snippet = self._extract_entity_context(original, entity)
                if context_snippet:
                    supplements.append(context_snippet)

            if supplements:
                compressed += " [Key context: " + " | ".join(supplements) + "]"

        return compressed

    def _extract_entity_context(self, text: str, entity: str, window: int = 50) -> str:
        """Extract context around an entity mention."""
        pos = text.lower().find(entity.lower())
        if pos == -1:
            return None

        start = max(0, pos - window)
        end = min(len(text), pos + len(entity) + window)

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet


def compute_training_loss(
    model: EntityPreservingCompressor,
    batch: dict,
    entity_weight: float = 0.3
) -> torch.Tensor:
    """
    Training loss with entity preservation penalty.

    Loss = LM_loss + entity_weight * entity_loss

    entity_loss is higher for:
    - Entities referenced in more turns
    - Entities that appeared in answers
    """
    # Standard language modeling loss
    outputs = model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    lm_loss = outputs.loss

    # Entity preservation loss
    with torch.no_grad():
        generated = model.model.generate(
            batch["input_ids"],
            max_length=512
        )
        generated_text = model.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )

    entity_loss = torch.tensor(0.0)
    for gen, entities, weights in zip(
        generated_text,
        batch["entities"],
        batch["entity_weights"]
    ):
        for entity, weight in zip(entities, weights):
            if entity.lower() not in gen.lower():
                entity_loss += weight

    entity_loss = entity_loss / len(generated_text)

    return lm_loss + entity_weight * entity_loss
```

---

## Full Pipeline

```python
from dataclasses import dataclass

@dataclass
class CompressionResult:
    """Result of compression."""
    text: str
    original_tokens: int
    compressed_tokens: int
    entities_preserved: int
    entities_total: int
    compression_ratio: float


class REET:
    """
    Full REET compression pipeline.

    Latency breakdown:
    - Entity registry: <5ms (lookup)
    - Token scorer: <30ms
    - Compressor: <50ms (optional)
    - Total: <100ms
    """

    def __init__(
        self,
        token_scorer: TurnAwareTokenScorer,
        compressor: EntityPreservingCompressor,
        entity_registry: EntityRegistry = None
    ):
        self.token_scorer = token_scorer
        self.compressor = compressor
        self.entity_registry = entity_registry or EntityRegistry()

    @classmethod
    def from_pretrained(cls, model_name: str = "reet-base"):
        """Load pre-trained REET models from HuggingFace."""
        token_scorer = TurnAwareTokenScorer.from_pretrained(
            f"reet/{model_name}-scorer"
        )
        compressor = EntityPreservingCompressor.from_pretrained(
            f"reet/{model_name}-compressor"
        )
        return cls(token_scorer, compressor)

    def compress(
        self,
        context: str,
        query: str,
        turn_history: list[dict] = None,
        target_ratio: float = 0.5,
        use_abstractive: bool = True
    ) -> CompressionResult:
        """
        Compress context with multi-turn awareness.

        Args:
            context: Current context to compress
            query: Current query (for relevance scoring)
            turn_history: Previous turns for context
            target_ratio: Target compression (0.5 = 50%)
            use_abstractive: Whether to use compressor (slower but better)

        Returns:
            CompressionResult with compressed text and metrics
        """
        # Update entity registry
        self.entity_registry.update(context, is_answer=False)
        entity_weights = self.entity_registry.get_importance_weights()
        must_preserve = self.entity_registry.get_must_preserve()

        # Score tokens with turn awareness
        all_turns = (turn_history or []) + [{"content": context, "role": "context"}]
        scores, meta = self.token_scorer.score_conversation(all_turns, query)

        # Select top tokens based on scores
        keep_ratio = target_ratio if not use_abstractive else 0.7
        selected_text = self._select_by_score(context, scores, keep_ratio)

        # Optionally compress further with abstractive model
        if use_abstractive:
            final_text = self.compressor.compress(
                selected_text,
                target_ratio=target_ratio / keep_ratio,
                must_preserve=must_preserve,
                entity_weights=entity_weights
            )
        else:
            final_text = selected_text

        # Count preserved entities
        preserved = sum(
            1 for e in must_preserve
            if e.lower() in final_text.lower()
        )

        return CompressionResult(
            text=final_text,
            original_tokens=meta["total_tokens"],
            compressed_tokens=len(self.token_scorer.tokenizer.encode(final_text)),
            entities_preserved=preserved,
            entities_total=len(must_preserve),
            compression_ratio=len(final_text) / len(context)
        )

    def create_session(self) -> "REETSession":
        """Create a session for multi-turn compression."""
        return REETSession(self)


class REETSession:
    """
    Session manager for multi-turn conversations.

    Maintains state across turns for proper entity tracking
    and importance compounding.
    """

    def __init__(self, reet: REET):
        self.reet = reet
        self.turns: list[dict] = []
        self.metrics: list[dict] = []

    def compress(
        self,
        context: str,
        query: str,
        target_ratio: float = 0.5
    ) -> CompressionResult:
        """Compress with full conversation history."""
        result = self.reet.compress(
            context=context,
            query=query,
            turn_history=self.turns,
            target_ratio=target_ratio
        )

        self.metrics.append({
            "turn": len(self.turns) + 1,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "entities_preserved": result.entities_preserved
        })

        return result

    def add_response(self, response: str):
        """Add LLM response to history (for entity tracking)."""
        self.reet.entity_registry.update(response, is_answer=True)
        self.turns.append({"content": response, "role": "assistant"})
        self.reet.entity_registry.advance_turn()

    def add_context(self, context: str):
        """Add user context to history."""
        self.turns.append({"content": context, "role": "user"})

    def get_metrics_summary(self) -> dict:
        """Get summary of compression metrics across turns."""
        if not self.metrics:
            return {}

        return {
            "total_turns": len(self.metrics),
            "avg_compression": sum(m["compressed_tokens"] / m["original_tokens"]
                                   for m in self.metrics) / len(self.metrics),
            "entity_retention": sum(m["entities_preserved"]
                                    for m in self.metrics) / len(self.metrics)
        }
```

---

## Latency Targets

| Component | Target | Rationale |
|-----------|--------|-----------|
| Entity Registry | <5ms | Simple lookup/update |
| Token Scorer | **<30ms** | Must beat LLMLingua-2 (100ms) |
| Compressor | <50ms | Optional, adds quality |
| **Total** | **<100ms** | Real-time viable |

### Achieving <30ms Token Scoring

```python
# Options for faster scoring:

# 1. Use TinyBERT instead of DistilBERT
scorer = TurnAwareTokenScorer(base_model="huawei-noah/TinyBERT_General_4L_312D")
# 4 layers, 14.5M params, ~15ms

# 2. Use MiniLM
scorer = TurnAwareTokenScorer(base_model="microsoft/MiniLM-L6-H384-uncased")
# 6 layers, 22M params, ~20ms

# 3. Quantization
import torch
scorer = torch.quantization.quantize_dynamic(
    scorer, {nn.Linear}, dtype=torch.qint8
)
# ~40% speedup with <1% accuracy loss

# 4. ONNX export for production
import onnx
torch.onnx.export(scorer, dummy_input, "scorer.onnx")
# ~2x speedup
```

---

## Training Pipeline

### Multi-Turn Training Data

```python
def generate_multiturn_training_data(
    conversations: list[list[dict]],
    model: str = "gpt-4o"
) -> list[dict]:
    """
    Generate training data for turn-aware token scoring.

    Key: label importance based on what matters for LATER turns.
    """
    training_data = []

    for conversation in conversations:
        for turn_idx in range(1, len(conversation)):
            # Context: all previous turns
            context = " ".join(t["content"] for t in conversation[:turn_idx])
            query = conversation[turn_idx]["content"]
            answer = conversation[turn_idx].get("answer", "")

            # Get importance labels from frontier model
            importance = get_attention_based_importance(
                model, context, query, answer
            )

            training_data.append({
                "context": context,
                "query": query,
                "turn_idx": turn_idx,
                "importance_labels": importance,
                "entities": extract_entities(context),
                "answer_entities": extract_entities(answer)
            })

    return training_data
```

### Training Script

```python
def train_token_scorer(
    model: TurnAwareTokenScorer,
    train_data: list[dict],
    val_data: list[dict],
    epochs: int = 10,
    lr: float = 5e-5
):
    """Train the turn-aware token scorer."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in DataLoader(train_data, batch_size=16):
            optimizer.zero_grad()

            # Forward pass
            scores = model(
                batch["input_ids"],
                batch["attention_mask"],
                turn_positions=batch["turn_positions"],
                entity_mask=batch["entity_mask"]
            )

            # Binary cross-entropy with importance labels
            loss = F.binary_cross_entropy(
                scores,
                batch["importance_labels"]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_loss = evaluate(model, val_data)
        print(f"Epoch {epoch+1}: Train={total_loss:.4f}, Val={val_loss:.4f}")
```

---

## Project Structure

```
reet/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── token_scorer.py      # TurnAwareTokenScorer
│   ├── compressor.py        # EntityPreservingCompressor
│   └── base.py              # Base classes
├── core/
│   ├── __init__.py
│   ├── entity_registry.py   # EntityRegistry
│   ├── pipeline.py          # REET, REETSession
│   └── utils.py             # Helpers
├── training/
│   ├── __init__.py
│   ├── data.py              # Training data generation
│   ├── train_scorer.py      # Train token scorer
│   ├── train_compressor.py  # Train compressor
│   └── evaluate.py          # Training evaluation
├── benchmarks/
│   ├── __init__.py
│   ├── scbench/             # Multi-turn benchmark (PRIMARY)
│   ├── longbench/           # Single-turn benchmark
│   ├── nolima/              # Semantic retrieval
│   ├── hotpotqa/            # Entity preservation
│   └── baselines/           # LLMLingua-2, RECOMP wrappers
└── integrations/
    ├── __init__.py
    ├── langchain.py
    └── langgraph.py
```

---

## Dependencies

```toml
[project]
name = "reet"
version = "0.1.0"
requires-python = ">=3.9"

dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "tiktoken>=0.5.0",
    "spacy>=3.5.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
training = [
    "datasets>=2.14.0",
    "accelerate>=0.21.0",
    "wandb>=0.15.0",
    "openai>=1.0.0",
]
benchmarks = [
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
]
dev = [
    "pytest>=7.0.0",
    "ruff",
    "mypy",
]
```

---

## Future: Knowledge Graph Memory (v2)

```
┌─────────────────────────────────────────────────────────────────┐
│                   v2: Knowledge Graph Layer                      │
│                                                                  │
│  Stage 4: KG Memory (async, doesn't block main path)             │
│  - Extract entities and relations from context                   │
│  - Update temporal knowledge graph                               │
│  - Query graph for relevant historical context                   │
│  - Invalidate stale facts over time                             │
│                                                                  │
│  Inspired by: Zep, A-MEM, Graphiti                              │
└─────────────────────────────────────────────────────────────────┘
```
