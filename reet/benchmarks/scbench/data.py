"""SCBench data loading utilities."""

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class Turn:
    """A single turn in multi-turn conversation."""

    input: str  # The query/question for this turn
    answer: str | list[str]  # Expected answer(s)
    options: list[str] | None = None  # For multiple choice tasks


@dataclass
class SCBenchSample:
    """A single SCBench sample with multi-turn structure."""

    id: str
    context: str  # The long context (document, code, etc.)
    turns: list[Turn]  # Multi-turn queries and answers
    task: str  # Task type (kv, prefix_suffix, qa_eng, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


# Available SCBench datasets
SCBENCH_DATASETS = [
    "scbench_kv",  # Key-value retrieval
    "scbench_prefix_suffix",  # Prefix-suffix matching
    "scbench_vt",  # Variable tracking
    "scbench_repoqa",  # Repository QA
    "scbench_qa_eng",  # English QA
    "scbench_qa_chn",  # Chinese QA
    "scbench_choice_eng",  # English multiple choice
    "scbench_many_shot",  # Many-shot learning
    "scbench_summary",  # Summarization
    "scbench_summary_with_needles",  # Summary + needle retrieval
    "scbench_mf",  # Multi-field extraction
    "scbench_repoqa_and_kv",  # Combined task
]


class SCBenchDataLoader:
    """Load SCBench datasets from HuggingFace."""

    def __init__(self, cache_dir: str | None = None):
        """Initialize data loader.

        Args:
            cache_dir: Optional cache directory for datasets.
        """
        self.cache_dir = cache_dir

    def load(
        self,
        dataset_name: str = "scbench_kv",
        split: str = "test",
        limit: int | None = None,
    ) -> list[SCBenchSample]:
        """Load a specific SCBench dataset.

        Args:
            dataset_name: Name of the SCBench dataset (e.g., "scbench_kv").
            split: Data split to load.
            limit: Maximum number of samples to load.

        Returns:
            List of SCBenchSample objects.
        """
        if dataset_name not in SCBENCH_DATASETS:
            available = ", ".join(SCBENCH_DATASETS)
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

        # Load from HuggingFace
        dataset = load_dataset(
            "microsoft/SCBench",
            dataset_name,
            split=split,
            cache_dir=self.cache_dir,
        )

        samples = []
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break

            # Parse multi-turn structure
            turns = []
            for turn_data in item.get("multi_turns", []):
                turn = Turn(
                    input=turn_data.get("input", ""),
                    answer=turn_data.get("answer", ""),
                    options=turn_data.get("options"),
                )
                turns.append(turn)

            sample = SCBenchSample(
                id=item.get("id", str(i)),
                context=item.get("context", ""),
                turns=turns,
                task=dataset_name,
                metadata={
                    "original_index": i,
                },
            )
            samples.append(sample)

        return samples

    def load_all(
        self,
        split: str = "test",
        limit_per_dataset: int | None = None,
    ) -> dict[str, list[SCBenchSample]]:
        """Load all SCBench datasets.

        Args:
            split: Data split to load.
            limit_per_dataset: Maximum samples per dataset.

        Returns:
            Dict mapping dataset name to list of samples.
        """
        all_samples = {}
        for dataset_name in SCBENCH_DATASETS:
            try:
                samples = self.load(dataset_name, split, limit_per_dataset)
                all_samples[dataset_name] = samples
            except Exception as e:
                print(f"Warning: Failed to load {dataset_name}: {e}")
                continue
        return all_samples

    @staticmethod
    def list_datasets() -> list[str]:
        """List available SCBench datasets."""
        return SCBENCH_DATASETS.copy()
