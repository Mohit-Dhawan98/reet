"""Base classes for benchmark evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """A single benchmark sample."""

    id: str
    context: str
    query: str
    ground_truth: str | list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedResult:
    """Result from a compressor."""

    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result from evaluating a single sample."""

    sample_id: str
    prediction: str
    ground_truth: str | list[str]
    is_correct: bool
    score: float
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""

    benchmark_name: str
    compressor_name: str
    target_ratio: float
    actual_ratio: float
    accuracy: float
    exact_match: float | None = None
    f1: float | None = None
    retention: float | None = None  # accuracy / baseline_accuracy
    num_samples: int = 0
    results: list[EvaluationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTurnResult:
    """Results for multi-turn evaluation (e.g., SCBench)."""

    benchmark_name: str
    compressor_name: str
    target_ratio: float
    turn_results: dict[int, BenchmarkResult]  # turn_number -> result
    degradation: float  # (turn_1_accuracy - final_turn_accuracy) / turn_1_accuracy
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the benchmark."""
        pass

    @abstractmethod
    def load_data(self, split: str = "test", limit: int | None = None) -> list[Sample]:
        """Load benchmark samples.

        Args:
            split: Data split to load (train/validation/test).
            limit: Maximum number of samples to load (for quick testing).

        Returns:
            List of Sample objects.
        """
        pass

    @abstractmethod
    def evaluate_sample(
        self,
        sample: Sample,
        prediction: str,
    ) -> EvaluationResult:
        """Evaluate a single sample prediction.

        Args:
            sample: The benchmark sample.
            prediction: Model prediction.

        Returns:
            EvaluationResult with scores.
        """
        pass

    def evaluate(
        self,
        samples: list[Sample],
        predictions: list[str],
    ) -> BenchmarkResult:
        """Evaluate all predictions.

        Args:
            samples: List of benchmark samples.
            predictions: List of model predictions.

        Returns:
            Aggregated BenchmarkResult.
        """
        if len(samples) != len(predictions):
            raise ValueError(
                f"Mismatch: {len(samples)} samples vs {len(predictions)} predictions"
            )

        results = []
        for sample, pred in zip(samples, predictions):
            result = self.evaluate_sample(sample, pred)
            results.append(result)

        # Aggregate metrics
        accuracy = sum(r.is_correct for r in results) / len(results) if results else 0
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        avg_ratio = (
            sum(r.compression_ratio for r in results) / len(results) if results else 0
        )

        return BenchmarkResult(
            benchmark_name=self.name,
            compressor_name="",  # Set by caller
            target_ratio=0.0,  # Set by caller
            actual_ratio=avg_ratio,
            accuracy=accuracy,
            num_samples=len(results),
            results=results,
        )
