"""SCBench evaluation logic for multi-turn compression testing."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from tqdm import tqdm

from reet.benchmarks.baselines.base import BaseCompressor
from reet.benchmarks.metrics import compute_f1, normalize_answer
from reet.benchmarks.scbench.data import SCBenchDataLoader, SCBenchSample


@dataclass
class TurnResult:
    """Result for a single turn."""

    turn_idx: int
    prediction: str
    ground_truth: str | list[str]
    score: float
    compression_ratio: float


@dataclass
class SampleResult:
    """Result for a full multi-turn sample."""

    sample_id: str
    task: str
    turn_results: list[TurnResult]
    avg_score: float
    degradation: float  # score_turn_1 - score_last_turn


@dataclass
class SCBenchResult:
    """Aggregated SCBench results."""

    dataset_name: str
    compressor_name: str
    target_ratio: float
    num_samples: int

    # Per-turn accuracy (the key metric)
    turn_accuracies: dict[int, float]  # turn_idx -> accuracy

    # Overall metrics
    avg_accuracy: float
    degradation: float  # (turn_1 - last_turn) / turn_1

    # Detailed results
    sample_results: list[SampleResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SCBenchEvaluator:
    """Evaluator for SCBench multi-turn benchmark."""

    def __init__(
        self,
        llm_client: Any,  # OpenAI or Anthropic client
        model: str = "gpt-5.2",
        provider: str = "openai",
    ):
        """Initialize evaluator.

        Args:
            llm_client: LLM client (OpenAI or Anthropic).
            model: Model name to use for evaluation.
            provider: "openai" or "anthropic".
        """
        self.llm_client = llm_client
        self.model = model
        self.provider = provider

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt and return response."""
        if self.provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=512,  # GPT-5.2 uses max_completion_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(
        self,
        context: str,
        query: str,
        task: str,
        previous_qa: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build prompt for a turn.

        Args:
            context: The (possibly compressed) context.
            query: Current turn's query.
            task: Task type for appropriate formatting.
            previous_qa: List of (question, answer) from previous turns.

        Returns:
            Formatted prompt string.
        """
        # Task-specific prompt templates
        if "kv" in task:
            template = (
                "Extract the value corresponding to the specified key in the JSON object below.\n\n"
                "{context}\n\n"
                "Key: {query}\n"
                "Value:"
            )
        elif "qa" in task:
            template = (
                "Answer the question based on the given context.\n\n"
                "Context: {context}\n\n"
                "Question: {query}\n"
                "Answer:"
            )
        elif "summary" in task:
            template = (
                "Based on the following content, answer the question.\n\n"
                "{context}\n\n"
                "Question: {query}\n"
                "Answer:"
            )
        elif "choice" in task:
            template = (
                "Answer the multiple choice question based on the context.\n\n"
                "Context: {context}\n\n"
                "Question: {query}\n"
                "Answer (just the letter):"
            )
        else:
            # Default template
            template = (
                "{context}\n\n"
                "Question: {query}\n"
                "Answer:"
            )

        # Add previous Q&A for multi-turn context
        if previous_qa:
            qa_history = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in previous_qa
            )
            context = f"{context}\n\nPrevious conversation:\n{qa_history}"

        return template.format(context=context, query=query)

    def _score_prediction(
        self,
        prediction: str,
        ground_truth: str | list[str],
        task: str,
    ) -> float:
        """Score a prediction against ground truth.

        Args:
            prediction: Model prediction.
            ground_truth: Expected answer(s).
            task: Task type for appropriate scoring.

        Returns:
            Score between 0 and 1.
        """
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        # KV tasks: exact match after normalization
        if "kv" in task:
            pred_norm = normalize_answer(prediction)
            for gt in ground_truth:
                if normalize_answer(gt) == pred_norm:
                    return 1.0
            return 0.0

        # Choice tasks: first letter match
        if "choice" in task:
            pred_letter = re.search(r"[A-D]", prediction.upper())
            if pred_letter:
                pred_letter = pred_letter.group()
                for gt in ground_truth:
                    gt_letter = re.search(r"[A-D]", gt.upper())
                    if gt_letter and gt_letter.group() == pred_letter:
                        return 1.0
            return 0.0

        # Default: F1 score
        return compute_f1(prediction, ground_truth)

    def evaluate_sample(
        self,
        sample: SCBenchSample,
        compressor: BaseCompressor,
        target_ratio: float = 0.5,
    ) -> SampleResult:
        """Evaluate a single multi-turn sample.

        Args:
            sample: SCBench sample with multi-turn structure.
            compressor: Compressor to use.
            target_ratio: Target compression ratio.

        Returns:
            SampleResult with per-turn scores.
        """
        turn_results = []
        previous_qa: list[tuple[str, str]] = []

        for turn_idx, turn in enumerate(sample.turns):
            # Compress context (query-aware if supported)
            compressed = compressor.compress(
                context=sample.context,
                query=turn.input,
                target_ratio=target_ratio,
            )

            # Build prompt with compressed context
            prompt = self._build_prompt(
                context=compressed.compressed_text,
                query=turn.input,
                task=sample.task,
                previous_qa=previous_qa,
            )

            # Get LLM response
            prediction = self._call_llm(prompt)

            # Score the prediction
            score = self._score_prediction(prediction, turn.answer, sample.task)

            turn_results.append(TurnResult(
                turn_idx=turn_idx,
                prediction=prediction,
                ground_truth=turn.answer,
                score=score,
                compression_ratio=compressed.compression_ratio,
            ))

            # Track Q&A history for next turn
            previous_qa.append((turn.input, prediction))

        # Calculate summary metrics
        scores = [r.score for r in turn_results]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Degradation: how much did accuracy drop from turn 1 to last turn
        if len(scores) >= 2:
            degradation = scores[0] - scores[-1]
        else:
            degradation = 0.0

        return SampleResult(
            sample_id=sample.id,
            task=sample.task,
            turn_results=turn_results,
            avg_score=avg_score,
            degradation=degradation,
        )

    def evaluate(
        self,
        samples: list[SCBenchSample],
        compressor: BaseCompressor,
        target_ratio: float = 0.5,
        show_progress: bool = True,
    ) -> SCBenchResult:
        """Evaluate compressor on SCBench samples.

        Args:
            samples: List of SCBench samples.
            compressor: Compressor to evaluate.
            target_ratio: Target compression ratio.
            show_progress: Show progress bar.

        Returns:
            SCBenchResult with aggregated metrics.
        """
        sample_results = []
        iterator = tqdm(samples, desc=f"Evaluating {compressor.name}") if show_progress else samples

        for sample in iterator:
            # Reset stateful compressors between samples (e.g., query-chained)
            if hasattr(compressor, "reset"):
                compressor.reset()

            result = self.evaluate_sample(sample, compressor, target_ratio)
            sample_results.append(result)

        # Aggregate per-turn accuracies
        turn_scores: dict[int, list[float]] = {}
        for result in sample_results:
            for turn_result in result.turn_results:
                if turn_result.turn_idx not in turn_scores:
                    turn_scores[turn_result.turn_idx] = []
                turn_scores[turn_result.turn_idx].append(turn_result.score)

        turn_accuracies = {
            turn_idx: sum(scores) / len(scores)
            for turn_idx, scores in turn_scores.items()
        }

        # Overall metrics
        all_scores = [r.avg_score for r in sample_results]
        avg_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0

        # Degradation across turns
        if turn_accuracies:
            first_turn = turn_accuracies.get(0, 0)
            last_turn = turn_accuracies.get(max(turn_accuracies.keys()), 0)
            if first_turn > 0:
                degradation = (first_turn - last_turn) / first_turn
            else:
                degradation = 0.0
        else:
            degradation = 0.0

        return SCBenchResult(
            dataset_name=samples[0].task if samples else "unknown",
            compressor_name=compressor.name,
            target_ratio=target_ratio,
            num_samples=len(samples),
            turn_accuracies=turn_accuracies,
            avg_accuracy=avg_accuracy,
            degradation=degradation,
            sample_results=sample_results,
        )


def run_scbench_evaluation(
    compressor: BaseCompressor,
    dataset_name: str = "scbench_kv",
    target_ratio: float = 0.5,
    limit: int | None = None,
    llm_client: Any = None,
    model: str = "gpt-5.2",
    provider: str = "openai",
) -> SCBenchResult:
    """Convenience function to run SCBench evaluation.

    Args:
        compressor: Compressor to evaluate.
        dataset_name: SCBench dataset to use.
        target_ratio: Target compression ratio.
        limit: Max samples to evaluate.
        llm_client: LLM client (will create OpenAI client if None).
        model: Model name for LLM.
        provider: LLM provider.

    Returns:
        SCBenchResult with evaluation results.
    """
    # Load data
    loader = SCBenchDataLoader()
    samples = loader.load(dataset_name, limit=limit)

    # Create LLM client if not provided
    if llm_client is None:
        if provider == "openai":
            from openai import OpenAI
            llm_client = OpenAI()
        elif provider == "anthropic":
            from anthropic import Anthropic
            llm_client = Anthropic()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Run evaluation
    evaluator = SCBenchEvaluator(llm_client, model, provider)
    return evaluator.evaluate(samples, compressor, target_ratio)
