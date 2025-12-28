"""Common metrics for benchmark evaluation."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison.

    Lower text, remove punctuation, articles, and extra whitespace.
    Standard normalization used by SQuAD, HotpotQA, etc.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str | list[str]) -> float:
    """Compute exact match score.

    Args:
        prediction: Model prediction.
        ground_truth: Ground truth answer(s).

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    pred_normalized = normalize_answer(prediction)
    for gt in ground_truth:
        if normalize_answer(gt) == pred_normalized:
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truth: str | list[str]) -> float:
    """Compute token-level F1 score.

    Args:
        prediction: Model prediction.
        ground_truth: Ground truth answer(s).

    Returns:
        F1 score between 0 and 1.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    pred_tokens = normalize_answer(prediction).split()

    best_f1 = 0.0
    for gt in ground_truth:
        gt_tokens = normalize_answer(gt).split()

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(gt_tokens) if gt_tokens else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        best_f1 = max(best_f1, f1)

    return best_f1


def compute_accuracy(predictions: list[str], ground_truths: list[str | list[str]]) -> float:
    """Compute accuracy over a list of predictions.

    Args:
        predictions: List of model predictions.
        ground_truths: List of ground truth answer(s).

    Returns:
        Accuracy between 0 and 1.
    """
    if not predictions:
        return 0.0

    correct = sum(
        compute_exact_match(pred, gt) for pred, gt in zip(predictions, ground_truths)
    )
    return correct / len(predictions)


def compute_retention(
    compressed_accuracy: float,
    baseline_accuracy: float,
) -> float:
    """Compute accuracy retention after compression.

    Args:
        compressed_accuracy: Accuracy with compressed context.
        baseline_accuracy: Accuracy with full context (no compression).

    Returns:
        Retention ratio (compressed / baseline).
    """
    if baseline_accuracy == 0:
        return 0.0
    return compressed_accuracy / baseline_accuracy


def compute_compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """Compute compression ratio.

    Args:
        original_tokens: Number of tokens before compression.
        compressed_tokens: Number of tokens after compression.

    Returns:
        Compression ratio (compressed / original).
    """
    if original_tokens == 0:
        return 1.0
    return compressed_tokens / original_tokens


def contains_answer(prediction: str, ground_truth: str | list[str]) -> bool:
    """Check if prediction contains the answer (for retrieval tasks).

    Args:
        prediction: Model prediction/generated text.
        ground_truth: Ground truth answer(s).

    Returns:
        True if any ground truth is contained in prediction.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    pred_normalized = normalize_answer(prediction)
    for gt in ground_truth:
        if normalize_answer(gt) in pred_normalized:
            return True
    return False
