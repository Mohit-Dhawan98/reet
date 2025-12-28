#!/usr/bin/env python3
"""Run SCBench evaluation on baseline compressors.

Usage:
    python scripts/run_scbench.py --quick  # Quick test with 5 samples
    python scripts/run_scbench.py --dataset scbench_kv --limit 50
    python scripts/run_scbench.py --all --limit 20  # All datasets, 20 samples each
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables from .env
load_dotenv()

from reet.benchmarks.baselines import (
    CompressorRegistry,
    TruncationCompressor,
    LLMLingua2Compressor,
    LongLLMLinguaCompressor,
)
from reet.benchmarks.scbench import SCBenchDataLoader, SCBenchEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Run SCBench evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="scbench_kv",
        help="SCBench dataset to evaluate on",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate on all SCBench datasets",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 5 samples",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.5,0.3",
        help="Comma-separated compression ratios to test",
    )
    parser.add_argument(
        "--compressors",
        type=str,
        default="truncation",
        help="Comma-separated compressor names",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="LLM model for evaluation (gpt-5.2, gpt-5.2-chat-latest, gpt-5.2-pro)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    return parser.parse_args()


def format_turn_accuracies(turn_accuracies: dict[int, float]) -> str:
    """Format turn accuracies as a string."""
    if not turn_accuracies:
        return "N/A"
    sorted_turns = sorted(turn_accuracies.items())
    return " → ".join(f"T{t+1}:{acc*100:.1f}%" for t, acc in sorted_turns)


def main():
    args = parse_args()

    # Quick test mode
    if args.quick:
        args.limit = 5

    # Parse ratios and compressors
    ratios = [float(r) for r in args.ratios.split(",")]
    compressor_names = args.compressors.split(",")

    # Initialize compressors
    # Force registration by importing
    _ = TruncationCompressor  # noqa
    if LLMLingua2Compressor:
        _ = LLMLingua2Compressor  # noqa
    if LongLLMLinguaCompressor:
        _ = LongLLMLinguaCompressor  # noqa

    compressors = []
    for name in compressor_names:
        try:
            compressor = CompressorRegistry.get(name)
            compressors.append(compressor)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    if not compressors:
        print("No valid compressors found. Available:", CompressorRegistry.list_available())
        return

    # Initialize LLM client
    if args.provider == "openai":
        from openai import OpenAI
        llm_client = OpenAI()
    else:
        from anthropic import Anthropic
        llm_client = Anthropic()

    # Initialize evaluator
    evaluator = SCBenchEvaluator(llm_client, args.model, args.provider)
    loader = SCBenchDataLoader()

    # Determine datasets to evaluate
    if args.all:
        datasets = loader.list_datasets()
    else:
        datasets = [args.dataset]

    # Run evaluation
    all_results = []
    results_table = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            samples = loader.load(dataset_name, limit=args.limit)
            print(f"Loaded {len(samples)} samples")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            continue

        for compressor in compressors:
            for ratio in ratios:
                print(f"\n--- {compressor.name} @ {ratio*100:.0f}% ---")

                result = evaluator.evaluate(
                    samples=samples,
                    compressor=compressor,
                    target_ratio=ratio,
                )

                # Store result
                all_results.append({
                    "dataset": dataset_name,
                    "compressor": compressor.name,
                    "target_ratio": ratio,
                    "num_samples": result.num_samples,
                    "turn_accuracies": result.turn_accuracies,
                    "avg_accuracy": result.avg_accuracy,
                    "degradation": result.degradation,
                })

                # Format for table
                results_table.append([
                    dataset_name,
                    compressor.name,
                    f"{ratio*100:.0f}%",
                    format_turn_accuracies(result.turn_accuracies),
                    f"{result.avg_accuracy*100:.1f}%",
                    f"{result.degradation*100:.1f}%",
                ])

                print(f"Turn Accuracies: {format_turn_accuracies(result.turn_accuracies)}")
                print(f"Avg Accuracy: {result.avg_accuracy*100:.1f}%")
                print(f"Degradation: {result.degradation*100:.1f}%")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    headers = ["Dataset", "Compressor", "Ratio", "Turn Accuracies", "Avg", "Degradation"]
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "args": vars(args),
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
