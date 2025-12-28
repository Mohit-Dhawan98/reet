"""SCBench: Multi-turn context compression benchmark.

SCBench tests how compression methods degrade across multiple conversation turns.
This is the PRIMARY benchmark for REET - it directly measures multi-turn robustness.

Paper: https://arxiv.org/abs/2412.10319
Data: https://huggingface.co/datasets/microsoft/SCBench
"""

from reet.benchmarks.scbench.data import SCBenchDataLoader, SCBenchSample
from reet.benchmarks.scbench.evaluate import SCBenchEvaluator

__all__ = [
    "SCBenchDataLoader",
    "SCBenchSample",
    "SCBenchEvaluator",
]
