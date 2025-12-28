"""Baseline compression methods for comparison."""

from reet.benchmarks.baselines.base import BaseCompressor, CompressorRegistry
from reet.benchmarks.baselines.truncation import TruncationCompressor

# Import LLMLingua compressors (requires llmlingua package)
try:
    from reet.benchmarks.baselines.llmlingua import (
        LLMLingua2Compressor,
        LongLLMLinguaCompressor,
    )
    _HAS_LLMLINGUA = True
except ImportError:
    _HAS_LLMLINGUA = False
    LLMLingua2Compressor = None
    LongLLMLinguaCompressor = None

__all__ = [
    "BaseCompressor",
    "CompressorRegistry",
    "TruncationCompressor",
    "LLMLingua2Compressor",
    "LongLLMLinguaCompressor",
]
