"""Simple truncation baseline - keep last N tokens."""

import tiktoken

from reet.benchmarks.baselines.base import BaseCompressor, CompressedResult, CompressorRegistry


@CompressorRegistry.register("truncation")
class TruncationCompressor(BaseCompressor):
    """Truncation baseline that keeps the last N tokens.

    This is the simplest baseline - just truncate from the beginning,
    keeping the most recent context. Often surprisingly effective for
    tasks where recent context matters most.
    """

    def __init__(self, model: str = "gpt-4"):
        """Initialize truncation compressor.

        Args:
            model: Model name for tiktoken encoding.
        """
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (GPT-4 / GPT-3.5-turbo encoding)
            self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "truncation"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoding.encode(text))

    def compress(
        self,
        context: str,
        query: str | None = None,
        target_ratio: float = 0.5,
    ) -> CompressedResult:
        """Truncate context to target ratio (keep last N tokens).

        Args:
            context: The text to compress.
            query: Ignored (truncation is query-agnostic).
            target_ratio: Target compression ratio (0.5 = keep 50% of tokens).

        Returns:
            CompressedResult with truncated text.
        """
        tokens = self._encoding.encode(context)
        original_count = len(tokens)

        # Calculate target token count
        target_count = int(original_count * target_ratio)
        target_count = max(1, target_count)  # Keep at least 1 token

        if target_count >= original_count:
            # No truncation needed
            return CompressedResult(
                compressed_text=context,
                original_tokens=original_count,
                compressed_tokens=original_count,
                compression_ratio=1.0,
            )

        # Keep last N tokens (most recent context)
        truncated_tokens = tokens[-target_count:]
        compressed_text = self._encoding.decode(truncated_tokens)

        return CompressedResult(
            compressed_text=compressed_text,
            original_tokens=original_count,
            compressed_tokens=len(truncated_tokens),
            compression_ratio=len(truncated_tokens) / original_count,
        )


@CompressorRegistry.register("truncation_start")
class TruncationStartCompressor(TruncationCompressor):
    """Truncation baseline that keeps the first N tokens.

    Alternative truncation strategy - keep the beginning of the context.
    Useful for comparison and for tasks where initial context is more important.
    """

    @property
    def name(self) -> str:
        return "truncation_start"

    def compress(
        self,
        context: str,
        query: str | None = None,
        target_ratio: float = 0.5,
    ) -> CompressedResult:
        """Truncate context to target ratio (keep first N tokens).

        Args:
            context: The text to compress.
            query: Ignored (truncation is query-agnostic).
            target_ratio: Target compression ratio.

        Returns:
            CompressedResult with truncated text.
        """
        tokens = self._encoding.encode(context)
        original_count = len(tokens)

        target_count = int(original_count * target_ratio)
        target_count = max(1, target_count)

        if target_count >= original_count:
            return CompressedResult(
                compressed_text=context,
                original_tokens=original_count,
                compressed_tokens=original_count,
                compression_ratio=1.0,
            )

        # Keep first N tokens
        truncated_tokens = tokens[:target_count]
        compressed_text = self._encoding.decode(truncated_tokens)

        return CompressedResult(
            compressed_text=compressed_text,
            original_tokens=original_count,
            compressed_tokens=len(truncated_tokens),
            compression_ratio=len(truncated_tokens) / original_count,
        )
