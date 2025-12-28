"""LLMLingua-2 baseline wrapper.

LLMLingua-2 is our primary baseline to beat. It achieves 20x compression
on single-turn tasks but degrades significantly in multi-turn settings.

Paper: https://aclanthology.org/2024.acl-long.91.pdf
Code: https://github.com/microsoft/LLMLingua
"""

from reet.benchmarks.baselines.base import BaseCompressor, CompressedResult, CompressorRegistry


@CompressorRegistry.register("llmlingua2")
class LLMLingua2Compressor(BaseCompressor):
    """LLMLingua-2 prompt compression wrapper.

    Uses the microsoft/llmlingua-2-xlm-roberta-large-meetingbank model
    by default for best performance.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device: str = "cpu",  # Default to CPU for compatibility
    ):
        """Initialize LLMLingua-2 compressor.

        Args:
            model_name: HuggingFace model name for LLMLingua-2.
            device: Device to run on ("cuda" or "cpu").
        """
        try:
            from llmlingua import PromptCompressor
        except ImportError:
            raise ImportError(
                "llmlingua is required for LLMLingua2Compressor. "
                "Install with: pip install llmlingua"
            )

        self._compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "llmlingua2"

    def compress(
        self,
        context: str,
        query: str | None = None,
        target_ratio: float = 0.5,
    ) -> CompressedResult:
        """Compress context using LLMLingua-2.

        Args:
            context: The text to compress.
            query: Optional query (LLMLingua-2 is NOT query-aware by default).
            target_ratio: Target compression ratio (0.5 = keep 50% of tokens).

        Returns:
            CompressedResult with compressed text.
        """
        # LLMLingua uses 'rate' parameter (1.0 - compression_ratio)
        # rate=0.5 means keep 50% of tokens
        result = self._compressor.compress_prompt(
            context,
            rate=target_ratio,
            force_tokens=["\n", ".", "?", "!"],  # Preserve structure
        )

        return CompressedResult(
            compressed_text=result["compressed_prompt"],
            original_tokens=result["origin_tokens"],
            compressed_tokens=result["compressed_tokens"],
            compression_ratio=result["compressed_tokens"] / result["origin_tokens"],
            metadata={
                "rate": result.get("rate", target_ratio),
                "saving": result.get("saving", ""),
            },
        )


@CompressorRegistry.register("longllmlingua")
class LongLLMLinguaCompressor(BaseCompressor):
    """LongLLMLingua - query-aware variant.

    LongLLMLingua extends LLMLingua with question-aware compression,
    using the query to determine which parts of context are relevant.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device: str = "cpu",  # Default to CPU for compatibility
    ):
        """Initialize LongLLMLingua compressor.

        Args:
            model_name: HuggingFace model name.
            device: Device to run on.
        """
        try:
            from llmlingua import PromptCompressor
        except ImportError:
            raise ImportError(
                "llmlingua is required for LongLLMLinguaCompressor. "
                "Install with: pip install llmlingua"
            )

        self._compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "longllmlingua"

    def compress(
        self,
        context: str,
        query: str | None = None,
        target_ratio: float = 0.5,
    ) -> CompressedResult:
        """Compress context using LongLLMLingua (query-aware).

        Args:
            context: The text to compress.
            query: Query for relevance-based compression (recommended).
            target_ratio: Target compression ratio.

        Returns:
            CompressedResult with compressed text.
        """
        # Build prompt with query if provided
        if query:
            # LongLLMLingua mode: question-aware compression
            result = self._compressor.compress_prompt(
                context,
                question=query,
                rate=target_ratio,
                force_tokens=["\n", ".", "?", "!"],
                condition_in_question="after_condition",
                reorder_context="sort",
                dynamic_context_compression_ratio=0.3,
                condition_compare=True,
                context_budget="+100",
            )
        else:
            # Fallback to standard LLMLingua-2
            result = self._compressor.compress_prompt(
                context,
                rate=target_ratio,
                force_tokens=["\n", ".", "?", "!"],
            )

        return CompressedResult(
            compressed_text=result["compressed_prompt"],
            original_tokens=result["origin_tokens"],
            compressed_tokens=result["compressed_tokens"],
            compression_ratio=result["compressed_tokens"] / result["origin_tokens"],
            metadata={
                "rate": result.get("rate", target_ratio),
                "query_aware": query is not None,
            },
        )
