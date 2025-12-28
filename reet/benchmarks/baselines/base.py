"""Base class for compression methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressedResult:
    """Result from a compressor."""

    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseCompressor(ABC):
    """Abstract base class for context compressors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the compressor for reporting."""
        pass

    @abstractmethod
    def compress(
        self,
        context: str,
        query: str | None = None,
        target_ratio: float = 0.5,
    ) -> CompressedResult:
        """Compress context to target ratio.

        Args:
            context: The text to compress.
            query: Optional query for query-aware compression.
            target_ratio: Target compression ratio (0.5 = keep 50% of tokens).

        Returns:
            CompressedResult with compressed text and statistics.
        """
        pass

    def compress_multi_turn(
        self,
        contexts: list[str],
        query: str | None = None,
        target_ratio: float = 0.5,
    ) -> list[CompressedResult]:
        """Compress multiple turns of context.

        Default implementation compresses each turn independently.
        Subclasses can override for turn-aware compression.

        Args:
            contexts: List of context strings (one per turn).
            query: Optional query for query-aware compression.
            target_ratio: Target compression ratio.

        Returns:
            List of CompressedResult, one per turn.
        """
        return [self.compress(ctx, query, target_ratio) for ctx in contexts]


class CompressorRegistry:
    """Registry of available compressors."""

    _compressors: dict[str, type[BaseCompressor]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a compressor class."""

        def decorator(compressor_cls: type[BaseCompressor]):
            cls._compressors[name] = compressor_cls
            return compressor_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseCompressor:
        """Get a compressor instance by name."""
        if name not in cls._compressors:
            available = ", ".join(cls._compressors.keys())
            raise ValueError(f"Unknown compressor: {name}. Available: {available}")
        return cls._compressors[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered compressor names."""
        return list(cls._compressors.keys())
