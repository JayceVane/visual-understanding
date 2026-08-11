"""Abstract vision provider interface.

All providers implement ``chat()`` which accepts a *normalized* content-block
list and returns a :class:`ChatResult`. The normalized format is provider-neutral;
each concrete provider translates it to its own wire format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config import ProviderConfig

# Normalised content block types ------------------------------------------------
#
# A content block is a plain dict:
#   {"type": "image", "url": "<https-url-or-data-url>"}
#   {"type": "video", "url": "<https-url>"}
#   {"type": "file",  "url": "<https-url>"}
#   {"type": "text",  "text": "..."}
ContentBlock = dict[str, Any]


@dataclass
class ChatResult:
    """Result of a multimodal chat completion."""

    success: bool
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None  # full API response for debugging


class VisionProvider(ABC):
    """Abstract base for vision providers.

    Concrete providers translate :data:`ContentBlock` lists into their own API
    format and return a normalised :class:`ChatResult`.
    """

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config

    # -- public API -----------------------------------------------------------

    @abstractmethod
    async def chat(
        self,
        content: list[ContentBlock],
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> ChatResult:
        """Send a multimodal chat request.

        Args:
            content: Normalised content blocks (images/videos/files + text).
            model: Model name (defaults to ``config.default_chat_model``).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            ChatResult with the model's text response or an error.
        """
        ...

    # -- shared helpers -------------------------------------------------------

    @property
    def capabilities(self) -> set[str]:
        return self.config.capabilities

    @property
    def is_configured(self) -> bool:
        """Whether the API key is available."""
        return self.config.is_configured

    def ensure_configured(self) -> str:
        """Return the API key or raise a helpful error."""
        key = self.config.api_key
        if not key:
            raise RuntimeError(
                f"Provider '{self.name}' is not configured. "
                f"Set the environment variable '{self.config.api_key_env}' with your API key."
            )
        return key

    def resolve_model(self, model: str | None) -> str:
        """Return *model* or the provider's default, validating against the known list."""
        m = model or self.config.default_chat_model
        if not m:
            raise ValueError(
                f"No model specified and provider '{self.name}' has no default_chat_model."
            )
        return m
