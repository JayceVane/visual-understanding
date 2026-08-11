"""Provider registry and factory.

Maps provider ``type`` strings from the config to concrete :class:`VisionProvider`
classes. Adding a new provider type means registering it here.
"""

from __future__ import annotations

from ..config import AppConfig, ProviderConfig
from .base import ChatResult, ContentBlock, VisionProvider
from .anthropic import AnthropicProvider
from .openai_compat import OpenAICompatProvider

# Type → provider class mapping
_PROVIDER_CLASSES: dict[str, type[VisionProvider]] = {
    "openai_compat": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
}


def create_provider(name: str, config: ProviderConfig) -> VisionProvider:
    """Instantiate a provider from its config.

    Raises ValueError if the provider ``type`` is unknown.
    """
    cls = _PROVIDER_CLASSES.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unknown provider type '{config.type}' for provider '{name}'. "
            f"Known types: {', '.join(_PROVIDER_CLASSES)}"
        )
    return cls(name, config)


def get_provider(
    app_config: AppConfig, name: str | None = None
) -> tuple[str, VisionProvider]:
    """Resolve a provider by name (or the default) from the app config.

    Returns ``(name, provider_instance)``.
    Raises ValueError if the provider does not exist.
    """
    resolved_name, cfg = app_config.get_provider(name)
    return resolved_name, create_provider(resolved_name, cfg)


__all__ = [
    "VisionProvider",
    "ChatResult",
    "ContentBlock",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "create_provider",
    "get_provider",
]
