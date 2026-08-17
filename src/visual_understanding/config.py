"""Configuration models and YAML loading for visual-understanding.

Providers are defined in a YAML config file. API keys are ALWAYS referenced by
environment variable name (``api_key_env``) — never hardcoded in the config file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# --- Pydantic models ---------------------------------------------------------


class ProviderConfig(BaseModel):
    """Configuration for a single vision provider.

    API keys can be supplied either inline via ``api_key`` or by referencing an
    environment variable via ``api_key_env`` (inline takes precedence). Inline
    keys are convenient for MCP setups (one config file configures everything)
    but keep the file out of version control.
    """

    type: Literal["openai_compat", "anthropic"] = "openai_compat"
    api_key_env: str | None = Field(
        default=None,
        description="Name of the environment variable holding the API key.",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "API key configured directly in the config file (takes precedence "
            "over api_key_env). WARNING: keep this file out of version control."
        ),
    )
    base_url: str = Field(..., description="API base URL (no trailing slash expected).")
    chat_models: list[str] = Field(
        default_factory=list, description="Available vision/chat models."
    )
    default_chat_model: str = Field("", description="Default model for chat tasks.")
    supports_video: bool = False
    supports_files: bool = False
    native_grounding: bool = Field(
        False, description="Provider supports native grounding with normalized coords."
    )
    max_images: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)
    model_protocols: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-model wire protocol override: {model_name: 'openai' | 'anthropic'}. "
            "Lets a single provider gateway route individual models to either the "
            "OpenAI-compatible /chat/completions or the Anthropic /messages format "
            "(e.g. OpenCode Go serves some models via each)."
        ),
    )
    images_require_base64: bool = Field(
        False,
        description=(
            "When True, remote image URLs are fetched and re-encoded as base64 "
            "data URLs before sending. Some gateways (e.g. OpenCode Go) reject "
            "remote image URLs and only accept base64."
        ),
    )
    model_defaults: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-model parameter overrides that always apply, e.g. "
            "{model: {temperature: 1, max_tokens: 8192}}. Used for hard model "
            "constraints like OpenCode Go models that only accept temperature=1."
        ),
    )

    @property
    def api_key_value(self) -> str | None:
        """Resolve the API key: inline ``api_key`` first, then ``api_key_env``."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    @property
    def key_source(self) -> str:
        """Where the key comes from — 'config file', 'env var', or 'missing'."""
        if self.api_key:
            return "config file"
        if self.api_key_env and os.environ.get(self.api_key_env):
            return f"env var {self.api_key_env}"
        return "missing"

    @property
    def is_configured(self) -> bool:
        """Whether an API key is present."""
        return bool(self.api_key_value)

    @property
    def capabilities(self) -> set[str]:
        caps = {"chat"}
        if self.supports_video:
            caps.add("video")
        if self.supports_files:
            caps.add("files")
        if self.native_grounding:
            caps.add("grounding")
        return caps


class AppConfig(BaseModel):
    """Top-level application configuration."""

    default_provider: str = "zhipu"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    def get_provider(self, name: str | None = None) -> tuple[str, ProviderConfig]:
        """Return (name, config) for the requested or default provider.

        Raises ValueError if the provider does not exist.
        """
        name = name or self.default_provider
        if name not in self.providers:
            available = ", ".join(self.providers) or "(none)"
            raise ValueError(
                f"Provider '{name}' not found in config. Available: {available}"
            )
        return name, self.providers[name]


# --- Default config -----------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "default_provider": "zhipu",
    "providers": {
        "zhipu": {
            "type": "openai_compat",
            "api_key_env": "ZHIPU_API_KEY",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "chat_models": [
                "glm-5v",
                "glm-5v-turbo",
                "glm-4.6v",
                "glm-4.6v-flash",
                "glm-4.6v-flashx",
            ],
            "default_chat_model": "glm-5v-turbo",
            "supports_video": True,
            "supports_files": True,
            "native_grounding": True,
            "max_images": 50,
        },
        "openai": {
            "type": "openai_compat",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "chat_models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
            "default_chat_model": "gpt-4o",
            "max_images": 10,
        },
        "anthropic": {
            "type": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com/v1",
            "chat_models": [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-3-5-sonnet-20241022",
            ],
            "default_chat_model": "claude-sonnet-4-20250514",
            "max_images": 20,
        },
        # --- Third-party OpenAI-compatible vision providers (presets) ---
        "dashscope": {
            "type": "openai_compat",
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "chat_models": [
                "qwen-vl-max",
                "qwen-vl-plus",
                "qwen2.5-vl-72b-instruct",
                "qwen2.5-vl-32b-instruct",
                "qwen2.5-vl-7b-instruct",
            ],
            "default_chat_model": "qwen-vl-max",
            "max_images": 10,
        },
        "siliconflow": {
            "type": "openai_compat",
            "api_key_env": "SILICONFLOW_API_KEY",
            "base_url": "https://api.siliconflow.cn/v1",
            "chat_models": [
                "Qwen/Qwen2.5-VL-72B-Instruct",
                "Qwen/Qwen2.5-VL-32B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ],
            "default_chat_model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "max_images": 10,
        },
        "openrouter": {
            "type": "openai_compat",
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "chat_models": [
                "qwen/qwen2.5-vl-72b-instruct",
                "openai/gpt-4o",
                "google/gemini-2.0-flash-001",
            ],
            "default_chat_model": "qwen/qwen2.5-vl-72b-instruct",
            "max_images": 10,
        },
    },
}

# --- Config loading -----------------------------------------------------------

_ENV_VAR = "VISUAL_UNDERSTANDING_CONFIG"
_USER_CONFIG_PATH = Path.home() / ".config" / "visual-understanding" / "config.yaml"


def _find_config_path() -> Path | None:
    """Three-level config file lookup.

    1. ``VISUAL_UNDERSTANDING_CONFIG`` env var
    2. ``~/.config/visual-understanding/config.yaml``
    3. (falls back to built-in defaults — returns None)
    """
    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    if _USER_CONFIG_PATH.exists():
        return _USER_CONFIG_PATH

    return None


def load_config() -> AppConfig:
    """Load configuration from YAML file, falling back to built-in defaults.

    The default config includes Zhipu, OpenAI, and Anthropic. A user config file
    completely replaces the defaults (merge is intentionally not done to keep
    semantics simple — copy what you need from ``config.example.yaml``).
    """
    path = _find_config_path()
    if path is None:
        return AppConfig(**DEFAULT_CONFIG)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(**raw)
