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
    base_url: str = Field(
        default="",
        description=(
            "API base URL (no trailing slash). May be left empty if base_url_env "
            "or the generic VISUAL_UNDERSTANDING_BASE_URL env var provides it."
        ),
    )
    base_url_env: str | None = Field(
        default=None,
        description=(
            "Name of the environment variable holding the base URL. Lets the MCP "
            "client inject the endpoint via its `env` field. Inline base_url takes "
            "precedence."
        ),
    )
    requires_auth: bool = Field(
        default=True,
        description=(
            "Whether an API key is required. Set False for auth-less OpenAI-"
            "compatible endpoints (local vLLM/Ollama, internal proxies) so no "
            "Authorization header is sent."
        ),
    )
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
        """Where the key comes from — 'config file', 'env var', 'not required', or 'missing'."""
        if not self.requires_auth:
            return "not required"
        if self.api_key:
            return "config file"
        if self.api_key_env and os.environ.get(self.api_key_env):
            return f"env var {self.api_key_env}"
        return "missing"

    @property
    def base_url_value(self) -> str:
        """Resolve the base URL: inline ``base_url``, then ``base_url_env``, then generic env."""
        if self.base_url:
            return self.base_url
        if self.base_url_env:
            return os.environ.get(self.base_url_env, "")
        return os.environ.get("VISUAL_UNDERSTANDING_BASE_URL", "")

    @property
    def base_url_source(self) -> str:
        """Where the base URL comes from (for diagnostics)."""
        if self.base_url:
            return "config file"
        if self.base_url_env and os.environ.get(self.base_url_env):
            return f"env var {self.base_url_env}"
        if os.environ.get("VISUAL_UNDERSTANDING_BASE_URL"):
            return "env var VISUAL_UNDERSTANDING_BASE_URL"
        return "missing"

    @property
    def is_configured(self) -> bool:
        """Whether the provider is usable: base URL present and key present (or not required)."""
        if not self.base_url_value:
            return False
        if not self.requires_auth:
            return True
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

# Generic env vars that auto-register a 'custom' OpenAI-compatible provider.
# The MCP client can inject these via its `env` field to point at any
# third-party OpenAI-compatible vision endpoint without touching config.yaml.
_ENV_BASE_URL = "VISUAL_UNDERSTANDING_BASE_URL"
_ENV_API_KEY = "VISUAL_UNDERSTANDING_API_KEY"
_ENV_MODEL = "VISUAL_UNDERSTANDING_MODEL"


def _apply_custom_provider(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge the generic ``VISUAL_UNDERSTANDING_*`` env vars into a 'custom' provider.

    If ``VISUAL_UNDERSTANDING_BASE_URL`` is set, a provider named ``custom`` is
    ensured (created or updated — explicit env values win over config.yaml).
    ``VISUAL_UNDERSTANDING_API_KEY`` is optional: when absent the provider runs
    auth-less (``requires_auth: False``).
    """
    base_url = os.environ.get(_ENV_BASE_URL)
    if not base_url:
        return raw

    api_key = os.environ.get(_ENV_API_KEY)
    model = os.environ.get(_ENV_MODEL)

    providers = dict(raw.get("providers", {}))
    custom = dict(providers.get("custom", {}))
    custom["type"] = custom.get("type", "openai_compat")
    # Reference the env var so diagnostics show the correct source
    custom["base_url_env"] = _ENV_BASE_URL
    custom.pop("base_url", None)
    if api_key:
        custom["api_key_env"] = _ENV_API_KEY
        custom.pop("api_key", None)
        custom["requires_auth"] = True
    else:
        custom.pop("api_key", None)
        custom.pop("api_key_env", None)
        custom["requires_auth"] = False
    if model:
        custom["default_chat_model"] = model
        if model not in custom.get("chat_models", []):
            custom["chat_models"] = [model] + list(custom.get("chat_models", []))
    providers["custom"] = custom

    result = dict(raw)
    result["providers"] = providers
    return result


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

    The default config includes Zhipu, OpenAI, Anthropic, and third-party OpenAI-
    compatible presets (DashScope / SiliconFlow / OpenRouter). A user config file
    completely replaces the defaults (merge is intentionally not done to keep
    semantics simple — copy what you need from ``config.example.yaml``).

    If the generic ``VISUAL_UNDERSTANDING_BASE_URL`` env var is set, a ``custom``
    provider is merged in (see :func:`_apply_custom_provider`).
    """
    path = _find_config_path()
    if path is None:
        raw = dict(DEFAULT_CONFIG)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    raw = _apply_custom_provider(raw)
    return AppConfig(**raw)
