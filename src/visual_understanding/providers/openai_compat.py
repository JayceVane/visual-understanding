"""OpenAI-compatible provider.

Covers any endpoint that follows the OpenAI Chat Completions wire format:
``POST {base_url}/chat/completions`` with ``image_url`` content blocks.

This includes OpenAI, Zhipu (GLM-V), Azure OpenAI, Together, Groq, and local
servers (vLLM, Ollama, LM Studio) that expose an OpenAI-compatible API.

Zhipu extensions (``video_url``, ``file_url`` content-block types) are emitted
when the provider config has ``supports_video`` / ``supports_files`` enabled.

**Mixed protocols:** gateways like OpenCode Go serve some models via the OpenAI
format and others via the Anthropic ``/messages`` format under one base URL.
Set ``model_protocols: {model: 'openai' | 'anthropic'}`` in the provider config
and the correct wire format is used per model automatically.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import ProviderConfig
from ..media import fetch_image_as_data_url
from .anthropic import url_to_anthropic_image_block
from .base import ChatResult, ContentBlock, VisionProvider

_ANTHROPIC_API_VERSION = "2023-06-01"


class OpenAICompatProvider(VisionProvider):
    """Provider for any OpenAI-compatible chat completions endpoint."""

    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self._endpoint = f"{config.base_url.rstrip('/')}/chat/completions"

    async def _translate_content(
        self, content: list[ContentBlock]
    ) -> list[dict[str, Any]]:
        """Translate normalised blocks to the OpenAI content-block format."""
        out: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                out.append({"type": "text", "text": block["text"]})
            elif btype == "image":
                url = block["url"]
                # Some gateways (OpenCode Go) reject remote URLs — re-encode as base64
                if self.config.images_require_base64 and not url.startswith("data:"):
                    url = await fetch_image_as_data_url(url)
                out.append({"type": "image_url", "image_url": {"url": url}})
            elif btype == "video":
                if not self.config.supports_video:
                    raise ValueError(
                        f"Provider '{self.name}' does not support video input."
                    )
                out.append(
                    {"type": "video_url", "video_url": {"url": block["url"]}}
                )
            elif btype == "file":
                if not self.config.supports_files:
                    raise ValueError(
                        f"Provider '{self.name}' does not support file input."
                    )
                out.append(
                    {"type": "file_url", "file_url": {"url": block["url"]}}
                )
            else:
                raise ValueError(f"Unknown content block type: {btype!r}")
        return out

    async def chat(
        self,
        content: list[ContentBlock],
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> ChatResult:
        api_key = self.ensure_configured()
        resolved_model = self.resolve_model(model)

        # Apply per-model parameter overrides (hard model constraints)
        defaults = self.config.model_defaults.get(resolved_model, {})
        if "temperature" in defaults:
            temperature = defaults["temperature"]
        if "max_tokens" in defaults:
            max_tokens = defaults["max_tokens"]

        # Mixed-protocol gateway: route to Anthropic /messages format if configured
        if self.config.model_protocols.get(resolved_model) == "anthropic":
            return await self._chat_anthropic_format(
                api_key, resolved_model, content, temperature, max_tokens
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {
                    "role": "user",
                    "content": await self._translate_content(content),
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self._endpoint, headers=headers, json=payload
                )
        except httpx.RequestError as exc:
            return ChatResult(success=False, error=f"Network error: {exc}")

        if resp.status_code in (401, 403):
            return ChatResult(
                success=False,
                error=self._format_error(resp, "Authentication failed"),
            )

        if resp.status_code == 429:
            return ChatResult(
                success=False,
                error=self._format_error(resp, "Rate limit exceeded"),
            )

        if resp.status_code != 200:
            return ChatResult(
                success=False,
                error=self._format_error(resp, f"API error ({resp.status_code})"),
            )

        try:
            data = resp.json()
        except Exception as exc:
            return ChatResult(success=False, error=f"Failed to parse response: {exc}")

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        finish_reason = choice.get("finish_reason")

        result = ChatResult(
            success=True,
            text=text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
            usage=data.get("usage", {}),
            finish_reason=finish_reason,
        )

        # Zhipu safety review flag
        if finish_reason == "sensitive":
            result.finish_reason = finish_reason

        return result

    async def _chat_anthropic_format(
        self,
        api_key: str,
        model: str,
        content: list[ContentBlock],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        """Call the Anthropic ``/messages`` format for a mixed-protocol gateway model."""
        # Translate content blocks to Anthropic format (images → base64 source)
        translated: list[dict[str, Any]] = []
        try:
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    translated.append({"type": "text", "text": block["text"]})
                elif btype == "image":
                    translated.append(await url_to_anthropic_image_block(block["url"]))
                else:
                    return ChatResult(
                        success=False,
                        error=(
                            f"Model '{model}' (Anthropic protocol) does not "
                            f"support {btype} input."
                        ),
                    )
        except httpx.RequestError as exc:
            return ChatResult(success=False, error=f"Failed to fetch image: {exc}")

        endpoint = f"{self.config.base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": translated}],
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
        except httpx.RequestError as exc:
            return ChatResult(success=False, error=f"Network error: {exc}")

        if resp.status_code in (401, 403):
            return ChatResult(
                success=False,
                error=self._format_error(resp, "Authentication failed"),
            )
        if resp.status_code == 429:
            return ChatResult(
                success=False,
                error=self._format_error(resp, "Rate limit exceeded"),
            )
        if resp.status_code != 200:
            return ChatResult(
                success=False,
                error=self._format_error(resp, f"API error ({resp.status_code})"),
            )

        try:
            data = resp.json()
        except Exception as exc:
            return ChatResult(success=False, error=f"Failed to parse response: {exc}")

        # Anthropic returns content as a list of blocks; concatenate text blocks.
        text_parts: list[str] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "".join(text_parts)

        return ChatResult(
            success=True,
            text=text,
            usage={
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            finish_reason=data.get("stop_reason"),
        )

    @staticmethod
    def _format_error(resp: httpx.Response, prefix: str) -> str:
        """Best-effort extraction of an error message from a non-200 response."""
        try:
            body = resp.json()
            if isinstance(body, dict):
                err = body.get("error", body)
                if isinstance(err, dict):
                    msg = err.get("message", str(err))
                else:
                    msg = str(err)
                return f"{prefix}: {msg}"
        except Exception:
            pass
        return f"{prefix}: {resp.text[:200]}"
