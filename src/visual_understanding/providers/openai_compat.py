"""OpenAI-compatible provider.

Covers any endpoint that follows the OpenAI Chat Completions wire format:
``POST {base_url}/chat/completions`` with ``image_url`` content blocks.

This includes OpenAI, Zhipu (GLM-V), Azure OpenAI, Together, Groq, and local
servers (vLLM, Ollama, LM Studio) that expose an OpenAI-compatible API.

Zhipu extensions (``video_url``, ``file_url`` content-block types) are emitted
when the provider config has ``supports_video`` / ``supports_files`` enabled.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import ProviderConfig
from .base import ChatResult, ContentBlock, VisionProvider


class OpenAICompatProvider(VisionProvider):
    """Provider for any OpenAI-compatible chat completions endpoint."""

    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self._endpoint = f"{config.base_url.rstrip('/')}/chat/completions"

    def _translate_content(
        self, content: list[ContentBlock]
    ) -> list[dict[str, Any]]:
        """Translate normalised blocks to the OpenAI content-block format."""
        out: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                out.append({"type": "text", "text": block["text"]})
            elif btype == "image":
                out.append(
                    {"type": "image_url", "image_url": {"url": block["url"]}}
                )
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
                    "content": self._translate_content(content),
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
