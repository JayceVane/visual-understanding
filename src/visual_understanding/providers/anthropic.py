"""Anthropic Claude provider.

Claude's Messages API uses a different wire format from OpenAI:
  - ``POST {base_url}/messages``
  - Header ``x-api-key`` (not Bearer) + ``anthropic-version``
  - Images must be base64 ``source`` blocks — Claude does not accept image URLs
    directly, so http(s) URLs are fetched and re-encoded.

Video and file inputs are not supported by the Messages API and will raise.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from ..config import ProviderConfig
from .base import ChatResult, ContentBlock, VisionProvider

_API_VERSION = "2023-06-01"


class AnthropicProvider(VisionProvider):
    """Provider for Anthropic's Claude Messages API."""

    def __init__(self, name: str, config: ProviderConfig) -> None:
        super().__init__(name, config)
        self._endpoint = f"{config.base_url.rstrip('/')}/messages"

    async def _url_to_image_block(self, url: str) -> dict[str, Any]:
        """Convert an image URL (http or data:) to an Anthropic image source block."""
        # data: URL — parse directly, no network fetch
        if url.startswith("data:"):
            # Format: data:<media_type>;base64,<data>
            header, _, b64data = url.partition(",")
            media_type = "image/jpeg"
            if ";" in header:
                media_type = header.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64data,
                },
            }

        # http(s) URL — fetch and re-encode
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        media_type = "image/jpeg"
        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("image/"):
            media_type = ctype.split(";")[0]

        b64data = base64.b64encode(resp.content).decode()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64data,
            },
        }

    async def _translate_content(
        self, content: list[ContentBlock]
    ) -> list[dict[str, Any]]:
        """Translate normalised blocks to Anthropic's content format."""
        out: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                out.append({"type": "text", "text": block["text"]})
            elif btype == "image":
                out.append(await self._url_to_image_block(block["url"]))
            elif btype in ("video", "file"):
                raise ValueError(
                    f"Provider '{self.name}' (Anthropic) does not support {btype} input."
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

        try:
            translated = await self._translate_content(content)
        except httpx.RequestError as exc:
            return ChatResult(success=False, error=f"Failed to fetch image: {exc}")
        except ValueError as exc:
            return ChatResult(success=False, error=str(exc))

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        # Anthropic requires max_tokens and expects temperature at top level.
        payload: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": translated}],
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

        # Anthropic returns content as a list of blocks; concatenate text blocks.
        text_parts: list[str] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "".join(text_parts)

        stop_reason = data.get("stop_reason")

        return ChatResult(
            success=True,
            text=text,
            usage={
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            finish_reason=stop_reason,
        )

    @staticmethod
    def _format_error(resp: httpx.Response, prefix: str) -> str:
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
