"""FastMCP server — exposes visual-understanding tools via the MCP protocol.

Start with ``visual-understanding serve`` (or ``python -m visual_understanding serve``).
The server reads stdio and exposes three tools:

  - ``vision_analyze``  — multimodal understanding (caption / OCR / Q&A)
  - ``vision_ground``   — object localisation + optional box visualisation
  - ``list_providers``  — introspection of configured providers
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from .config import AppConfig, load_config
from .ops import do_analyze, do_ground, do_list_providers

mcp = FastMCP("visual-understanding")

# Config is loaded lazily on first use (so import-time side-effects are zero).
_config: AppConfig | None = None


def _cfg() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


@mcp.tool()
async def vision_analyze(
    images: list[str] | None = None,
    videos: list[str] | None = None,
    files: list[str] | None = None,
    prompt: str = "Describe this image in detail",
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Analyze images, videos, or documents using a multimodal vision model.

    Supports captioning, OCR, visual Q&A, document understanding, and multi-image
    comparison. Provide one of ``images``, ``videos``, or ``files`` (mutually
    exclusive per API limits).

    Args:
        images: Image URLs, local paths, or ``base64:`` strings (supports multiple).
        videos: Video URLs only (mp4/mkv/mov). Provider must support video.
        files: Document URLs (pdf/docx/txt/xlsx/pptx). Provider must support files.
        prompt: Instruction for the model (e.g. "Describe this image", "Extract all text").
        provider: Provider name from config (default: config's default_provider).
        model: Model name (default: provider's default_chat_model).
        temperature: Sampling temperature 0-2 (default 0.8).
        max_tokens: Max output tokens (default 2048).

    Returns:
        Dict with ``success``, ``text``, ``usage``, ``provider``, ``model``,
        and optionally ``error`` / ``warning``.
    """
    return await do_analyze(
        _cfg(), images, videos, files, prompt,
        provider, model, temperature, max_tokens,
    )


@mcp.tool()
async def vision_ground(
    image: str,
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    format: str = "bbox_2d",
    visualize: bool = False,
    box_color: str = "red",
    box_thickness: int = 3,
) -> list[Any] | dict[str, Any]:
    """Locate objects in an image and optionally draw bounding boxes.

    Uses the vision model's grounding capability to find targets described by
    ``prompt``. Coordinates are normalised to 0-1000 (relative to image size).
    Providers with ``native_grounding`` (e.g. Zhipu GLM-V) produce the most
    accurate results.

    Args:
        image: Image URL, local path, or ``base64:`` string.
        prompt: What to locate (e.g. "all people wearing red hats").
        provider: Provider name (default: config's default_provider).
        model: Model name (default: provider's default_chat_model).
        format: Output format — ``bbox_2d`` (list of boxes) or ``detection_json``
            (list of {label, bbox_2d} objects).
        visualize: If True, also return a visualised image with boxes drawn.
        box_color: Box color name/hex for visualisation (default "red").
        box_thickness: Box line thickness in pixels (default 3).

    Returns:
        Dict with ``success``, ``coordinates``, ``raw_text``, ``provider``,
        ``model``. When ``visualize=True``, also returns an image content block.
    """
    output = await do_ground(
        _cfg(), image, prompt, provider, model,
        format, visualize, box_color, box_thickness,
    )

    if output.image_bytes:
        # Return both the JSON result and the visualised image
        return [output.result, Image(data=output.image_bytes, format="png")]
    return output.result


@mcp.tool()
async def list_providers() -> dict[str, Any]:
    """List all configured vision providers, their capabilities, and models.

    Returns:
        Dict with ``default_provider`` and a ``providers`` map containing
        each provider's type, models, capabilities, and configuration status.
    """
    return do_list_providers(_cfg())


def main() -> None:
    """Entry point for the ``serve`` subcommand — starts the MCP stdio server."""
    mcp.run()


if __name__ == "__main__":
    main()
