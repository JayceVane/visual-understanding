"""Shared operation logic — called by both the MCP server and the CLI.

This module is the single source of truth for the three operations:
``do_analyze``, ``do_ground``, ``do_list_providers``. Both the FastMCP tools
(``server.py``) and argparse subcommands (``cli.py``) delegate here so there is
zero business-logic duplication between the two interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .grounding import (
    GroundingFormat,
    build_grounding_prompt,
    draw_boxes,
    load_image_for_viz,
    parse_coordinates,
)
from .media import resolve_file, resolve_image, resolve_video
from .providers import get_provider


# ---------------------------------------------------------------------------
# vision_analyze
# ---------------------------------------------------------------------------

async def do_analyze(
    config: AppConfig,
    images: list[str] | None = None,
    videos: list[str] | None = None,
    files: list[str] | None = None,
    prompt: str = "Describe this image in detail",
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Multimodal understanding: send images/videos/files + prompt, get text."""
    # Validate input groups (mutually exclusive per API limits)
    input_count = sum(1 for x in [images, videos, files] if x)
    if input_count == 0:
        return {"success": False, "error": "Must provide at least one of: images, videos, files"}
    if input_count > 1:
        return {"success": False, "error": "images, videos, and files are mutually exclusive in one request"}

    # Resolve provider
    try:
        name, prov = get_provider(config, provider)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    if not prov.is_configured:
        hint = (
            f"Set `api_key` for provider '{name}' in the config file, "
            f"or set the environment variable '{prov.config.api_key_env}'."
            if prov.config.api_key_env
            else f"Provider '{name}' needs an 'api_key' in the config file."
        )
        return {"success": False, "error": f"Provider '{name}' API key not set. {hint}"}

    # Build normalised content blocks
    content: list[dict[str, Any]] = []
    try:
        if images:
            if len(images) > prov.config.max_images:
                return {
                    "success": False,
                    "error": f"Too many images: {len(images)} (max {prov.config.max_images} for '{name}')",
                }
            for img in images:
                content.append({"type": "image", "url": resolve_image(img)})

        elif videos:
            if not prov.config.supports_video:
                return {"success": False, "error": f"Provider '{name}' does not support video input"}
            for vid in videos:
                content.append({"type": "video", "url": resolve_video(vid)})

        elif files:
            if not prov.config.supports_files:
                return {"success": False, "error": f"Provider '{name}' does not support file input"}
            for f in files:
                content.append({"type": "file", "url": resolve_file(f)})
    except (ValueError, FileNotFoundError) as exc:
        return {"success": False, "error": str(exc)}

    content.append({"type": "text", "text": prompt})

    result = await prov.chat(
        content, model=model, temperature=temperature, max_tokens=max_tokens
    )

    response: dict[str, Any] = {
        "success": result.success,
        "text": result.text,
        "usage": result.usage,
        "provider": name,
        "model": model or prov.config.default_chat_model,
    }
    if result.error:
        response["error"] = result.error
    if result.finish_reason == "sensitive":
        response["warning"] = "Content may have been blocked by safety review (finish_reason: sensitive)"
    return response


# ---------------------------------------------------------------------------
# vision_ground
# ---------------------------------------------------------------------------

@dataclass
class GroundOutput:
    """Result of do_ground — JSON-serialisable result + optional image bytes."""
    result: dict[str, Any]
    image_bytes: bytes | None = None


async def do_ground(
    config: AppConfig,
    image: str,
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    format: GroundingFormat = "bbox_2d",
    visualize: bool = False,
    box_color: str = "red",
    box_thickness: int = 3,
    distinct_colors: bool = False,
) -> GroundOutput:
    """Object grounding: locate targets in an image, optionally draw boxes."""
    # Resolve provider
    try:
        name, prov = get_provider(config, provider)
    except ValueError as exc:
        return GroundOutput(result={"success": False, "error": str(exc)})

    if not prov.is_configured:
        hint = (
            f"Set `api_key` for provider '{name}' in the config file, "
            f"or set the environment variable '{prov.config.api_key_env}'."
            if prov.config.api_key_env
            else f"Provider '{name}' needs an 'api_key' in the config file."
        )
        return GroundOutput(result={
            "success": False,
            "error": f"Provider '{name}' API key not set. {hint}",
        })

    native_grounding = prov.config.native_grounding

    # Build the grounding prompt + content
    grounding_prompt = build_grounding_prompt(prompt, format, native=native_grounding)
    try:
        resolved_image = resolve_image(image)
    except (ValueError, FileNotFoundError) as exc:
        return GroundOutput(result={"success": False, "error": str(exc)})

    content = [
        {"type": "image", "url": resolved_image},
        {"type": "text", "text": grounding_prompt},
    ]

    result = await prov.chat(content, model=model, temperature=0.1, max_tokens=2048)

    if not result.success:
        return GroundOutput(result={
            "success": False,
            "error": result.error,
            "provider": name,
            "model": model or prov.config.default_chat_model,
        })

    # Parse coordinates from the model response
    parsed = parse_coordinates(result.text, format)

    response: dict[str, Any] = {
        "success": True,
        "coordinates": parsed.coordinates,
        "raw_text": parsed.raw_text,
        "provider": name,
        "model": model or prov.config.default_chat_model,
        "native_grounding": native_grounding,
    }
    if parsed.labels:
        response["labels"] = parsed.labels

    # Optionally draw boxes
    image_bytes = None
    if visualize and parsed.coordinates:
        try:
            pil_img = await load_image_for_viz(resolved_image)
            image_bytes = draw_boxes(
                pil_img,
                parsed.coordinates,
                labels=parsed.labels,
                box_color=box_color,
                box_thickness=box_thickness,
                distinct_colors=distinct_colors,
            )
        except Exception as exc:
            response["visualization_error"] = f"Failed to draw boxes: {exc}"

    return GroundOutput(result=response, image_bytes=image_bytes)


# ---------------------------------------------------------------------------
# list_providers
# ---------------------------------------------------------------------------

def do_list_providers(config: AppConfig) -> dict[str, Any]:
    """Return a summary of all configured providers."""
    providers_info: dict[str, Any] = {}
    for name, prov_cfg in config.providers.items():
        providers_info[name] = {
            "type": prov_cfg.type,
            "models": prov_cfg.chat_models,
            "default_model": prov_cfg.default_chat_model,
            "capabilities": sorted(prov_cfg.capabilities),
            "configured": prov_cfg.is_configured,
            "key_source": prov_cfg.key_source,
            "api_key_env": prov_cfg.api_key_env,
        }

    return {
        "default_provider": config.default_provider,
        "providers": providers_info,
    }
