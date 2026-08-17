"""CLI entry point — ``visual-understanding <subcommand>``.

Subcommands:
  analyze         Multimodal understanding (images/videos/files → text)
  ground          Object localisation + optional box visualisation
  list-providers  Show configured providers and their capabilities
  serve           Start the MCP stdio server

All subcommands except ``serve`` print JSON to stdout and are designed for
consumption by agents (Skill mode) or shell scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from .config import load_config


def _print_json(data: object, output: str | None = None) -> None:
    """Print *data* as JSON to stdout, optionally also save to *output* file."""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    print(text)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"\nResult saved to: {output}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def _cmd_analyze(args: argparse.Namespace) -> int:
    config = load_config()
    from .ops import do_analyze

    result = asyncio.run(do_analyze(
        config,
        images=args.images,
        videos=args.videos,
        files=args.files,
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))
    _print_json(result, args.output)
    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# Subcommand: ground
# ---------------------------------------------------------------------------

def _cmd_ground(args: argparse.Namespace) -> int:
    config = load_config()
    from .ops import do_ground

    output = asyncio.run(do_ground(
        config,
        image=args.image,
        prompt=args.prompt,
        provider=args.provider,
        model=args.model,
        format=args.format,
        visualize=args.visualize,
        box_color=args.box_color,
        box_thickness=args.box_thickness,
    ))

    result = dict(output.result)

    # Handle visualised image
    if output.image_bytes:
        save_path = args.save_path
        if not save_path:
            # Save to a temp file if no explicit path given
            tmp = Path(tempfile.gettempdir()) / "visual_understanding_ground.png"
            save_path = str(tmp)
        Path(save_path).write_bytes(output.image_bytes)
        result["visualization_saved_path"] = str(Path(save_path).resolve())

    _print_json(result, args.output)
    return 0 if result.get("success") else 1


# ---------------------------------------------------------------------------
# Subcommand: list-providers
# ---------------------------------------------------------------------------

def _cmd_list_providers(args: argparse.Namespace) -> int:
    config = load_config()
    from .ops import do_list_providers

    result = do_list_providers(config)
    _print_json(result, args.output)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: doctor
# ---------------------------------------------------------------------------

def _cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose configuration — locate config file, check CLI, test endpoints."""
    import os
    from . import config as config_mod
    from .config import load_config

    print(f"▶ config file: {config_mod._find_config_path() or '(none — using built-in defaults)'}")
    print(f"▶ config env var ({config_mod._ENV_VAR}): {os.environ.get(config_mod._ENV_VAR) or '(not set)'}")
    print()

    cfg = load_config()
    print(f"▶ default provider: {cfg.default_provider}")
    print()

    from .media import is_public_url
    import httpx

    for name, p in cfg.providers.items():
        ok = p.is_configured
        print(f"[{'✅' if ok else '❌'}] {name}")
        print(f"    type: {p.type} | models: {len(p.chat_models)} | default: {p.default_chat_model}")
        print(f"    key source: {p.key_source}")
        print(f"    base_url: {p.base_url}")
        # Basic connectivity check (config-level only, no API call)
        if p.base_url and is_public_url(p.base_url):
            try:
                resp = httpx.get(p.base_url, timeout=5, follow_redirects=True)
                status = resp.status_code
                note = " — reachable" if status in (200, 301, 302, 307, 308) else (
                    " — reachable (root path responds, API path may differ)" if status in (400, 401, 403, 404, 405) else ""
                )
                print(f"    endpoint: HTTP {status}{note}")
            except Exception as exc:
                print(f"    endpoint: unreachable ({exc})")
        print()

    # Exit nonzero if any provider is unconfigured
    missing = [n for n, p in cfg.providers.items() if not p.is_configured]
    if missing:
        print(f"⚠️ {len(missing)} provider(s) missing API key: {', '.join(missing)}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: serve
# ---------------------------------------------------------------------------

def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import main as serve_main
    serve_main()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-understanding",
        description="Multi-provider visual understanding tool (MCP + CLI).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Multimodal understanding (caption/OCR/Q&A)")
    grp = p_analyze.add_argument_group("Input (pick one)")
    grp.add_argument("--images", "-i", nargs="+", help="Image URLs, local paths, or base64: strings")
    grp.add_argument("--videos", "-v", nargs="+", help="Video URLs (mp4/mkv/mov)")
    grp.add_argument("--files", "-f", nargs="+", help="Document URLs (pdf/docx/txt/xlsx/pptx)")
    p_analyze.add_argument("--prompt", "-p", default="Describe this image in detail", help="Instruction for the model")
    p_analyze.add_argument("--provider", default=None, help="Provider name (default: config default)")
    p_analyze.add_argument("--model", "-m", default=None, help="Model name (default: provider default)")
    p_analyze.add_argument("--temperature", "-t", type=float, default=0.8, help="Sampling temperature")
    p_analyze.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    p_analyze.add_argument("--output", "-o", default=None, help="Save JSON result to file")
    p_analyze.set_defaults(func=_cmd_analyze)

    # --- ground ---
    p_ground = sub.add_parser("ground", help="Object localisation + optional box visualisation")
    p_ground.add_argument("--image", "-i", required=True, help="Image URL, local path, or base64: string")
    p_ground.add_argument("--prompt", "-p", required=True, help="What to locate (e.g. 'all people')")
    p_ground.add_argument("--provider", default=None, help="Provider name")
    p_ground.add_argument("--model", "-m", default=None, help="Model name")
    p_ground.add_argument("--format", default="bbox_2d", choices=["bbox_2d", "detection_json"],
                          help="Coordinate output format (default: bbox_2d)")
    p_ground.add_argument("--visualize", action="store_true", help="Draw bounding boxes on the image")
    p_ground.add_argument("--box-color", default="red", help="Box color (default: red)")
    p_ground.add_argument("--box-thickness", type=int, default=3, help="Box line thickness (default: 3)")
    p_ground.add_argument("--save-path", default=None, help="Save visualised image to this path")
    p_ground.add_argument("--output", "-o", default=None, help="Save JSON result to file")
    p_ground.set_defaults(func=_cmd_ground)

    # --- list-providers ---
    p_list = sub.add_parser("list-providers", help="List configured providers and capabilities")
    p_list.add_argument("--output", "-o", default=None, help="Save JSON result to file")
    p_list.set_defaults(func=_cmd_list_providers)

    # --- doctor ---
    p_doctor = sub.add_parser("doctor", help="Diagnose config, API keys, and endpoint connectivity")
    p_doctor.set_defaults(func=_cmd_doctor)

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start the MCP stdio server")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main() -> None:
    """CLI entry point (registered as ``visual-understanding`` in pyproject.toml)."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
