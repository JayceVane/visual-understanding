"""Media input resolution — normalise image/video/file inputs for API calls.

Supports:
  - http(s) URLs  → passed through
  - local paths   → images encoded as ``data:`` base64 URLs; videos/files rejected
  - ``data:`` URLs → passed through
  - ``base64:`` prefix → wrapped into a ``data:image/jpeg;base64,...`` URL
"""

from __future__ import annotations

import base64
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

# Supported image extensions (per common VLM API constraints)
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
MAX_IMAGE_SIZE_MB = 10


def is_url(s: str) -> bool:
    """True if *s* looks like an http(s) URL."""
    return s.strip().startswith(("http://", "https://"))


def is_public_url(s: str) -> bool:
    """Validate that *s* is a public http(s) URL (blocks localhost / private IPs).

    Prevents SSRF: model ``file_url`` / ``video_url`` fetches should not hit
    internal network targets.
    """
    s = s.strip()
    if not is_url(s):
        return False
    parsed = urlparse(s)
    hostname = parsed.hostname
    if not hostname:
        return False
    # Block obvious local hostnames
    if hostname in ("localhost", "0.0.0.0", "::1"):
        return False
    # Block private / loopback / link-local IP literals
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    except ValueError:
        pass  # hostname is a domain name, not an IP — allow
    return True


def _validate_local_image(path: Path) -> str:
    """Return error message if the local image is invalid, empty string if OK."""
    if not path.exists():
        return f"Image not found: {path}"
    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTS:
        return (
            f"Unsupported image format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return (
            f"Image too large: {size_mb:.1f}MB (max {MAX_IMAGE_SIZE_MB}MB). "
            "Consider resizing or compressing."
        )
    return ""


def _load_image_as_data_url(path: Path) -> str:
    """Read a local image file and return a ``data:`` base64 URL."""
    err = _validate_local_image(path)
    if err:
        raise ValueError(err)

    mime = IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def resolve_image(image_input: str) -> str:
    """Normalise an image input to a URL or ``data:`` URL.

    Handles:
      - ``http(s)://...``          → passthrough (validated as public URL)
      - ``data:image/...;base64,..``→ passthrough
      - ``base64:<raw>``           → wrapped into ``data:image/jpeg;base64,...``
      - local file path            → read + base64-encode as ``data:`` URL
    """
    s = image_input.strip()

    if s.startswith("data:"):
        return s

    if s.startswith("base64:"):
        return f"data:image/jpeg;base64,{s[7:]}"

    if is_url(s):
        if not is_public_url(s):
            raise ValueError(
                f"Image URL must be a public http(s) address (blocked: {s})"
            )
        return s

    return _load_image_as_data_url(Path(s))


def resolve_video(video_input: str, provider_supports_video: bool = True) -> str:
    """Normalise a video input. Most VLM APIs only accept video URLs.

    Returns the URL if valid. Raises ValueError for local paths or private URLs.
    """
    s = video_input.strip()
    if not is_url(s):
        raise ValueError(
            f"Video inputs must be public URLs (local paths / base64 not supported): {s}"
        )
    if not is_public_url(s):
        raise ValueError(f"Video URL must be a public http(s) address (blocked: {s})")
    return s


def resolve_file(file_input: str, provider_supports_files: bool = True) -> str:
    """Normalise a document file input. APIs typically require URLs for files.

    Returns the URL if valid. Raises ValueError for local paths or private URLs.
    """
    s = file_input.strip()
    if not is_url(s):
        raise ValueError(
            f"File inputs must be public URLs (local paths not supported): {s}"
        )
    if not is_public_url(s):
        raise ValueError(f"File URL must be a public http(s) address (blocked: {s})")
    return s
