"""Grounding utilities — prompt construction, coordinate parsing, box visualisation.

Grounding coordinates from models like GLM-V are **relative values normalised to
0-1000** based on image width/height::

    x = round(x_pixel / W * 1000)
    y = round(y_pixel / H * 1000)

The origin (0, 0) is the top-left corner. These utilities convert between
normalised and pixel coordinates and draw bounding boxes on images.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from PIL import Image, ImageDraw, ImageFont

GroundingFormat = Literal["bbox_2d", "detection_json"]
COORD_NORM = 1000  # models output coordinates normalised to this range

# Default colors for multi-target visualisation
_DEFAULT_COLORS = [
    "red", "green", "blue", "yellow", "cyan", "magenta",
    "orange", "purple", "lime", "pink",
]


@dataclass
class GroundingResult:
    """Parsed grounding output."""

    coordinates: list[list[int]] = field(default_factory=list)
    labels: list[str] | None = None
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_grounding_prompt(
    target: str,
    fmt: GroundingFormat = "bbox_2d",
    native: bool = True,
) -> str:
    """Build the instruction prompt sent to the vision model.

    Args:
        target: What to locate (e.g. "all people wearing red hats").
        fmt: Desired output format — ``bbox_2d`` (list of boxes) or
            ``detection_json`` (list of ``{label, bbox_2d}`` objects).
        native: Whether the provider has native grounding support. When False,
            the prompt adds extra guidance to coax non-grounding models.
    """
    norm_clause = (
        "Coordinates MUST be relative values normalised to 0-1000 based on "
        "image width and height: x = round(x_pixel / W * 1000), "
        "y = round(y_pixel / H * 1000). The origin (0, 0) is the top-left corner."
    )

    if fmt == "detection_json":
        format_clause = (
            f'Output ONLY a JSON array: [{{"label": "category", '
            f'"bbox_2d": [x1, y1, x2, y2]}}, ...]. '
            f"Each bbox_2d is [top-left-x, top-left-y, bottom-right-x, bottom-right-y]."
        )
    else:  # bbox_2d
        format_clause = (
            "Output ONLY a JSON array of bounding boxes: "
            "[[x1, y1, x2, y2], ...]. "
            "Each box is [top-left-x, top-left-y, bottom-right-x, bottom-right-y]."
        )

    native_clause = "" if native else (
        "\n\nEven if you are not specifically trained for grounding, do your best "
        "to estimate the target locations and output the coordinates as instructed."
    )

    return (
        f"Locate and box the following target(s) in this image: {target}\n\n"
        f"{format_clause}\n{norm_clause}{native_clause}\n\n"
        "Respond with ONLY the JSON array, no other text."
    )


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

def parse_coordinates(
    text: str, fmt: GroundingFormat = "bbox_2d"
) -> GroundingResult:
    """Extract grounding coordinates from a model's text response.

    Tries multiple strategies:
      1. Direct JSON parse of the entire response.
      2. Regex extraction of JSON arrays from surrounding text.
      3. Fallback: regex extraction of bare coordinate lists.
    """
    text = text.strip()

    # Strategy 1: try direct JSON parse
    try:
        data = json.loads(text)
        result = _extract_from_parsed(data, fmt)
        if result.coordinates:
            result.raw_text = text
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: regex — find JSON arrays in the text
    result = _regex_extract(text, fmt)
    if result.coordinates:
        result.raw_text = text
        return result

    # Fallback: no coordinates found
    return GroundingResult(raw_text=text)


def _extract_from_parsed(
    data: Any, fmt: GroundingFormat
) -> GroundingResult:
    """Extract coordinates from already-parsed JSON data."""
    if not isinstance(data, list):
        return GroundingResult()

    # detection_json: [{"label": "...", "bbox_2d": [...]}, ...]
    if fmt == "detection_json" or all(isinstance(item, dict) for item in data):
        coords: list[list[int]] = []
        labels: list[str] = []
        for item in data:
            if isinstance(item, dict):
                box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
                if isinstance(box, list) and len(box) == 4:
                    coords.append([int(v) for v in box])
                    labels.append(str(item.get("label", item.get("category", "object"))))
        if coords:
            return GroundingResult(coordinates=coords, labels=labels)

    # bbox_2d: [[x1, y1, x2, y2], ...]
    if all(isinstance(item, (list, tuple)) and len(item) == 4 for item in data):
        coords = [[int(v) for v in item] for item in data]
        return GroundingResult(coordinates=coords)

    return GroundingResult()


def _regex_extract(text: str, fmt: GroundingFormat) -> GroundingResult:
    """Regex-based extraction of coordinate arrays from free-form text."""
    # Look for detection JSON objects: {"label": ..., "bbox_2d": [...]}
    det_pattern = r'\{\s*"label"\s*:\s*"[^"]*"\s*,\s*"bbox_2d"\s*:\s*\[([0-9,\s]+)\]'
    det_matches = re.findall(det_pattern, text, re.IGNORECASE)
    if det_matches:
        coords = []
        labels = []
        # Re-extract labels alongside
        label_pattern = r'\{\s*"label"\s*:\s*"([^"]*)"\s*,\s*"bbox_2d"'
        label_matches = re.findall(label_pattern, text, re.IGNORECASE)
        for i, nums in enumerate(det_matches):
            vals = [int(x.strip()) for x in nums.split(",") if x.strip().isdigit()]
            if len(vals) == 4:
                coords.append(vals)
                if i < len(label_matches):
                    labels.append(label_matches[i])
        if coords:
            return GroundingResult(coordinates=coords, labels=labels)

    # Look for bare bbox arrays: [[x1, y1, x2, y2], ...]
    # Match the outermost [...] that contains sub-arrays of 4 numbers
    box_pattern = r"\[(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\]"
    matches = re.findall(box_pattern, text)
    if matches:
        coords = [[int(v) for v in m] for m in matches]
        # Filter: valid normalised coordinates should be 0-1000
        valid = [c for c in coords if all(0 <= v <= COORD_NORM for v in c)]
        if valid:
            return GroundingResult(coordinates=valid)

    return GroundingResult()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

async def load_image_for_viz(source: str) -> Image.Image:
    """Load a PIL image from a URL, local path, or data URL."""
    from .media import is_url

    if source.startswith("data:"):
        # Parse data URL
        _, _, b64data = source.partition(",")
        import base64
        return Image.open(io.BytesIO(base64.b64decode(b64data)))

    if is_url(source):
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(source)
            resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    # Local path
    return Image.open(source)


def denormalize_box(
    box: list[int], width: int, height: int, norm: int = COORD_NORM
) -> list[int]:
    """Convert a normalised [0-norm] box to pixel coordinates."""
    x1 = round(box[0] / norm * width)
    y1 = round(box[1] / norm * height)
    x2 = round(box[2] / norm * width)
    y2 = round(box[3] / norm * height)
    return [x1, y1, x2, y2]


def draw_boxes(
    image: Image.Image,
    boxes: list[list[int]],
    labels: list[str] | None = None,
    box_color: str = "red",
    box_thickness: int = 3,
    distinct_colors: bool = False,
) -> bytes:
    """Draw bounding boxes on a PIL image and return PNG bytes.

    Args:
        image: PIL Image to draw on (not modified in place).
        boxes: Normalised [0-1000] bounding boxes.
        labels: Optional label strings (one per box).
        box_color: Color name or hex for the boxes.
        box_thickness: Line width in pixels.
        distinct_colors: Use a different color per box (cycles through palette).

    Returns:
        PNG image bytes.
    """
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        px_box = denormalize_box(box, w, h)
        color = (
            _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            if distinct_colors
            else box_color
        )
        draw.rectangle(px_box, outline=color, width=box_thickness)

        if labels and i < len(labels):
            label = labels[i]
            # Draw label text with a small background for readability
            text_x = px_box[0]
            text_y = max(0, px_box[1] - 16)
            if font:
                bbox = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle(
                    [bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1],
                    fill=color,
                )
                draw.text((text_x, text_y), label, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
