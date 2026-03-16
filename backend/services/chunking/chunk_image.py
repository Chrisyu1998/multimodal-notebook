"""
Image chunking — converts a PNG or JPEG file into chunk dicts.

Implements Dual-Stream Embedding:
- Stream 1: one global chunk for the full image.
- Stream 2: one local chunk per region of interest detected by Gemini Flash.

The global chunk carries a rich Gemini-generated caption (verbatim text
transcription + visual description) so it is meaningful for BM25 keyword
search.  The caption call and the region detection call run in parallel to
avoid adding wall-clock latency.

Global chunk shape:
    type="image", image_bytes, text, source, page, chunk_index,
    modality="image_global", region_type="full", crop_bbox=None,
    region_count, parent_image_id

Local chunk shape:
    type="image", image_bytes, text, source, page, chunk_index,
    modality="image_local", region_type, label, crop_bbox, parent_image_id,
    width_px, height_px, crop_bbox_normalized
"""

import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types as genai_types
from loguru import logger
from PIL import Image, ImageOps

import backend.config as config

_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

_REGION_PROMPT_TEMPLATE = """Analyze this image and identify visually distinct regions of meaningful content.

IMPORTANT: This image is exactly {width} pixels wide and {height} pixels tall.
All bbox coordinates must be integers within these bounds:
  x values: 0 to {width}
  y values: 0 to {height}
The bottom-right corner of the image is ({width}, {height}).
Do not return any y value greater than {height} or any x value greater than {width}.

Rules:
- Prefer SECTION-level chunks: a labeled group of components is one region, not each component separately
- Avoid tiny regions under ~3% of image area unless they are standalone text blocks
- Do NOT nest or overlap regions; each area should appear in at most one bbox
- Identify layout structure first (rows, columns, panels) then assign one bbox per logical section
- Label region_type precisely using: title, section_label, diagram_panel, table, chart, text_block, icon_group, pipeline_strip
- If you are uncertain about a region's exact boundary, shrink the bbox inward rather than guess

Return ONLY a JSON array, no explanation, no markdown fences.
Coordinates are pixel values [x1, y1, x2, y2] where (0,0) is top-left.

[
  {{
    "region_type": "diagram_panel",
    "label": "short human-readable description of content",
    "bbox": [x1, y1, x2, y2]
  }}
]"""

_CAPTION_PROMPT: str = (
    "Analyze this image and respond in exactly two parts:\n\n"
    "TRANSCRIPTION: Extract all visible text verbatim — titles, headings, labels, "
    "axis values, legend entries, bullet points, table cells, captions, annotations, "
    "and any other readable text in the image. Preserve technical terms, codes, and "
    "numbers exactly as written. If there is no text, write 'No text.'\n\n"
    "DESCRIPTION: Describe the visual content — the image type (diagram, chart, photo, "
    "screenshot, etc.), what it depicts, key entities, relationships, structure, and "
    "purpose. Be specific and factual.\n\n"
    "Do not add any other commentary or headings."
)


def _clamp_bbox(
    bbox: list[int], img_width: int, img_height: int, min_side: int = 10
) -> Optional[list[int]]:
    """Clamp bbox to image bounds; return None if the result is too small to be useful."""
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(img_width, int(bbox[2]))
    y2 = min(img_height, int(bbox[3]))
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return [x1, y1, x2, y2]


def _rescue_oob_bbox(
    bbox: list[int], img_width: int, img_height: int
) -> Optional[list[int]]:
    """
    Attempt proportional rescale for a bbox that is slightly out of bounds.

    Gemini sometimes returns coordinates in an internal space that is slightly
    larger than the actual image (e.g. the bottom row placed just below the
    real image bottom). If the overshoot is within 25% on either axis we
    rescale that axis back into bounds before clamping.  Beyond 25% the
    hallucination is too large to reliably recover and we return None.
    """
    x2, y2 = int(bbox[2]), int(bbox[3])
    y_scale = img_height / y2 if y2 > img_height else 1.0
    x_scale = img_width / x2 if x2 > img_width else 1.0

    if y_scale < 0.75 or x_scale < 0.75:
        return None

    rescaled = [
        round(int(bbox[0]) * x_scale),
        round(int(bbox[1]) * y_scale),
        round(x2 * x_scale),
        round(y2 * y_scale),
    ]
    return _clamp_bbox(rescaled, img_width, img_height)


def _generate_image_caption(image_bytes: bytes, mime_type: str) -> str:
    """Generate a rich text caption for an image via Gemini Flash.

    Output format: "TRANSCRIPTION: ... DESCRIPTION: ..."
    Captures both verbatim text (for BM25 exact-term matching) and semantic
    description (for paraphrase coverage).
    Returns empty string on failure — callers fall back to filename.
    """
    try:
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, _CAPTION_PROMPT],
        )
        text = "".join(
            part.text
            for part in response.candidates[0].content.parts
            if hasattr(part, "text") and part.text
        ).strip()
        return text
    except Exception as exc:
        logger.warning(f"chunk_image: caption generation failed: {exc}")
        return ""


def _detect_regions(
    image_bytes: bytes, mime_type: str, img_width: int, img_height: int
) -> list[dict]:
    """Call Gemini Flash to detect visually distinct regions in an image.

    Returns a list of region dicts with region_type, label, bbox.
    Returns empty list on failure — callers skip Stream 2 gracefully.
    """
    try:
        region_prompt = _REGION_PROMPT_TEMPLATE.format(width=img_width, height=img_height)
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = _gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image_part, region_prompt],
        )
        response_text = "".join(
            part.text for part in response.candidates[0].content.parts
            if hasattr(part, "text") and part.text
        ).strip()

        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

        regions: list[dict] = json.loads(response_text)
        if not isinstance(regions, list):
            logger.warning("chunk_image: region detection returned non-list JSON — skipping Stream 2")
            return []
        return regions
    except Exception as exc:
        logger.warning(f"chunk_image: region detection failed: {exc}")
        return []


def chunk_image(filepath: str) -> list[dict]:
    """
    Dual-stream chunking for a single image file.

    Stream 1: one global chunk for the full image, with a Gemini-generated
    caption as the text field for rich BM25 coverage.
    Stream 2: one local chunk per region of interest detected by Gemini Flash.

    The caption call (gemini-2.5-flash) and region detection call
    (gemini-3-flash-preview) run in parallel to avoid serial latency.
    Returns a combined list; does not call embed_chunks or add_chunks.
    """
    path = Path(filepath)
    filename = path.name

    # ── Load image, apply EXIF orientation, normalise to PNG/JPEG if needed ──
    image = ImageOps.exif_transpose(Image.open(filepath))
    fmt = Image.open(filepath).format  # exif_transpose drops .format; re-read it
    if fmt not in {"PNG", "JPEG"}:
        logger.info(f"chunk_image: converting {fmt} → PNG for {filename}")
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        fmt = "PNG"
    else:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        raw_bytes = buf.getvalue()

    global_image_id = hashlib.sha256(raw_bytes).hexdigest()
    mime_type = "image/png" if fmt == "PNG" else "image/jpeg"
    img_width, img_height = image.size

    # ── Run caption + region detection in parallel ────────────────────────────
    logger.info(
        f"chunk_image: starting caption + region detection for {filename} "
        f"({fmt}, {len(raw_bytes)} bytes)"
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        caption_future = executor.submit(_generate_image_caption, raw_bytes, mime_type)
        region_future = executor.submit(
            _detect_regions, raw_bytes, mime_type, img_width, img_height
        )
        caption = caption_future.result()
        regions = region_future.result()

    if caption:
        logger.info(f"chunk_image: caption generated ({len(caption)} chars) for {filename}")
    else:
        logger.warning(f"chunk_image: caption empty — falling back to filename for {filename}")

    # ── Stream 1 — Global chunk ───────────────────────────────────────────────
    global_chunk: dict = {
        "type": "image",
        "image_bytes": raw_bytes,
        "text": caption or f"Image: {filename}",
        "source": filepath,
        "page": 0,
        "chunk_index": 0,
        "modality": "image_global",
        "region_type": "full",
        "crop_bbox": None,
        "region_count": 0,  # backfilled after region loop
        "parent_image_id": global_image_id,
    }
    chunks: list[dict] = [global_chunk]

    # ── Stream 2 — Local region chunks ────────────────────────────────────────
    if not regions:
        logger.warning(
            f"chunk_image: no regions detected for {filename} — returning global chunk only"
        )
        return chunks

    # Deduplicate by exact bbox to prevent identical region chunks
    seen_bboxes: set[tuple] = set()
    deduped: list[dict] = []
    for r in regions:
        key = tuple(r.get("bbox", []))
        if key not in seen_bboxes:
            seen_bboxes.add(key)
            deduped.append(r)
    if len(deduped) < len(regions):
        logger.info(
            f"chunk_image: removed {len(regions) - len(deduped)} duplicate region(s) for {filename}"
        )
    regions = deduped

    chunk_index = 1
    for region in regions:
        region_index = chunk_index
        chunk_index += 1
        try:
            region_type: str = region["region_type"]
            label: str = region.get("label", region_type)
            raw_bbox: list[int] = region["bbox"]

            clamped = _clamp_bbox(raw_bbox, img_width, img_height)
            if clamped is None:
                clamped = _rescue_oob_bbox(raw_bbox, img_width, img_height)
                if clamped is None:
                    logger.warning(
                        f"chunk_image: region {region_index} '{label}' is out-of-bounds "
                        f"and unrescuable, skipping — raw bbox={raw_bbox} for {filename}"
                    )
                    continue
                logger.info(
                    f"chunk_image: region {region_index} '{label}' rescued via rescale "
                    f"raw={raw_bbox} → clamped={clamped} for {filename}"
                )
            x1_px, y1_px, x2_px, y2_px = clamped

            crop = image.crop((x1_px, y1_px, x2_px, y2_px))
            crop_w, crop_h = crop.size
            buf = io.BytesIO()
            crop.save(buf, format=fmt)
            crop_bytes = buf.getvalue()

            logger.info(
                f"chunk_image: region {region_index} — {region_type} '{label}' "
                f"bbox=({x1_px},{y1_px},{x2_px},{y2_px}) for {filename}"
            )
            chunks.append(
                {
                    "type": "image",
                    "image_bytes": crop_bytes,
                    "text": f"Image: {filename} | Region: {region_type} | {label}",
                    "source": filepath,
                    "page": 0,
                    "chunk_index": region_index,
                    "modality": "image_local",
                    "parent_image_id": global_image_id,
                    "region_type": region_type,
                    "label": label,
                    "crop_bbox": [x1_px, y1_px, x2_px, y2_px],
                    "width_px": crop_w,
                    "height_px": crop_h,
                    "crop_bbox_normalized": [
                        x1_px / img_width,
                        y1_px / img_height,
                        x2_px / img_width,
                        y2_px / img_height,
                    ],
                }
            )

        except Exception as exc:
            logger.warning(
                f"chunk_image: skipping bad region {region_index} in {filename}: {exc} — region={region}"
            )
            continue

    global_chunk["region_count"] = len(chunks) - 1
    return chunks
