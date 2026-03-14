"""
Image chunking — converts a PNG or JPEG file into chunk dicts.

Implements Dual-Stream Embedding:
- Stream 1: one global chunk for the full image.
- Stream 2: one local chunk per region of interest detected by Gemini Flash.

Global chunk shape:
    type="image", image_bytes, text, source, page, chunk_index,
    modality="image_global", region_type="full", crop_bbox=None,
    region_count, parent_image_id

Local chunk shape:
    type="image", image_bytes, text, source, page, chunk_index,
    modality="image_local", region_type, crop_bbox, parent_image_id,
    width_px, height_px, crop_bbox_normalized
"""

import hashlib
import io
import json
import re
from pathlib import Path

from google import genai
from google.genai import types as genai_types
from loguru import logger
from PIL import Image

import backend.config as config

_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

_REGION_PROMPT = (
    "Identify all visually distinct regions in this image that contain "
    "meaningful content: tables, diagrams, charts, text blocks, figures. "
    "Return a JSON array only, no explanation. "
    "Format:\n"
    "[\n"
    "  {\n"
    "    \"region_type\": \"table\",\n"
    "    \"bbox\": [x1, y1, x2, y2]  // as fractions of image width/height (0.0-1.0)\n"
    "  }\n"
    "]"
)


def chunk_image(filepath: str) -> list[dict]:
    """
    Dual-stream chunking for a single image file.

    Stream 1: one global chunk for the full image.
    Stream 2: one local chunk per region of interest detected by Gemini Flash.
    Returns a combined list; does not call embed_chunks or add_chunks.
    """
    path = Path(filepath)
    filename = path.name

    # ── Validate format and read raw bytes ───────────────────────────────────
    image = Image.open(filepath)
    fmt = image.format  # set by Pillow after open
    if fmt not in {"PNG", "JPEG"}:
        raise ValueError(
            f"chunk_image: unsupported format '{fmt}' for {filename}. "
            "Only PNG and JPEG are supported."
        )

    raw_bytes = path.read_bytes()
    global_image_id = hashlib.sha256(raw_bytes).hexdigest()
    mime_type = "image/png" if fmt == "PNG" else "image/jpeg"
    img_width, img_height = image.size

    # ── Stream 1 — Global chunk (region_count backfilled after region loop) ───
    logger.info(f"chunk_image: global chunk for {filename} ({fmt}, {len(raw_bytes)} bytes)")
    global_chunk: dict = {
        "type": "image",
        "image_bytes": raw_bytes,
        "text": f"Image: {filename}",
        "source": filepath,
        "page": 0,
        "chunk_index": 0,
        "modality": "image_global",
        "region_type": "full",
        "crop_bbox": None,
        "region_count": 0,
        "parent_image_id": global_image_id,
    }
    chunks: list[dict] = [global_chunk]

    # ── Stream 2 — Local region chunks ────────────────────────────────────────
    try:
        image_part = genai_types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)
        response = _gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[image_part, _REGION_PROMPT],
        )
        response_text = response.text.strip()

        # Strip markdown code fences if present
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

        regions: list[dict] = json.loads(response_text)
        if not isinstance(regions, list) or len(regions) == 0:
            logger.warning(
                f"chunk_image: Gemini Flash returned no regions for {filename}, skipping Stream 2"
            )
            return chunks

    except Exception as exc:
        logger.warning(
            f"chunk_image: region detection failed for {filename}, skipping Stream 2: {exc}"
        )
        return chunks

    chunk_index = 1
    for region in regions:
        try:
            region_type: str = region["region_type"]
            bbox_frac: list[float] = region["bbox"]  # [x1, y1, x2, y2] as fractions

            x1_px = int(bbox_frac[0] * img_width)
            y1_px = int(bbox_frac[1] * img_height)
            x2_px = int(bbox_frac[2] * img_width)
            y2_px = int(bbox_frac[3] * img_height)

            crop = image.crop((x1_px, y1_px, x2_px, y2_px))
            crop_w, crop_h = crop.size
            buf = io.BytesIO()
            crop.save(buf, format=fmt)
            crop_bytes = buf.getvalue()

            logger.info(
                f"chunk_image: region {chunk_index} — {region_type} "
                f"bbox=({x1_px},{y1_px},{x2_px},{y2_px}) for {filename}"
            )
            chunks.append(
                {
                    "type": "image",
                    "image_bytes": crop_bytes,
                    "text": f"Image: {filename} | Region: {region_type}",
                    "source": filepath,
                    "page": 0,
                    "chunk_index": chunk_index,
                    "modality": "image_local",
                    "parent_image_id": global_image_id,
                    "region_type": region_type,
                    "crop_bbox": [x1_px, y1_px, x2_px, y2_px],
                    "width_px": crop_w,
                    "height_px": crop_h,
                    "crop_bbox_normalized": bbox_frac,
                }
            )
            chunk_index += 1

        except Exception as exc:
            logger.warning(
                f"chunk_image: skipping bad region in {filename}: {exc} — region={region}"
            )
            continue

    global_chunk["region_count"] = chunk_index - 1
    return chunks
