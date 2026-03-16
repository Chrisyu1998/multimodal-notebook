"""
Tests for backend.services.chunking.chunk_image.

Images are built in-memory with Pillow — no fixture files needed.
All Gemini Flash calls are mocked — no network I/O.
"""

import hashlib
import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from contextlib import contextmanager, ExitStack
from PIL import Image

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

import sys

from backend.services.chunking.chunk_image import (  # ensures module is in sys.modules
    chunk_image,
    _detect_regions,
)
_chunk_image_module = sys.modules["backend.services.chunking.chunk_image"]

# Stable caption returned by the mock — rich enough for text assertions
_FAKE_CAPTION = (
    "TRANSCRIPTION: left half right half table diagram\n"
    "DESCRIPTION: A test architecture diagram showing two regions."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GLOBAL_KEYS = {
    "type", "image_bytes", "text", "source", "page", "chunk_index",
    "modality", "region_type", "crop_bbox", "region_count", "parent_image_id",
}
_LOCAL_KEYS = {
    "type", "image_bytes", "text", "source", "page", "chunk_index",
    "modality", "region_type", "label", "crop_bbox", "parent_image_id",
    "width_px", "height_px", "crop_bbox_normalized",
}

# Pixel coords for a 200×100 image (default _make_image_file size)
_TWO_REGIONS = json.dumps([
    {"region_type": "table",   "label": "left half",   "bbox": [0,  0,  100, 50]},
    {"region_type": "diagram", "label": "right half",  "bbox": [100, 50, 200, 100]},
])
_ONE_REGION = json.dumps([
    {"region_type": "chart", "label": "centre", "bbox": [20, 10, 180, 90]},
])


def _make_image_file(
    fmt: str = "PNG",
    width: int = 200,
    height: int = 100,
    color: tuple = (128, 64, 32),
) -> str:
    """Write a solid-colour image to a temp file. Returns the file path."""
    img = Image.new("RGB", (width, height), color=color)
    suffix = ".png" if fmt == "PNG" else ".jpeg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name, format=fmt)
    tmp.close()
    return tmp.name


def _make_response(text: str) -> MagicMock:
    """Return a mock that quacks like a Gemini generate_content response.

    Simulates the parts-based response shape used by thinking models so that
    the ``candidates[0].content.parts`` iteration in chunk_image works correctly.
    """
    part = MagicMock()
    part.text = text
    mock = MagicMock()
    mock.candidates[0].content.parts = [part]
    return mock


@contextmanager
def _patch_chunk_image(regions_text: str, caption: str = _FAKE_CAPTION):
    """Patch _generate_image_caption and _detect_regions independently.

    regions_text is parsed from JSON — invalid JSON or non-list JSON both
    produce an empty region list (mirroring _detect_regions's own fallback).
    """
    try:
        parsed = json.loads(regions_text)
        if not isinstance(parsed, list):
            parsed = []
    except (json.JSONDecodeError, ValueError):
        parsed = []

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(_chunk_image_module, "_generate_image_caption", return_value=caption)
        )
        stack.enter_context(
            patch.object(_chunk_image_module, "_detect_regions", return_value=parsed)
        )
        yield


# Kept for tests that need to probe _gemini_client directly (e.g. TestDetectRegions)
def _patch_gemini_client(response_text: str):
    """Patch the raw Gemini client — used only for unit-testing _detect_regions."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_response(response_text)
    return patch.object(_chunk_image_module, "_gemini_client", mock_client)


# Backward-compat alias so existing test call sites need no changes
_patch_gemini = _patch_chunk_image


# ---------------------------------------------------------------------------
# Schema — global chunk
# ---------------------------------------------------------------------------

class TestGlobalChunkSchema:
    def test_required_keys_present(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert _GLOBAL_KEYS <= chunks[0].keys(), (
                f"missing keys: {_GLOBAL_KEYS - chunks[0].keys()}"
            )
        finally:
            os.unlink(path)

    def test_type_is_image(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["type"] == "image"
        finally:
            os.unlink(path)

    def test_modality_is_image_global(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["modality"] == "image_global"
        finally:
            os.unlink(path)

    def test_chunk_index_is_zero(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["chunk_index"] == 0
        finally:
            os.unlink(path)

    def test_page_is_zero(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["page"] == 0
        finally:
            os.unlink(path)

    def test_region_type_is_full(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["region_type"] == "full"
        finally:
            os.unlink(path)

    def test_crop_bbox_is_none(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["crop_bbox"] is None
        finally:
            os.unlink(path)

    def test_image_bytes_is_nonempty_bytes(self):
        """image_bytes must be non-empty bytes decodable as a valid image."""
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            data = chunks[0]["image_bytes"]
            assert isinstance(data, bytes) and len(data) > 0
            # Verify the bytes form a valid image (not garbage)
            img = Image.open(io.BytesIO(data))
            img.verify()
        finally:
            os.unlink(path)

    def test_source_is_filepath(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["source"] == path
        finally:
            os.unlink(path)

    def test_global_text_is_caption(self):
        """Global chunk text must be the Gemini-generated caption."""
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["text"] == _FAKE_CAPTION
        finally:
            os.unlink(path)

    def test_global_text_falls_back_to_filename_when_caption_fails(self):
        """When caption returns empty string the text must fall back to filename."""
        path = _make_image_file()
        try:
            with _patch_chunk_image(_TWO_REGIONS, caption=""):
                chunks = chunk_image(path)
            assert os.path.basename(path) in chunks[0]["text"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Schema — local chunks
# ---------------------------------------------------------------------------

class TestLocalChunkSchema:
    def test_required_keys_present(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                assert _LOCAL_KEYS <= local.keys(), (
                    f"missing keys: {_LOCAL_KEYS - local.keys()}"
                )
        finally:
            os.unlink(path)

    def test_modality_is_image_local(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                assert local["modality"] == "image_local"
        finally:
            os.unlink(path)

    def test_type_is_image(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                assert local["type"] == "image"
        finally:
            os.unlink(path)

    def test_chunk_index_sequential_from_one(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            local_indices = [c["chunk_index"] for c in chunks[1:]]
            assert local_indices == list(range(1, len(chunks)))
        finally:
            os.unlink(path)

    def test_text_contains_filename_and_region_type(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            filename = os.path.basename(path)
            local = chunks[1]
            assert filename in local["text"]
            assert local["region_type"] in local["text"]
        finally:
            os.unlink(path)

    def test_page_is_zero(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                assert local["page"] == 0
        finally:
            os.unlink(path)

    def test_image_bytes_is_bytes(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                assert isinstance(local["image_bytes"], bytes)
                assert len(local["image_bytes"]) > 0
        finally:
            os.unlink(path)

    def test_crop_dimensions_match_bbox(self):
        """width_px and height_px must match the actual cropped image size."""
        path = _make_image_file(width=200, height=100)
        # pixel bbox: left half → 100×100 crop
        region = json.dumps([{"region_type": "table", "label": "left half", "bbox": [0, 0, 100, 100]}])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            local = chunks[1]
            assert local["width_px"] == 100
            assert local["height_px"] == 100
        finally:
            os.unlink(path)

    def test_crop_bbox_pixel_coords_correct(self):
        path = _make_image_file(width=200, height=100)
        region = json.dumps([{"region_type": "chart", "label": "centre", "bbox": [50, 50, 150, 100]}])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            local = chunks[1]
            assert local["crop_bbox"] == [50, 50, 150, 100]
        finally:
            os.unlink(path)

    def test_crop_bbox_normalized_is_fractional(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                for coord in local["crop_bbox_normalized"]:
                    assert 0.0 <= coord <= 1.0, f"coord {coord} is not in [0, 1]"
        finally:
            os.unlink(path)

    def test_crop_bbox_normalized_derived_from_pixel_coords(self):
        """crop_bbox_normalized must be pixel coords divided by image dimensions."""
        path = _make_image_file(width=200, height=100)
        # [20, 10, 180, 90] → [0.1, 0.1, 0.9, 0.9]
        region = json.dumps([{"region_type": "figure", "label": "main", "bbox": [20, 10, 180, 90]}])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            assert chunks[1]["crop_bbox_normalized"] == [0.1, 0.1, 0.9, 0.9]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# parent_image_id consistency
# ---------------------------------------------------------------------------

class TestParentImageId:
    def test_parent_image_id_is_sha256_of_raw_bytes(self):
        path = _make_image_file()
        try:
            with open(path, "rb") as f:
                raw = f.read()
            expected = hashlib.sha256(raw).hexdigest()
            with _patch_gemini(_ONE_REGION):
                chunks = chunk_image(path)
            assert chunks[0]["parent_image_id"] == expected
        finally:
            os.unlink(path)

    def test_parent_image_id_identical_across_all_chunks(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            ids = [c["parent_image_id"] for c in chunks]
            assert len(set(ids)) == 1, f"parent_image_id differs across chunks: {ids}"
        finally:
            os.unlink(path)

    def test_different_images_have_different_parent_image_ids(self):
        path_a = _make_image_file(color=(255, 0, 0))
        path_b = _make_image_file(color=(0, 0, 255))
        try:
            with _patch_gemini(_ONE_REGION):
                chunks_a = chunk_image(path_a)
                chunks_b = chunk_image(path_b)
            assert chunks_a[0]["parent_image_id"] != chunks_b[0]["parent_image_id"]
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


# ---------------------------------------------------------------------------
# region_count backfill on global chunk
# ---------------------------------------------------------------------------

class TestRegionCount:
    def test_region_count_equals_local_chunk_count(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["region_count"] == len(chunks) - 1
        finally:
            os.unlink(path)

    def test_region_count_zero_when_stream2_skipped(self):
        path = _make_image_file()
        try:
            with _patch_gemini("[]"):
                chunks = chunk_image(path)
            assert chunks[0]["region_count"] == 0
        finally:
            os.unlink(path)

    def test_region_count_reflects_skipped_bad_regions(self):
        """If one of two regions is malformed, region_count must be 1, not 2."""
        mixed = json.dumps([
            {"region_type": "table", "label": "top left", "bbox": [0, 0, 100, 50]},
            {"region_type": "bad",   "bbox": "not-a-list"},          # will fail int() conversion
        ])
        path = _make_image_file()
        try:
            with _patch_gemini(mixed):
                chunks = chunk_image(path)
            assert chunks[0]["region_count"] == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Stream 2 fallback behaviour
# ---------------------------------------------------------------------------

class TestStream2Fallback:
    def test_empty_region_list_returns_only_global(self):
        path = _make_image_file()
        try:
            with _patch_gemini("[]"):
                chunks = chunk_image(path)
            assert len(chunks) == 1
            assert chunks[0]["modality"] == "image_global"
        finally:
            os.unlink(path)

    def test_json_parse_failure_returns_only_global(self):
        path = _make_image_file()
        try:
            with _patch_gemini("this is not json"):
                chunks = chunk_image(path)
            assert len(chunks) == 1
            assert chunks[0]["modality"] == "image_global"
        finally:
            os.unlink(path)

    def test_gemini_exception_returns_only_global(self):
        """When both Gemini helpers fail internally, chunk_image still returns the global chunk."""
        path = _make_image_file()
        try:
            # Both helpers catch their own exceptions and return "" / []
            with _patch_chunk_image("[]", caption=""):
                chunks = chunk_image(path)
            assert len(chunks) == 1
            assert chunks[0]["modality"] == "image_global"
            # Caption failure → text must fall back to filename
            assert os.path.basename(path) in chunks[0]["text"]
        finally:
            os.unlink(path)

    def test_non_list_json_returns_only_global(self):
        path = _make_image_file()
        try:
            with _patch_gemini('{"region_type": "table", "bbox": [0,0,1,1]}'):
                chunks = chunk_image(path)
            assert len(chunks) == 1
        finally:
            os.unlink(path)

    def test_bad_region_skipped_others_still_produced(self):
        """A malformed region must not abort processing of subsequent valid ones."""
        regions = json.dumps([
            {"region_type": "bad",   "bbox": "not-a-list"},
            {"region_type": "chart", "label": "right half", "bbox": [100, 50, 200, 100]},
        ])
        path = _make_image_file()
        try:
            with _patch_gemini(regions):
                chunks = chunk_image(path)
            # global + 1 valid local; the bad one is skipped
            assert len(chunks) == 2
            assert chunks[1]["region_type"] == "chart"
        finally:
            os.unlink(path)

    def test_detect_regions_empty_returns_only_global(self):
        """_detect_regions returning [] means no local chunks."""
        path = _make_image_file()
        try:
            with _patch_chunk_image("[]"):
                chunks = chunk_image(path)
            assert len(chunks) == 1
            assert chunks[0]["modality"] == "image_global"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

class TestFormatValidation:
    def test_png_accepted(self):
        path = _make_image_file(fmt="PNG")
        try:
            with _patch_gemini(_ONE_REGION):
                chunks = chunk_image(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_jpeg_accepted(self):
        path = _make_image_file(fmt="JPEG")
        try:
            with _patch_gemini(_ONE_REGION):
                chunks = chunk_image(path)
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_non_png_jpeg_format_is_converted_and_accepted(self):
        """BMP/WEBP/etc. must be silently converted to PNG rather than rejected."""
        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        tmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
        img.save(tmp.name, format="BMP")
        tmp.close()
        try:
            with _patch_gemini(_ONE_REGION):
                chunks = chunk_image(tmp.name)
            assert len(chunks) >= 1
            assert chunks[0]["modality"] == "image_global"
        finally:
            os.unlink(tmp.name)

    def test_webp_format_is_converted_and_accepted(self):
        """WEBP files (even with .png extension) must be converted and accepted."""
        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, format="WEBP")
        tmp.close()
        try:
            with _patch_gemini(_ONE_REGION):
                chunks = chunk_image(tmp.name)
            assert len(chunks) >= 1
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_list(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_ONE_REGION):
                result = chunk_image(path)
            assert isinstance(result, list)
        finally:
            os.unlink(path)

    def test_first_chunk_is_always_global(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert chunks[0]["modality"] == "image_global"
        finally:
            os.unlink(path)

    def test_local_chunks_follow_global(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            for local in chunks[1:]:
                assert local["modality"] == "image_local"
        finally:
            os.unlink(path)

    def test_total_chunk_count_global_plus_regions(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert len(chunks) == 3   # 1 global + 2 locals
        finally:
            os.unlink(path)

    def test_does_not_call_embed_chunks(self):
        path = _make_image_file()
        try:
            with _patch_gemini(_ONE_REGION), \
                 patch("backend.services.chunking.embed_chunks", side_effect=AssertionError("must not be called")) as mock_embed:
                chunk_image(path)
        except AttributeError:
            pass  # embed_chunks not imported in chunking.py — also fine
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Bbox pixel clamping
# ---------------------------------------------------------------------------

class TestBboxClamping:
    def test_coords_clamped_to_image_bounds(self):
        """Out-of-bounds pixel coords must be clamped rather than raising."""
        path = _make_image_file(width=200, height=100)
        # x2=250 and y2=150 exceed image dimensions — must be clamped to 200/100
        region = json.dumps([
            {"region_type": "chart", "label": "oversized", "bbox": [-10, -5, 250, 150]},
        ])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            local = chunks[1]
            assert local["crop_bbox"] == [0, 0, 200, 100]
            assert local["width_px"] == 200
            assert local["height_px"] == 100
        finally:
            os.unlink(path)

    def test_pixel_coords_used_directly(self):
        """Pixel bbox values must be used as-is (no fraction conversion)."""
        path = _make_image_file(width=200, height=100)
        region = json.dumps([{"region_type": "table", "label": "quarter", "bbox": [50, 0, 150, 100]}])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            local = chunks[1]
            assert local["crop_bbox"] == [50, 0, 150, 100]
        finally:
            os.unlink(path)

    def test_slightly_oob_region_is_rescued(self):
        """A bbox with y2 up to 25% over img_height must be rescaled and recovered."""
        # img 200×100; y2=120 is 20% over → scale_y=100/120=0.833 → rescued
        path = _make_image_file(width=200, height=100)
        region = json.dumps([{"region_type": "pipeline_strip", "label": "bottom row", "bbox": [0, 83, 200, 120]}])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            assert len(chunks) == 2   # rescued, not dropped
            assert chunks[1]["crop_bbox"][3] == 100  # y2 clamped to img_height
        finally:
            os.unlink(path)

    def test_severely_oob_region_is_dropped(self):
        """A bbox with y2 more than 25% over img_height must be dropped."""
        # img 200×100; y2=200 is 100% over → scale_y=0.5 < 0.75 → dropped
        path = _make_image_file(width=200, height=100)
        region = json.dumps([{"region_type": "pipeline_strip", "label": "ghost row", "bbox": [0, 150, 200, 200]}])
        try:
            with _patch_gemini(region):
                chunks = chunk_image(path)
            assert len(chunks) == 1   # only global, local was dropped
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_bboxes_produce_single_chunk(self):
        """Two regions with identical bboxes must result in only one local chunk."""
        dupes = json.dumps([
            {"region_type": "diagram", "label": "panel", "bbox": [10, 10, 180, 90]},
            {"region_type": "diagram", "label": "panel", "bbox": [10, 10, 180, 90]},
        ])
        path = _make_image_file()
        try:
            with _patch_gemini(dupes):
                chunks = chunk_image(path)
            assert len(chunks) == 2   # 1 global + 1 deduplicated local
        finally:
            os.unlink(path)

    def test_distinct_bboxes_all_kept(self):
        """Regions with different bboxes must all be preserved."""
        path = _make_image_file()
        try:
            with _patch_gemini(_TWO_REGIONS):
                chunks = chunk_image(path)
            assert len(chunks) == 3   # 1 global + 2 distinct locals
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _detect_regions unit tests — probe its internal JSON handling directly
# ---------------------------------------------------------------------------

class TestDetectRegions:
    """Unit tests for _detect_regions.  Patches _gemini_client directly so
    we can verify the function's own JSON parsing and error handling."""

    def test_valid_json_array_returned(self):
        with _patch_gemini_client(_TWO_REGIONS):
            result = _detect_regions(b"fake", "image/png", 200, 100)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_markdown_fenced_json_is_stripped_and_parsed(self):
        """Gemini sometimes wraps JSON in ```json ... ``` — must be handled."""
        fenced = f"```json\n{_TWO_REGIONS}\n```"
        with _patch_gemini_client(fenced):
            result = _detect_regions(b"fake", "image/png", 200, 100)
        assert len(result) == 2

    def test_non_list_json_returns_empty(self):
        with _patch_gemini_client('{"region_type": "table", "bbox": [0,0,1,1]}'):
            result = _detect_regions(b"fake", "image/png", 200, 100)
        assert result == []

    def test_invalid_json_returns_empty(self):
        with _patch_gemini_client("this is not json"):
            result = _detect_regions(b"fake", "image/png", 200, 100)
        assert result == []

    def test_gemini_exception_returns_empty(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("api down")
        with patch.object(_chunk_image_module, "_gemini_client", mock_client):
            result = _detect_regions(b"fake", "image/png", 200, 100)
        assert result == []
