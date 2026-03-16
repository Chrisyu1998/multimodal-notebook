"""
Tests for backend.services.chunking.chunk_video.

All scene detection, ffmpeg, and Gemini calls are mocked — no network or media I/O.
Fake temp files are used as filepath stand-ins; actual content is never read.
"""

import hashlib
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

# Stub optional heavy dependencies so the module can be imported without
# requiring ffmpeg-python or scenedetect to be installed in the test env.
for _dep in ("ffmpeg", "scenedetect", "scenedetect.detectors"):
    sys.modules.setdefault(_dep, MagicMock())

from backend.services.chunking.chunk_video import (  # noqa: E402
    chunk_video,
    _split_long_scenes,
)

_mod = sys.modules["backend.services.chunking.chunk_video"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_VIDEO_BYTES_A = b"fake-mp4-scene-0"
_FAKE_VIDEO_BYTES_B = b"fake-mp4-scene-1"
_FAKE_SUMMARY = "A person walks through a sunlit corridor carrying documents."

_CHUNK_A_KEYS = {
    "type", "video_bytes", "text", "source", "page", "chunk_index",
    "modality", "parent_scene_id", "start_time_seconds", "end_time_seconds",
    "scene_index", "forced_split",
}
_CHUNK_B_KEYS = {
    "type", "text", "source", "page", "chunk_index",
    "modality", "parent_scene_id", "start_time_seconds", "end_time_seconds",
    "scene_index",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video_file() -> str:
    """Create an empty temp file that acts as a filepath stand-in."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    return tmp.name


def _patch_detect(scenes: list[tuple[float, float]]):
    """Patch _detect_scenes to return *scenes* without touching PySceneDetect."""
    return patch.object(_mod, "_detect_scenes", return_value=scenes)


def _patch_clip(video_bytes_seq: list[bytes]):
    """Patch _extract_clip_bytes to yield successive bytes from *video_bytes_seq*."""
    iterator = iter(video_bytes_seq)
    return patch.object(_mod, "_extract_clip_bytes", side_effect=lambda *_: next(iterator))


def _patch_summary(text: str = _FAKE_SUMMARY):
    return patch.object(_mod, "_generate_visual_summary", return_value=text)


def _two_scene_chunks(path: str) -> list[dict]:
    """Run chunk_video with two clean scenes; each scene has a valid summary."""
    with (
        _patch_detect([(0.0, 30.0), (30.0, 60.0)]),
        _patch_clip([_FAKE_VIDEO_BYTES_A, _FAKE_VIDEO_BYTES_B]),
        _patch_summary(_FAKE_SUMMARY),
    ):
        return chunk_video(path)


# ---------------------------------------------------------------------------
# _split_long_scenes — pure function, no mocks needed
# ---------------------------------------------------------------------------

class TestSplitLongScenes:
    def test_short_scene_passes_through_unchanged(self):
        result = _split_long_scenes([(0.0, 60.0)])
        assert result == [(0.0, 60.0, False)]

    def test_exactly_120s_is_not_split(self):
        result = _split_long_scenes([(0.0, 120.0)])
        assert result == [(0.0, 120.0, False)]

    def test_scene_over_120s_is_force_split(self):
        result = _split_long_scenes([(0.0, 250.0)])
        assert len(result) > 1

    def test_all_subsegments_of_forced_split_flagged_true(self):
        result = _split_long_scenes([(0.0, 250.0)])
        for _, _, forced in result:
            assert forced is True

    def test_natural_scene_flagged_false(self):
        result = _split_long_scenes([(0.0, 30.0), (30.0, 60.0)])
        for _, _, forced in result:
            assert forced is False

    def test_forced_split_applies_5s_overlap(self):
        """Second sub-segment must start 5s before the first sub-segment's end."""
        result = _split_long_scenes([(0.0, 250.0)])
        assert len(result) >= 2
        first_end = result[0][1]
        second_start = result[1][0]
        assert second_start == pytest.approx(first_end - 5.0)

    def test_no_subsegment_exceeds_120s(self):
        result = _split_long_scenes([(0.0, 500.0)])
        for start, end, _ in result:
            assert end - start <= 120.0 + 1e-9

    def test_multiple_natural_scenes_all_preserved(self):
        raw = [(0.0, 20.0), (20.0, 50.0), (50.0, 90.0)]
        result = _split_long_scenes(raw)
        assert len(result) == 3

    def test_mixed_scenes_split_only_long_ones(self):
        raw = [(0.0, 30.0), (30.0, 210.0), (210.0, 240.0)]
        result = _split_long_scenes(raw)
        forced_flags = [forced for _, _, forced in result]
        # first and last are natural; middle is split into forced sub-segments
        assert forced_flags[0] is False
        assert forced_flags[-1] is False
        assert any(f is True for f in forced_flags)


# ---------------------------------------------------------------------------
# Chunk A schema
# ---------------------------------------------------------------------------

class TestChunkASchema:
    def test_required_keys_present(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            chunk_a_list = [c for c in chunks if c["modality"] == "video_clip"]
            assert chunk_a_list, "expected at least one Chunk A"
            for ch in chunk_a_list:
                missing = _CHUNK_A_KEYS - ch.keys()
                assert not missing, f"Chunk A missing keys: {missing}"
        finally:
            os.unlink(path)

    def test_type_is_video(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_clip"]:
                assert ch["type"] == "video"
        finally:
            os.unlink(path)

    def test_video_bytes_is_bytes(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_clip"]:
                assert isinstance(ch["video_bytes"], bytes) and len(ch["video_bytes"]) > 0
        finally:
            os.unlink(path)

    def test_source_is_filepath(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in chunks:
                assert ch["source"] == path
        finally:
            os.unlink(path)

    def test_page_is_zero(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in chunks:
                assert ch["page"] == 0
        finally:
            os.unlink(path)

    def test_forced_split_is_bool(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_clip"]:
                assert isinstance(ch["forced_split"], bool)
        finally:
            os.unlink(path)

    def test_forced_split_false_for_natural_scenes(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_clip"]:
                assert ch["forced_split"] is False
        finally:
            os.unlink(path)

    def test_text_is_visual_summary_when_available(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_clip"]:
                assert ch["text"] == _FAKE_SUMMARY
        finally:
            os.unlink(path)

    def test_text_fallback_when_summary_empty(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(""),          # empty → fallback
            ):
                chunks = chunk_video(path)
            ch = next(c for c in chunks if c["modality"] == "video_clip")
            assert "scene" in ch["text"].lower() or os.path.basename(path) in ch["text"]
        finally:
            os.unlink(path)

    def test_start_end_times_correct(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(5.0, 35.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            ch = next(c for c in chunks if c["modality"] == "video_clip")
            assert ch["start_time_seconds"] == pytest.approx(5.0)
            assert ch["end_time_seconds"] == pytest.approx(35.0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Chunk B schema
# ---------------------------------------------------------------------------

class TestChunkBSchema:
    def test_required_keys_present(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            chunk_b_list = [c for c in chunks if c["modality"] == "video_summary"]
            assert chunk_b_list, "expected at least one Chunk B"
            for ch in chunk_b_list:
                missing = _CHUNK_B_KEYS - ch.keys()
                assert not missing, f"Chunk B missing keys: {missing}"
        finally:
            os.unlink(path)

    def test_type_is_document(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_summary"]:
                assert ch["type"] == "document"
        finally:
            os.unlink(path)

    def test_text_is_visual_summary(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            for ch in [c for c in chunks if c["modality"] == "video_summary"]:
                assert ch["text"] == _FAKE_SUMMARY
        finally:
            os.unlink(path)

    def test_chunk_b_omitted_when_summary_empty(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(""),
            ):
                chunks = chunk_video(path)
            summaries = [c for c in chunks if c["modality"] == "video_summary"]
            assert summaries == [], "Chunk B must be skipped when summary is empty"
        finally:
            os.unlink(path)

    def test_chunk_b_shares_times_with_chunk_a(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(10.0, 40.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            a = next(c for c in chunks if c["modality"] == "video_clip")
            b = next(c for c in chunks if c["modality"] == "video_summary")
            assert b["start_time_seconds"] == a["start_time_seconds"]
            assert b["end_time_seconds"] == a["end_time_seconds"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# parent_scene_id
# ---------------------------------------------------------------------------

class TestParentSceneId:
    def test_parent_scene_id_is_sha256_of_video_bytes(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            expected = hashlib.sha256(_FAKE_VIDEO_BYTES_A).hexdigest()
            for ch in chunks:
                assert ch["parent_scene_id"] == expected
        finally:
            os.unlink(path)

    def test_parent_scene_id_identical_across_chunk_a_and_b(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            ids = {c["parent_scene_id"] for c in chunks}
            assert len(ids) == 1, f"parent_scene_id differs across chunks: {ids}"
        finally:
            os.unlink(path)

    def test_different_scenes_have_different_parent_scene_ids(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            clip_ids = [c["parent_scene_id"] for c in chunks if c["modality"] == "video_clip"]
            assert clip_ids[0] != clip_ids[1]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# chunk_index sequencing
# ---------------------------------------------------------------------------

class TestChunkIndex:
    def test_chunk_index_starts_at_zero(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            assert chunks[0]["chunk_index"] == 0
        finally:
            os.unlink(path)

    def test_chunk_index_globally_sequential(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            indices = [c["chunk_index"] for c in chunks]
            assert indices == list(range(len(chunks)))
        finally:
            os.unlink(path)

    def test_chunk_b_index_is_chunk_a_plus_one(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            a = next(c for c in chunks if c["modality"] == "video_clip")
            b = next(c for c in chunks if c["modality"] == "video_summary")
            assert b["chunk_index"] == a["chunk_index"] + 1
        finally:
            os.unlink(path)

    def test_chunk_index_sequential_with_skipped_summaries(self):
        """When scene 0 has a summary and scene 1 does not, indices must still be gapless."""
        path = _make_video_file()
        summaries = iter([_FAKE_SUMMARY, ""])
        try:
            with (
                _patch_detect([(0.0, 30.0), (30.0, 60.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A, _FAKE_VIDEO_BYTES_B]),

                patch.object(_mod, "_generate_visual_summary", side_effect=lambda *_: next(summaries)),
            ):
                chunks = chunk_video(path)
            indices = [c["chunk_index"] for c in chunks]
            assert indices == list(range(len(chunks)))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# scene_index
# ---------------------------------------------------------------------------

class TestSceneIndex:
    def test_scene_index_increments_per_scene(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            scene0 = [c for c in chunks if c.get("scene_index") == 0]
            scene1 = [c for c in chunks if c.get("scene_index") == 1]
            assert scene0 and scene1
        finally:
            os.unlink(path)

    def test_chunk_a_and_b_share_scene_index(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            a = next(c for c in chunks if c["modality"] == "video_clip")
            b = next(c for c in chunks if c["modality"] == "video_summary")
            assert a["scene_index"] == b["scene_index"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Total chunk count
# ---------------------------------------------------------------------------

class TestChunkCount:
    def test_two_scenes_with_summaries_produce_four_chunks(self):
        """Each scene with a summary produces Chunk A + Chunk B → 2 scenes = 4 chunks."""
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            assert len(chunks) == 4
        finally:
            os.unlink(path)

    def test_two_scenes_no_summaries_produce_two_chunks(self):
        """Without summaries Chunk B is skipped → 2 scenes = 2 Chunk A only."""
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0), (30.0, 60.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A, _FAKE_VIDEO_BYTES_B]),

                _patch_summary(""),
            ):
                chunks = chunk_video(path)
            assert len(chunks) == 2  # only Chunk A per scene
        finally:
            os.unlink(path)

    def test_one_scene_with_summary_produces_two_chunks(self):
        """One scene with a summary → Chunk A + Chunk B = 2 chunks."""
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            assert len(chunks) == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_list(self):
        path = _make_video_file()
        try:
            chunks = _two_scene_chunks(path)
            assert isinstance(chunks, list)
        finally:
            os.unlink(path)

    def test_returns_empty_list_when_no_scenes(self):
        path = _make_video_file()
        try:
            with _patch_detect([]):
                chunks = chunk_video(path)
            assert chunks == []
        finally:
            os.unlink(path)

    def test_does_not_call_embed_chunks(self):
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                _patch_summary(_FAKE_SUMMARY),
                patch("backend.services.chunking.chunk_video.embed_chunks",
                      side_effect=AssertionError("must not be called")),
            ):
                chunk_video(path)
        except AttributeError:
            pass   # embed_chunks not imported — also acceptable
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Robustness — bad scenes must be skipped, pipeline must continue
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_clip_extraction_failure_skips_scene_continues(self):
        """If clip extraction fails for one scene, the other scenes are still processed."""
        path = _make_video_file()
        call_count = [0]

        def clip_side_effect(*_):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("ffmpeg error")
            return _FAKE_VIDEO_BYTES_B

        try:
            with (
                _patch_detect([(0.0, 30.0), (30.0, 60.0)]),
                patch.object(_mod, "_extract_clip_bytes", side_effect=clip_side_effect),

                _patch_summary(_FAKE_SUMMARY),
            ):
                chunks = chunk_video(path)
            # Scene 0 skipped; scene 1 produces Chunk A + Chunk B (summary non-empty)
            assert len(chunks) == 2
            assert all(c["scene_index"] == 1 for c in chunks)
        finally:
            os.unlink(path)

    def test_summary_receives_video_bytes_not_keyframe(self):
        """_generate_visual_summary must be called with the raw MP4 clip bytes."""
        path = _make_video_file()
        received = []
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),
                patch.object(_mod, "_generate_visual_summary",
                             side_effect=lambda b: received.append(b) or _FAKE_SUMMARY),
            ):
                chunk_video(path)
            assert received == [_FAKE_VIDEO_BYTES_A]
        finally:
            os.unlink(path)

    def test_gemini_failure_produces_chunk_a_without_chunk_b(self):
        """Gemini call failure → empty summary → Chunk B skipped, Chunk A still emitted."""
        path = _make_video_file()
        try:
            with (
                _patch_detect([(0.0, 30.0)]),
                _patch_clip([_FAKE_VIDEO_BYTES_A]),

                patch.object(_mod, "_generate_visual_summary", return_value=""),
            ):
                chunks = chunk_video(path)
            assert len(chunks) == 1
            assert chunks[0]["modality"] == "video_clip"
        finally:
            os.unlink(path)

    def test_scene_detection_failure_returns_empty_list(self):
        path = _make_video_file()
        try:
            with patch.object(_mod, "_detect_scenes", side_effect=RuntimeError("scenedetect error")):
                chunks = chunk_video(path)
            assert chunks == []
        finally:
            os.unlink(path)
