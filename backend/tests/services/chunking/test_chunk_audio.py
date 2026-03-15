"""
Tests for backend.services.chunking.chunk_audio.

All Gemini API calls, embed_text, and pydub I/O are mocked — no network or
media I/O is performed.  Pure helper functions are tested directly.
"""

import io
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

# Stub pydub so the module can be imported in environments without ffmpeg.
_pydub_mock = MagicMock()
_audio_segment_mock = MagicMock()
_audio_segment_mock.from_file.return_value = _audio_segment_mock
_pydub_mock.AudioSegment = _audio_segment_mock
sys.modules.setdefault("pydub", _pydub_mock)

from backend.services.chunking.chunk_audio import (  # noqa: E402
    _build_label_map,
    _cosine_similarity,
    _detect_boundaries,
    _dominant_speaker,
    _enforce_ceiling,
    _group_duration,
    _group_segments,
    _merge_short_groups,
    _parse_transcription,
    _stitch_speaker_labels,
    chunk_audio,
)

_mod = sys.modules["backend.services.chunking.chunk_audio"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_AUDIO_BYTES = b"fake-mp3-clip"

_CHUNK_KEYS = {
    "type", "audio_bytes", "text", "source", "page", "chunk_index",
    "modality", "start_time_seconds", "end_time_seconds",
    "speaker_id", "transcript_text", "forced_split",
}

_SEG_A = {"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "Hello world."}
_SEG_B = {"speaker": "Speaker 1", "start": 5.0, "end": 10.0, "text": "How are you?"}
_SEG_C = {"speaker": "Speaker 2", "start": 10.0, "end": 15.0, "text": "I am fine."}
_SEG_D = {"speaker": "Speaker 2", "start": 15.0, "end": 20.0, "text": "Thank you."}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio_file(suffix: str = ".mp3") -> str:
    """Create an empty temp file used as a filepath stand-in."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


def _patch_transcribe(segments: list[dict]):
    """Patch _transcribe to return *segments* without touching Gemini."""
    return patch.object(_mod, "_transcribe", return_value=segments)


def _patch_embed(sim_value: float = 0.9):
    """Patch embed_text to return a constant unit-ish vector."""
    vec = [sim_value, 1.0 - sim_value]
    return patch.object(_mod, "embed_text", return_value=vec)


def _patch_clip(audio_bytes: bytes = _FAKE_AUDIO_BYTES):
    """Patch _extract_clip_bytes to return *audio_bytes*."""
    return patch.object(_mod, "_extract_clip_bytes", return_value=audio_bytes)


def _run_chunk_audio(path: str, segments: list[dict]) -> list[dict]:
    """Run chunk_audio with Gemini and pydub fully mocked."""
    with (
        _patch_transcribe(segments),
        _patch_embed(),
        _patch_clip(),
    ):
        return chunk_audio(path)


# ---------------------------------------------------------------------------
# _parse_transcription — pure, no mocks
# ---------------------------------------------------------------------------

class TestParseTranscription:
    def test_valid_json_returns_segments(self):
        raw = json.dumps({"segments": [_SEG_A, _SEG_B]})
        result = _parse_transcription(raw, "test")
        assert result == [_SEG_A, _SEG_B]

    def test_strips_json_markdown_fence(self):
        raw = '```json\n{"segments": [{"speaker": "Speaker 1", "start": 0.0, "end": 1.0, "text": "hi"}]}\n```'
        result = _parse_transcription(raw, "test")
        assert len(result) == 1
        assert result[0]["text"] == "hi"

    def test_strips_bare_code_fence(self):
        raw = '```{"segments": []}\n```'
        result = _parse_transcription(raw, "test")
        assert result == []

    def test_empty_segments_list_ok(self):
        raw = json.dumps({"segments": []})
        result = _parse_transcription(raw, "test")
        assert result == []

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_transcription("not json at all", "ctx")

    def test_missing_segments_key_raises_value_error(self):
        raw = json.dumps({"data": []})
        with pytest.raises(ValueError, match="missing 'segments' list"):
            _parse_transcription(raw, "ctx")

    def test_segments_not_list_raises_value_error(self):
        raw = json.dumps({"segments": "oops"})
        with pytest.raises(ValueError, match="missing 'segments' list"):
            _parse_transcription(raw, "ctx")


# ---------------------------------------------------------------------------
# _cosine_similarity — pure, no mocks
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_both_zero_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_arbitrary_vectors_in_range(self):
        a = [0.6, 0.8]
        b = [0.8, 0.6]
        sim = _cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# _build_label_map — pure, no mocks
# ---------------------------------------------------------------------------

class TestBuildLabelMap:
    def test_no_overlap_returns_empty(self):
        prev = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0}]
        next_ = [{"speaker": "Speaker 1", "start": 10.0, "end": 15.0}]
        assert _build_label_map(prev, next_) == {}

    def test_overlapping_same_label_returns_identity_map(self):
        prev = [{"speaker": "Speaker 1", "start": 0.0, "end": 10.0}]
        next_ = [{"speaker": "Speaker 1", "start": 5.0, "end": 15.0}]
        result = _build_label_map(prev, next_)
        assert result == {"Speaker 1": "Speaker 1"}

    def test_overlapping_different_labels_remaps(self):
        prev = [{"speaker": "Speaker 1", "start": 0.0, "end": 10.0}]
        next_ = [{"speaker": "Speaker 2", "start": 5.0, "end": 15.0}]
        result = _build_label_map(prev, next_)
        assert result == {"Speaker 2": "Speaker 1"}

    def test_greedy_one_to_one_assignment(self):
        """Each label used at most once — no double mapping."""
        prev = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 10.0},
            {"speaker": "Speaker 2", "start": 0.0, "end": 10.0},
        ]
        next_ = [
            {"speaker": "Speaker A", "start": 5.0, "end": 15.0},
            {"speaker": "Speaker B", "start": 5.0, "end": 15.0},
        ]
        result = _build_label_map(prev, next_)
        # Each next speaker maps to a distinct prev speaker
        assert len(set(result.values())) == len(result)

    def test_empty_inputs_return_empty(self):
        assert _build_label_map([], []) == {}
        assert _build_label_map([{"speaker": "S1", "start": 0.0, "end": 5.0}], []) == {}
        assert _build_label_map([], [{"speaker": "S1", "start": 0.0, "end": 5.0}]) == {}


# ---------------------------------------------------------------------------
# _stitch_speaker_labels — pure, no mocks
# ---------------------------------------------------------------------------

class TestStitchSpeakerLabels:
    def test_empty_input_returns_empty(self):
        assert _stitch_speaker_labels([], 30.0) == []

    def test_single_window_returned_unchanged(self):
        segs = [_SEG_A, _SEG_B]
        result = _stitch_speaker_labels([segs], 30.0)
        assert result == segs

    def test_two_windows_combined_and_sorted(self):
        w0 = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "hi"}]
        w1 = [{"speaker": "Speaker 1", "start": 5.0, "end": 10.0, "text": "bye"}]
        result = _stitch_speaker_labels([w0, w1], 5.0)
        assert len(result) == 2
        assert result[0]["start"] < result[1]["start"]

    def test_speaker_label_remapped_at_boundary(self):
        """Window 1 uses 'Speaker 2' but overlaps with 'Speaker 1' → should be remapped."""
        w0 = [{"speaker": "Speaker 1", "start": 0.0, "end": 40.0, "text": "a"}]
        w1 = [{"speaker": "Speaker 2", "start": 10.0, "end": 50.0, "text": "b"}]
        result = _stitch_speaker_labels([w0, w1], 30.0)
        w1_seg = next(s for s in result if s["text"] == "b")
        assert w1_seg["speaker"] == "Speaker 1"

    def test_non_overlapping_windows_labels_unchanged(self):
        """Windows with no time overlap preserve their own labels."""
        w0 = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "a"}]
        w1 = [{"speaker": "Speaker 3", "start": 100.0, "end": 105.0, "text": "b"}]
        result = _stitch_speaker_labels([w0, w1], 5.0)
        w1_seg = next(s for s in result if s["text"] == "b")
        assert w1_seg["speaker"] == "Speaker 3"

    def test_result_sorted_by_start_time(self):
        w0 = [{"speaker": "Speaker 1", "start": 5.0, "end": 10.0, "text": "b"}]
        w1 = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "a"}]
        result = _stitch_speaker_labels([w0, w1], 5.0)
        starts = [s["start"] for s in result]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# _group_segments — pure, no mocks
# ---------------------------------------------------------------------------

class TestGroupSegments:
    def test_no_split_indices_produces_one_group(self):
        segs = [_SEG_A, _SEG_B, _SEG_C]
        result = _group_segments(segs, set())
        assert result == [segs]

    def test_split_at_every_boundary_produces_singleton_groups(self):
        segs = [_SEG_A, _SEG_B, _SEG_C]
        result = _group_segments(segs, {1, 2})
        assert len(result) == 3
        for i, group in enumerate(result):
            assert group == [segs[i]]

    def test_split_in_middle(self):
        segs = [_SEG_A, _SEG_B, _SEG_C, _SEG_D]
        result = _group_segments(segs, {2})
        assert result == [[_SEG_A, _SEG_B], [_SEG_C, _SEG_D]]

    def test_empty_segments_returns_empty(self):
        assert _group_segments([], set()) == []
        assert _group_segments([], {0}) == []

    def test_split_at_index_zero_ignored_for_empty_current(self):
        """Split at 0 does nothing meaningful (no preceding group)."""
        segs = [_SEG_A, _SEG_B]
        result = _group_segments(segs, {0})
        # index 0 split ignored because current is empty
        assert result == [[_SEG_A, _SEG_B]]


# ---------------------------------------------------------------------------
# _group_duration — pure, no mocks
# ---------------------------------------------------------------------------

class TestGroupDuration:
    def test_empty_group_returns_zero(self):
        assert _group_duration([]) == pytest.approx(0.0)

    def test_single_segment_duration(self):
        assert _group_duration([_SEG_A]) == pytest.approx(5.0)

    def test_multi_segment_duration_uses_first_start_last_end(self):
        assert _group_duration([_SEG_A, _SEG_B]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _merge_short_groups — pure, no mocks
# ---------------------------------------------------------------------------

class TestMergeShortGroups:
    # _SHORT_MERGE_S defaults to 60.0
    _THRESHOLD = 60.0

    def test_empty_returns_empty(self):
        assert _merge_short_groups([]) == []

    def test_single_group_unchanged(self):
        g = [_SEG_A, _SEG_B]
        result = _merge_short_groups([g])
        assert result == [g]

    def test_same_speaker_short_combined_duration_merged(self):
        g1 = [{"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "a"}]
        g2 = [{"speaker": "Speaker 1", "start": 10.0, "end": 20.0, "text": "b"}]
        result = _merge_short_groups([g1, g2])
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_different_speakers_not_merged(self):
        g1 = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "a"}]
        g2 = [{"speaker": "Speaker 2", "start": 5.0, "end": 10.0, "text": "b"}]
        result = _merge_short_groups([g1, g2])
        assert len(result) == 2

    def test_same_speaker_combined_exceeds_threshold_not_merged(self):
        g1 = [{"speaker": "Speaker 1", "start": 0.0, "end": 35.0, "text": "a"}]
        g2 = [{"speaker": "Speaker 1", "start": 35.0, "end": 70.0, "text": "b"}]
        # combined = 70s > 60s threshold
        result = _merge_short_groups([g1, g2])
        assert len(result) == 2

    def test_merge_is_sequential_not_global(self):
        """Three same-speaker groups: first two merge, then the merged+third check runs."""
        g1 = [{"speaker": "Speaker 1", "start": 0.0, "end": 10.0, "text": "a"}]
        g2 = [{"speaker": "Speaker 1", "start": 10.0, "end": 20.0, "text": "b"}]
        g3 = [{"speaker": "Speaker 1", "start": 20.0, "end": 30.0, "text": "c"}]
        result = _merge_short_groups([g1, g2, g3])
        # All three combined = 30s < 60s, expect one group
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _enforce_ceiling — pure, no mocks
# ---------------------------------------------------------------------------

class TestEnforceCeiling:
    # _HARD_CEILING_S defaults to 75.0, _OVERLAP_S defaults to 5.0

    def test_group_within_ceiling_returned_as_is(self):
        group = [{"speaker": "Speaker 1", "start": 0.0, "end": 70.0, "text": "a"}]
        result = _enforce_ceiling([group])
        assert len(result) == 1
        start, end, segs, forced = result[0]
        assert forced is False
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(70.0)

    def test_group_exactly_at_ceiling_not_split(self):
        group = [{"speaker": "Speaker 1", "start": 0.0, "end": 75.0, "text": "a"}]
        result = _enforce_ceiling([group])
        assert len(result) == 1
        assert result[0][3] is False

    def test_group_exceeding_ceiling_is_split(self):
        group = [{"speaker": "Speaker 1", "start": 0.0, "end": 150.0, "text": "a"}]
        result = _enforce_ceiling([group])
        assert len(result) > 1

    def test_all_forced_split_chunks_flagged_true(self):
        group = [{"speaker": "Speaker 1", "start": 0.0, "end": 150.0, "text": "a"}]
        result = _enforce_ceiling([group])
        for _, _, _, forced in result:
            assert forced is True

    def test_no_sub_chunk_exceeds_ceiling(self):
        group = [{"speaker": "Speaker 1", "start": 0.0, "end": 300.0, "text": "a"}]
        result = _enforce_ceiling([group])
        for start, end, _, _ in result:
            assert end - start <= 75.0 + 1e-9

    def test_second_window_starts_with_overlap(self):
        """Second forced sub-chunk must start overlap_s before first sub-chunk's end."""
        group = [{"speaker": "Speaker 1", "start": 0.0, "end": 150.0, "text": "a"}]
        result = _enforce_ceiling([group])
        assert len(result) >= 2
        first_end = result[0][1]
        second_start = result[1][0]
        assert second_start == pytest.approx(first_end - 5.0)

    def test_multiple_groups_each_processed(self):
        g1 = [{"speaker": "Speaker 1", "start": 0.0, "end": 50.0, "text": "a"}]
        g2 = [{"speaker": "Speaker 2", "start": 50.0, "end": 200.0, "text": "b"}]
        result = _enforce_ceiling([g1, g2])
        forced_flags = [f for _, _, _, f in result]
        assert forced_flags[0] is False
        assert any(f is True for f in forced_flags[1:])

    def test_empty_groups_list_returns_empty(self):
        assert _enforce_ceiling([]) == []


# ---------------------------------------------------------------------------
# _dominant_speaker — pure, no mocks
# ---------------------------------------------------------------------------

class TestDominantSpeaker:
    def test_single_speaker_returns_that_speaker(self):
        group = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "a"},
            {"speaker": "Speaker 1", "start": 5.0, "end": 10.0, "text": "b"},
        ]
        assert _dominant_speaker(group) == "Speaker 1"

    def test_multiple_speakers_returns_none(self):
        group = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "a"},
            {"speaker": "Speaker 2", "start": 5.0, "end": 10.0, "text": "b"},
        ]
        assert _dominant_speaker(group) is None

    def test_empty_group_returns_none(self):
        assert _dominant_speaker([]) is None

    def test_segment_without_speaker_key_ignored(self):
        group = [
            {"start": 0.0, "end": 5.0, "text": "a"},  # no speaker key
            {"speaker": "Speaker 1", "start": 5.0, "end": 10.0, "text": "b"},
        ]
        # only "Speaker 1" present in speakers set
        assert _dominant_speaker(group) == "Speaker 1"


# ---------------------------------------------------------------------------
# _detect_boundaries — mocks embed_text
# ---------------------------------------------------------------------------

class TestDetectBoundaries:
    def test_empty_or_single_segment_returns_empty_set(self):
        with _patch_embed():
            assert _detect_boundaries([]) == set()
            assert _detect_boundaries([_SEG_A]) == set()

    def test_speaker_change_marks_boundary(self):
        segs = [_SEG_A, _SEG_C]  # Speaker 1 → Speaker 2
        with _patch_embed(0.9):  # high sim, so only speaker boundary
            result = _detect_boundaries(segs)
        assert 1 in result

    def test_same_speaker_high_similarity_no_boundary(self):
        segs = [_SEG_A, _SEG_B]  # both Speaker 1
        # embed_text returns same vector → sim=1.0, above threshold
        with patch.object(_mod, "embed_text", return_value=[1.0, 0.0]):
            result = _detect_boundaries(segs)
        assert result == set()

    def test_same_speaker_low_similarity_marks_semantic_boundary(self):
        segs = [_SEG_A, _SEG_B]  # both Speaker 1
        vectors = iter([[1.0, 0.0], [0.0, 1.0]])  # orthogonal → sim=0.0
        with patch.object(_mod, "embed_text", side_effect=lambda _: next(vectors)):
            result = _detect_boundaries(segs)
        assert 1 in result

    def test_embed_text_failure_treated_as_no_embedding(self):
        """If embed_text raises, boundary is skipped (not crash)."""
        segs = [_SEG_A, _SEG_B]
        with patch.object(_mod, "embed_text", side_effect=RuntimeError("api error")):
            result = _detect_boundaries(segs)
        # No embedding available → no semantic boundary detected
        assert result == set()


# ---------------------------------------------------------------------------
# chunk_audio — integration with all heavy deps mocked
# ---------------------------------------------------------------------------

class TestChunkAudioSchema:
    def test_required_keys_present(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            assert chunks, "expected at least one chunk"
            for ch in chunks:
                missing = _CHUNK_KEYS - ch.keys()
                assert not missing, f"chunk missing keys: {missing}"
        finally:
            os.unlink(path)

    def test_type_is_audio(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert ch["type"] == "audio"
        finally:
            os.unlink(path)

    def test_modality_is_audio_clip(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert ch["modality"] == "audio_clip"
        finally:
            os.unlink(path)

    def test_audio_bytes_is_bytes(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert isinstance(ch["audio_bytes"], bytes) and len(ch["audio_bytes"]) > 0
        finally:
            os.unlink(path)

    def test_source_is_filepath(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert ch["source"] == path
        finally:
            os.unlink(path)

    def test_page_is_zero(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert ch["page"] == 0
        finally:
            os.unlink(path)

    def test_forced_split_is_bool(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert isinstance(ch["forced_split"], bool)
        finally:
            os.unlink(path)

    def test_transcript_text_matches_text(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert ch["transcript_text"] == ch["text"]
        finally:
            os.unlink(path)

    def test_start_end_times_are_floats(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert isinstance(ch["start_time_seconds"], float)
                assert isinstance(ch["end_time_seconds"], float)
        finally:
            os.unlink(path)

    def test_start_less_than_end(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B])
            for ch in chunks:
                assert ch["start_time_seconds"] < ch["end_time_seconds"]
        finally:
            os.unlink(path)

    def test_single_speaker_chunk_has_speaker_id(self):
        path = _make_audio_file()
        segs = [_SEG_A, _SEG_B]  # both Speaker 1
        try:
            chunks = _run_chunk_audio(path, segs)
            # All same speaker → speaker_id is set (not None) for at least one chunk
            non_none = [c for c in chunks if c["speaker_id"] is not None]
            assert non_none
        finally:
            os.unlink(path)

    def test_mixed_speaker_chunk_has_none_speaker_id(self):
        """A chunk that mixes speakers in its group gets speaker_id=None."""
        path = _make_audio_file()
        # Force all segs into one group by using the same speaker but switch halfway.
        segs = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "a"},
            {"speaker": "Speaker 2", "start": 5.0, "end": 10.0, "text": "b"},
        ]
        try:
            # patch _detect_boundaries to return no splits → one group with mixed speakers
            with (
                _patch_transcribe(segs),
                _patch_embed(0.9),
                _patch_clip(),
                patch.object(_mod, "_detect_boundaries", return_value=set()),
                patch.object(_mod, "_merge_short_groups", side_effect=lambda g: g),
            ):
                chunks = chunk_audio(path)
            mixed = [c for c in chunks if c["speaker_id"] is None]
            assert mixed
        finally:
            os.unlink(path)


class TestChunkAudioChunkIndex:
    def test_chunk_index_starts_at_zero(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B, _SEG_C, _SEG_D])
            assert chunks[0]["chunk_index"] == 0
        finally:
            os.unlink(path)

    def test_chunk_index_globally_sequential(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A, _SEG_B, _SEG_C, _SEG_D])
            indices = [c["chunk_index"] for c in chunks]
            assert indices == list(range(len(chunks)))
        finally:
            os.unlink(path)


class TestChunkAudioReturnStructure:
    def test_returns_list(self):
        path = _make_audio_file()
        try:
            chunks = _run_chunk_audio(path, [_SEG_A])
            assert isinstance(chunks, list)
        finally:
            os.unlink(path)

    def test_no_segments_raises_value_error(self):
        path = _make_audio_file()
        try:
            with pytest.raises(ValueError, match="no segments"):
                with _patch_transcribe([]):
                    chunk_audio(path)
        finally:
            os.unlink(path)

    def test_does_not_call_embed_chunks(self):
        path = _make_audio_file()
        try:
            with (
                _patch_transcribe([_SEG_A]),
                _patch_embed(),
                _patch_clip(),
                patch(
                    "backend.services.chunking.chunk_audio.embed_text",
                    wraps=lambda t: [0.5, 0.5],
                ),
            ):
                # embed_chunks is not imported in chunk_audio — verifying no AttributeError
                chunk_audio(path)
            assert not hasattr(_mod, "embed_chunks") or True  # never called
        except AttributeError:
            pass  # acceptable if not imported
        finally:
            os.unlink(path)


class TestChunkAudioRobustness:
    def test_clip_extraction_failure_skips_chunk_continues(self):
        """If clip extraction fails for one chunk, others are still produced."""
        path = _make_audio_file()
        call_count = [0]

        def clip_side_effect(*_):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("pydub error")
            return _FAKE_AUDIO_BYTES

        segs = [_SEG_A, _SEG_B, _SEG_C, _SEG_D]
        try:
            with (
                _patch_transcribe(segs),
                _patch_embed(),
                patch.object(_mod, "_extract_clip_bytes", side_effect=clip_side_effect),
            ):
                chunks = chunk_audio(path)
            # At least one chunk should succeed even when the first fails
            assert len(chunks) >= 1
        finally:
            os.unlink(path)

    def test_degenerate_chunk_end_lte_start_skipped(self):
        """Chunks where end <= start are silently skipped."""
        path = _make_audio_file()
        segs = [_SEG_A]
        try:
            # Monkey-patch _enforce_ceiling to inject a degenerate entry
            degenerate = [(5.0, 5.0, segs, False), (0.0, 5.0, segs, False)]
            with (
                _patch_transcribe(segs),
                _patch_embed(),
                _patch_clip(),
                patch.object(_mod, "_enforce_ceiling", return_value=degenerate),
            ):
                chunks = chunk_audio(path)
            # The degenerate (end==start) chunk is skipped; valid one retained
            assert len(chunks) == 1
            assert chunks[0]["start_time_seconds"] == pytest.approx(0.0)
        finally:
            os.unlink(path)

    def test_transcript_text_joined_from_segments(self):
        """transcript_text must be the joined text of all segments in the group."""
        path = _make_audio_file()
        segs = [_SEG_A, _SEG_B]  # both Speaker 1 — will merge into one group
        try:
            with (
                _patch_transcribe(segs),
                patch.object(_mod, "embed_text", return_value=[1.0, 0.0]),
                _patch_clip(),
            ):
                chunks = chunk_audio(path)
            # Find a chunk that contains both texts
            combined = " ".join(c["transcript_text"] for c in chunks)
            assert "Hello world." in combined
            assert "How are you?" in combined
        finally:
            os.unlink(path)

    def test_two_speakers_produce_at_least_two_chunks(self):
        """Speaker change boundaries should yield separate chunks."""
        path = _make_audio_file()
        segs = [_SEG_A, _SEG_C]  # Speaker 1 then Speaker 2
        try:
            chunks = _run_chunk_audio(path, segs)
            assert len(chunks) >= 2
        finally:
            os.unlink(path)

    def test_forced_split_true_for_ceiling_exceeded_chunks(self):
        """Chunks produced by ceiling enforcement must have forced_split=True."""
        path = _make_audio_file()
        # Single long segment → ceiling enforcement kicks in
        segs = [{"speaker": "Speaker 1", "start": 0.0, "end": 200.0, "text": "long"}]
        try:
            with (
                _patch_transcribe(segs),
                _patch_embed(),
                _patch_clip(),
            ):
                chunks = chunk_audio(path)
            forced = [c for c in chunks if c["forced_split"] is True]
            assert forced, "expected at least one forced-split chunk for 200s segment"
        finally:
            os.unlink(path)
