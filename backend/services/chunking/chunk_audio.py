"""
Audio chunking — Speaker-Aligned Semantic Chunking.

Implements five steps:
1. Transcribe with Gemini Flash (word-level timestamps + speaker labels).
2. Embed each segment with embed_text() to detect semantic boundaries;
   also detect speaker-turn boundaries.
3. Group segments at boundary split points, merge short same-speaker
   groups (< 60 s combined), then enforce 75 s hard ceiling.
4. Slice audio clips via pydub and export as MP3 bytes.
5. Build and return chunk dicts.

Chunk shape:
    type="audio", audio_bytes, text, source, page, chunk_index,
    modality="audio_clip", start_time_seconds, end_time_seconds,
    speaker_id, transcript_text, forced_split
"""

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydub import AudioSegment

import backend.config as config
from backend.services.embeddings import embed_text

_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

_SEMANTIC_THRESHOLD: float = config.AUDIO_SEMANTIC_SIMILARITY_THRESHOLD
_SHORT_MERGE_S: float = config.AUDIO_SHORT_MERGE_THRESHOLD_S
_HARD_CEILING_S: float = config.AUDIO_HARD_CEILING_S
_OVERLAP_S: float = config.AUDIO_FORCED_SPLIT_OVERLAP_S
_TRANSCRIPTION_SEGMENT_S: float = config.AUDIO_TRANSCRIPTION_SEGMENT_S
_TRANSCRIPTION_OVERLAP_S: float = config.AUDIO_TRANSCRIPTION_OVERLAP_S
_TRANSCRIPTION_MAX_TOKENS: int = config.AUDIO_TRANSCRIPTION_MAX_TOKENS
_TRANSCRIPTION_MAX_RETRIES: int = config.AUDIO_TRANSCRIPTION_MAX_RETRIES
_TRANSCRIPTION_MAX_WORKERS: int = config.AUDIO_TRANSCRIPTION_MAX_WORKERS

_TRANSCRIPTION_PROMPT: str = (
    "Transcribe this audio with speaker diarization and precise timestamps. "
    "Rules:\n"
    "1. Timestamps must be in seconds with two decimal places (e.g. 12.34).\n"
    "2. Each segment is a continuous utterance by one speaker without interruption.\n"
    "3. Speaker labels must be consistent throughout: use 'Speaker 1', 'Speaker 2', etc. "
    "Do not use names, roles, or descriptions even if inferrable.\n"
    "4. If two speakers overlap, output each as a separate segment with overlapping time ranges.\n"
    "5. Exclude non-speech sounds (laughter, music, silence). "
    "If speech is unintelligible, use [inaudible].\n"
    "6. Do not merge segments across speaker changes, even if the same speaker resumes.\n"
    "7. Return JSON only — no explanation, no markdown, no code fences.\n"
    "Format:\n"
    '{"segments": [\n'
    '  {"speaker": "Speaker 1", "start": 0.00, "end": 4.20, '
    '"text": "Hello, welcome to the meeting."},\n'
    '  {"speaker": "Speaker 2", "start": 4.50, "end": 9.10, '
    '"text": "Thanks for having me."},\n'
    '  {"speaker": "Speaker 1", "start": 9.10, "end": 13.50, '
    '"text": "Today we are covering Q3 results."}\n'
    "]}"
)

# Supported input MIME types keyed by file extension
_AUDIO_MIME: dict[str, str] = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".mp4": "audio/mp4",
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Transcription
# ──────────────────────────────────────────────────────────────────────────────

def _parse_transcription(raw: str, context: str) -> list[dict]:
    """Strip markdown fences and parse transcription JSON into a segment list.

    Raises ValueError with context info if the response is not valid JSON or
    is missing the expected 'segments' list.  The prompt instructs the model
    to return bare JSON, but this function defensively handles the common case
    where Gemini wraps the output in ```json ... ``` anyway.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"chunk_audio: transcription response is not valid JSON ({context}): "
            f"{exc!r}. First 200 chars: {raw[:200]}"
        ) from exc

    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(
            f"chunk_audio: transcription JSON missing 'segments' list ({context}). "
            f"Keys present: {list(data.keys())}"
        )
    return segments


def _transcribe_window(
    audio: AudioSegment,
    window_start_ms: int,
    window_end_ms: int,
    offset_s: float,
    window_idx: int,
    filename: str,
) -> list[dict]:
    """Transcribe a single audio window and return timestamp-offset segments.

    Raises ValueError on API failure or unparseable JSON so the caller can
    propagate a clear error message.
    """
    buf = io.BytesIO()
    audio[window_start_ms:window_end_ms].export(buf, format="mp3")
    audio_part = genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/mp3")

    _transcription_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "start", "end", "text"],
                },
            }
        },
        "required": ["segments"],
    }

    try:
        response = _gemini_client.models.generate_content(
            model=config.AUDIO_TRANSCRIPTION_MODEL,
            contents=[audio_part, _TRANSCRIPTION_PROMPT],
            config=genai_types.GenerateContentConfig(
                max_output_tokens=_TRANSCRIPTION_MAX_TOKENS,
                response_mime_type="application/json",
                response_schema=_transcription_schema,
            ),
        )
    except Exception as exc:
        raise ValueError(
            f"chunk_audio: Gemini transcription failed for window {window_idx} "
            f"in {filename}: {exc}"
        ) from exc

    raw = "".join(
        part.text
        for part in response.candidates[0].content.parts
        if hasattr(part, "text") and part.text
    ).strip()

    context = f"window {window_idx} in {filename}"
    segments = _parse_transcription(raw, context)

    # Shift all timestamps from relative-to-window to absolute file time
    for seg in segments:
        seg["start"] = round(seg.get("start", 0.0) + offset_s, 3)
        seg["end"] = round(seg.get("end", 0.0) + offset_s, 3)

    return segments


def _build_label_map(prev_segs: list[dict], next_segs: list[dict]) -> dict[str, str]:
    """Map next-window speaker labels → prev-window speaker labels.

    For each (next_speaker, prev_speaker) pair, accumulates the total temporal
    overlap of their segments in the boundary zone.  The pair with the greatest
    overlap wins; greedy assignment ensures each label is used at most once.
    Unmapped labels (no overlap found) are not included — callers pass them
    through unchanged.
    """
    overlap_scores: dict[tuple[str, str], float] = {}
    for p in prev_segs:
        for n in next_segs:
            overlap = max(0.0, min(p["end"], n["end"]) - max(p["start"], n["start"]))
            if overlap > 0.0:
                key = (n["speaker"], p["speaker"])
                overlap_scores[key] = overlap_scores.get(key, 0.0) + overlap

    if not overlap_scores:
        return {}

    label_map: dict[str, str] = {}
    used_prev: set[str] = set()
    used_next: set[str] = set()

    for (next_sp, prev_sp), _ in sorted(overlap_scores.items(), key=lambda x: -x[1]):
        if next_sp in used_next or prev_sp in used_prev:
            continue
        label_map[next_sp] = prev_sp
        used_next.add(next_sp)
        used_prev.add(prev_sp)

    return label_map


def _stitch_speaker_labels(
    window_segs: list[list[dict]], overlap_s: float
) -> list[dict]:
    """Reconcile speaker labels across window boundaries and return a flat list.

    For each adjacent window pair (i, i+1), selects segments within overlap_s
    of the boundary on each side, builds a speaker label mapping via
    _build_label_map, and remaps window i+1's labels before merging.
    Labels with no mapping pass through unchanged.  Logs the total number of
    label remappings applied across all boundaries.
    """
    if not window_segs:
        return []
    if len(window_segs) == 1:
        return list(window_segs[0])

    total_remaps = 0
    result_windows: list[list[dict]] = [list(window_segs[0])]

    for i in range(1, len(window_segs)):
        prev_all = result_windows[i - 1]
        curr_segs = list(window_segs[i])

        if not prev_all or not curr_segs:
            result_windows.append(curr_segs)
            continue

        boundary = curr_segs[0]["start"]
        prev_zone = [s for s in prev_all if s["start"] >= boundary - overlap_s]
        next_zone = [s for s in curr_segs if s["start"] < boundary + overlap_s]

        label_map = _build_label_map(prev_zone, next_zone)

        remapped: list[dict] = []
        for seg in curr_segs:
            new_seg = dict(seg)
            new_sp = label_map.get(seg["speaker"], seg["speaker"])
            if new_sp != seg["speaker"]:
                total_remaps += 1
            new_seg["speaker"] = new_sp
            remapped.append(new_seg)

        result_windows.append(remapped)

    logger.info(
        f"chunk_audio: speaker stitching applied {total_remaps} label remappings "
        f"across {len(window_segs) - 1} window boundaries"
    )

    flat = [seg for window in result_windows for seg in window]
    flat.sort(key=lambda s: s["start"])
    return flat


def _transcribe(filepath: str) -> list[dict]:
    """Pre-split audio into overlapping windows, transcribe each, merge segments.

    Each window is _TRANSCRIPTION_SEGMENT_S seconds of core audio with
    _TRANSCRIPTION_OVERLAP_S of overlap on each side (clamped at file
    boundaries).  The overlap ensures sentences that straddle a split point
    are fully transcribed by at least one window.

    After transcription, timestamps are converted to absolute file time.
    Deduplication: only segments whose start time falls inside the core
    region are kept, so overlapping regions never produce duplicate segments.
    Speaker labels are then stitched across windows via _stitch_speaker_labels.
    """
    audio = AudioSegment.from_file(filepath)
    total_ms = len(audio)
    filename = Path(filepath).name

    segment_ms = int(_TRANSCRIPTION_SEGMENT_S * 1000)
    overlap_ms = int(_TRANSCRIPTION_OVERLAP_S * 1000)

    # Pre-compute all window boundaries so we can submit them concurrently.
    WindowParams = tuple[int, int, int, int, float, float]  # typed alias for clarity
    windows: list[WindowParams] = []
    core_start_ms = 0
    while core_start_ms < total_ms:
        core_end_ms = min(core_start_ms + segment_ms, total_ms)
        window_start_ms = max(0, core_start_ms - overlap_ms)
        window_end_ms = min(total_ms, core_end_ms + overlap_ms)
        core_start_s = core_start_ms / 1000.0
        core_end_s = core_end_ms / 1000.0
        offset_s = window_start_ms / 1000.0
        windows.append((core_start_ms, core_end_ms, window_start_ms, window_end_ms, core_start_s, core_end_s, offset_s))
        core_start_ms = core_end_ms

    def _transcribe_one(window_idx: int) -> list[dict]:
        """Transcribe a single window with retry and return its core segments."""
        _, _, window_start_ms, window_end_ms, core_start_s, core_end_s, offset_s = windows[window_idx]
        logger.info(
            f"chunk_audio: transcribing window {window_idx} "
            f"core=[{core_start_s:.0f}s – {core_end_s:.0f}s] "
            f"window=[{offset_s:.0f}s – {window_end_ms / 1000:.0f}s] — {filename}"
        )
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(1, _TRANSCRIPTION_MAX_RETRIES + 1):
            try:
                segments = _transcribe_window(
                    audio, window_start_ms, window_end_ms, offset_s, window_idx, filename
                )
                inverted = [s for s in segments if s.get("end", 0.0) <= s.get("start", 0.0)]
                if inverted:
                    raise ValueError(
                        f"chunk_audio: window {window_idx} contains {len(inverted)} "
                        f"segment(s) with inverted timestamps (e.g. start="
                        f"{inverted[0]['start']} end={inverted[0]['end']})"
                    )
                break
            except ValueError as exc:
                last_exc = exc
                if attempt < _TRANSCRIPTION_MAX_RETRIES:
                    logger.warning(
                        f"chunk_audio: window {window_idx} attempt {attempt} failed "
                        f"({exc}); retrying…"
                    )
        else:
            raise last_exc

        # Keep only segments that start inside the core region.
        core_segs = [s for s in segments if core_start_s <= s["start"] < core_end_s]
        logger.info(
            f"chunk_audio: window {window_idx} → {len(segments)} raw segments, "
            f"{len(core_segs)} kept after deduplication"
        )
        return core_segs

    n_workers = min(len(windows), _TRANSCRIPTION_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all windows; store futures keyed by index to preserve order.
        future_map = {executor.submit(_transcribe_one, i): i for i in range(len(windows))}
        results: dict[int, list[dict]] = {}
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()  # re-raises any exception from the thread

    window_segs = [results[i] for i in range(len(windows))]

    return _stitch_speaker_labels(window_segs, _TRANSCRIPTION_OVERLAP_S)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Semantic + speaker boundary detection
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _detect_boundaries(segments: list[dict]) -> set[int]:
    """Return the set of segment indices where a new chunk must begin.

    Split points come from two sources (union of both):
    - Speaker turn: speaker label changes between adjacent segments.
    - Semantic shift: cosine similarity of embed_text() < _SEMANTIC_THRESHOLD.
    These are throwaway embeddings used only for boundary detection.
    """
    split_indices: set[int] = set()
    if len(segments) < 2:
        return split_indices

    # Embed every segment text; keep a parallel list aligned with segments
    embeddings: list[Optional[list[float]]] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            try:
                emb = embed_text(text)
            except Exception as exc:
                logger.warning(
                    f"chunk_audio: embed_text failed for segment text "
                    f"{text[:40]!r}: {exc} — treating as no embedding"
                )
                emb = None
        else:
            emb = None
        embeddings.append(emb)

    for i in range(1, len(segments)):
        prev_speaker = segments[i - 1].get("speaker", "")
        curr_speaker = segments[i].get("speaker", "")

        # Speaker-turn boundary always wins
        if prev_speaker != curr_speaker:
            split_indices.add(i)
            continue

        # Semantic boundary (only when both embeddings are available)
        a = embeddings[i - 1]
        b = embeddings[i]
        if a is not None and b is not None:
            sim = _cosine_similarity(a, b)
            if sim < _SEMANTIC_THRESHOLD:
                split_indices.add(i)

    return split_indices


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Grouping, merging, ceiling enforcement
# ──────────────────────────────────────────────────────────────────────────────

def _group_segments(
    segments: list[dict], split_indices: set[int]
) -> list[list[dict]]:
    """Partition segments into groups at every split index."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for i, seg in enumerate(segments):
        if i in split_indices and current:
            groups.append(current)
            current = []
        current.append(seg)
    if current:
        groups.append(current)
    return groups


def _group_duration(group: list[dict]) -> float:
    """Return wall-clock duration of a group in seconds."""
    if not group:
        return 0.0
    return group[-1].get("end", 0.0) - group[0].get("start", 0.0)


def _merge_short_groups(groups: list[list[dict]]) -> list[list[dict]]:
    """Merge consecutive same-speaker groups when combined duration < _SHORT_MERGE_S."""
    if not groups:
        return groups

    merged: list[list[dict]] = [groups[0]]

    for group in groups[1:]:
        prev = merged[-1]
        prev_speaker = prev[0].get("speaker") if prev else None
        curr_speaker = group[0].get("speaker") if group else None

        combined_duration = (
            group[-1].get("end", 0.0) - prev[0].get("start", 0.0)
            if prev and group
            else 0.0
        )

        if prev_speaker == curr_speaker and combined_duration < _SHORT_MERGE_S:
            merged[-1] = prev + group
        else:
            merged.append(group)

    return merged


def _enforce_ceiling(
    groups: list[list[dict]],
) -> list[tuple[float, float, list[dict], bool]]:
    """Force-split any group exceeding _HARD_CEILING_S.

    Splits strictly by absolute time windows so that a single coarse segment
    (e.g. Gemini returning one 534 s block) is still broken into ≤75 s clips.
    Applies _OVERLAP_S backward overlap at forced split points.

    Returns (start_s, end_s, overlapping_segs, forced_split) tuples.
    start_s / end_s define the audio clip boundaries.
    overlapping_segs are all segments whose time range overlaps the window
    — used only for transcript text and speaker extraction.
    forced_split=True for every sub-chunk produced by a ceiling split.
    """
    result: list[tuple[float, float, list[dict], bool]] = []

    for group in groups:
        group_start = group[0].get("start", 0.0)
        group_end = group[-1].get("end", 0.0)

        if group_end - group_start <= _HARD_CEILING_S:
            result.append((group_start, group_end, group, False))
            continue

        # Walk the group in _HARD_CEILING_S-wide time windows
        window_start = group_start
        while window_start < group_end:
            window_end = min(window_start + _HARD_CEILING_S, group_end)

            # All segments that overlap [window_start, window_end]
            overlapping = [
                seg for seg in group
                if seg.get("end", 0.0) > window_start
                and seg.get("start", 0.0) < window_end
            ]

            result.append((window_start, window_end, overlapping or group, True))

            if window_end >= group_end:
                break

            # Backward overlap for the next window
            window_start = window_end - _OVERLAP_S

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Audio clip extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_clip_bytes(filepath: str, start_ms: int, end_ms: int) -> bytes:
    """Slice [start_ms, end_ms) from the audio file and return MP3 bytes."""
    audio = AudioSegment.from_file(filepath)
    clip = audio[start_ms:end_ms]
    buf = io.BytesIO()
    clip.export(buf, format="mp3")
    return buf.getvalue()


def _dominant_speaker(group: list[dict]) -> Optional[str]:
    """Return the speaker label when all segments share one; else None."""
    speakers = {seg.get("speaker") for seg in group if seg.get("speaker")}
    if len(speakers) == 1:
        return speakers.pop()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def chunk_audio(filepath: str) -> list[dict]:
    """Speaker-Aligned Semantic Chunking for an audio file.

    Step 1 — Transcribe with Gemini Flash (timestamps + speaker labels).
    Step 2 — Detect semantic and speaker-turn boundaries via embed_text().
    Step 3 — Group segments, merge short same-speaker groups, enforce 75 s ceiling.
    Step 4 — Extract MP3 clips via pydub.
    Step 5 — Build and return chunk dicts.
    Does not call embed_chunks or add_chunks.
    """
    filename = Path(filepath).name
    logger.info(f"chunk_audio: starting transcription for {filename}")

    # Step 1
    segments = _transcribe(filepath)  # raises ValueError on failure

    if not segments:
        raise ValueError(
            f"chunk_audio: transcription returned no segments for {filename}"
        )

    logger.info(
        f"chunk_audio: received {len(segments)} segments from Gemini for {filename}"
    )

    # Step 2
    split_indices = _detect_boundaries(segments)
    logger.info(
        f"chunk_audio: detected {len(split_indices)} boundary split points for {filename}"
    )

    # Step 3
    groups = _group_segments(segments, split_indices)
    groups = _merge_short_groups(groups)
    logger.info(
        f"chunk_audio: {len(groups)} groups after merging short same-speaker groups"
    )
    final_groups = _enforce_ceiling(groups)
    logger.info(
        f"chunk_audio: {len(final_groups)} final chunks after 75 s ceiling enforcement"
    )

    # Steps 4–5
    chunks: list[dict] = []
    chunk_index = 0

    for start_s, end_s, group_segs, forced_split in final_groups:
        if not group_segs:
            continue

        if end_s <= start_s:
            logger.warning(
                f"chunk_audio: skipping degenerate chunk [{start_s}s – {end_s}s] "
                f"in {filename}"
            )
            continue

        transcript_text: str = " ".join(
            seg.get("text", "").strip()
            for seg in group_segs
            if seg.get("text", "").strip()
        )

        speaker_id: Optional[str] = _dominant_speaker(group_segs)

        try:
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            audio_bytes = _extract_clip_bytes(filepath, start_ms, end_ms)
        except Exception as exc:
            logger.warning(
                f"chunk_audio: clip extraction failed for chunk {chunk_index} "
                f"[{start_s:.2f}s – {end_s:.2f}s] in {filename}: {exc} — skipping"
            )
            continue

        logger.info(
            f"chunk_audio: chunk {chunk_index} [{start_s:.2f}s – {end_s:.2f}s] "
            f"speaker={speaker_id!r} forced={forced_split} — {filename}"
        )

        chunks.append(
            {
                "type": "audio",
                "audio_bytes": audio_bytes,
                "text": transcript_text,
                "source": filepath,
                "page": 0,
                "chunk_index": chunk_index,
                "modality": "audio_clip",
                "start_time_seconds": start_s,
                "end_time_seconds": end_s,
                "speaker_id": speaker_id,
                "transcript_text": transcript_text,
                "forced_split": forced_split,
            }
        )
        chunk_index += 1

    logger.info(f"chunk_audio: produced {len(chunks)} chunks from {filename}")
    return chunks
