"""
Video chunking — Semantic Scene-Based Indexing for a video file.

Implements Dual-Stream Embedding per detected scene:
- Chunk A: native video clip bytes (for Gemini Embedding 2 video embedding)
- Chunk B: visual summary text wrapped as a single-page PDF (for BM25 + text embedding)

Chunk A shape:
    type="video", video_bytes, text, source, page, chunk_index,
    modality="video_clip", parent_scene_id, start_time_seconds,
    end_time_seconds, scene_index, forced_split

Chunk B shape:
    type="document", pdf_bytes, text, source, page, chunk_index,
    modality="video_summary", parent_scene_id, start_time_seconds,
    end_time_seconds, scene_index
"""

import hashlib
from pathlib import Path

import ffmpeg
from google import genai
from google.genai import types as genai_types
from loguru import logger
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

import backend.config as config
from backend.ingestion.utils import text_to_pdf

_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

_MAX_SCENE_DURATION: float = config.VIDEO_MAX_SCENE_DURATION
_FORCED_SPLIT_OVERLAP: float = config.VIDEO_FORCED_SPLIT_OVERLAP
_VISUAL_SUMMARY_PROMPT: str = (
    "First, transcribe or closely paraphrase the key things said "
    "(speech, narration, dialogue). Then describe what is visually happening. "
    "Include tone, emphasis, and any specific terms, names, or numbers mentioned."
)


def _detect_scenes(filepath: str) -> list[tuple[float, float]]:
    """
    Run PySceneDetect on *filepath* and return (start_s, end_s) pairs.

    Uses ContentDetector to find natural scene boundaries.
    """
    video = open_video(filepath)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # Treat the entire file as one scene
        duration = video.duration.get_seconds() if video.duration else 0.0
        logger.warning(
            f"chunk_video: no scene boundaries detected in {Path(filepath).name}, "
            f"treating as one scene ({duration:.1f}s)"
        )
        return [(0.0, duration)] if duration > 0 else []

    return [
        (start_tc.get_seconds(), end_tc.get_seconds())
        for start_tc, end_tc in scene_list
    ]


def _split_long_scenes(
    raw_scenes: list[tuple[float, float]],
) -> list[tuple[float, float, bool]]:
    """
    Force-split any scene that exceeds _MAX_SCENE_DURATION.

    Returns (start_s, end_s, forced_split) tuples.
    forced_split=True for every sub-segment produced by a forced 120s split.
    Natural scene boundaries get forced_split=False.
    A 5s overlap is applied between sub-segments at forced split points.
    """
    result: list[tuple[float, float, bool]] = []
    for start_s, end_s in raw_scenes:
        duration = end_s - start_s
        if duration <= _MAX_SCENE_DURATION:
            result.append((start_s, end_s, False))
            continue

        # Force-split at _MAX_SCENE_DURATION intervals with overlap
        seg_start = start_s
        while seg_start < end_s:
            seg_end = min(seg_start + _MAX_SCENE_DURATION, end_s)
            result.append((seg_start, seg_end, True))
            if seg_end >= end_s:
                break
            # Apply 5s backward overlap for the next sub-segment
            seg_start = seg_end - _FORCED_SPLIT_OVERLAP

    return result


def _extract_clip_bytes(filepath: str, start_s: float, end_s: float) -> bytes:
    """
    Extract a raw MP4 clip from *filepath* between start_s and end_s.

    Uses stream copy (no re-encoding). Returns raw MP4 bytes.
    """
    duration = end_s - start_s
    out, _ = (
        ffmpeg
        .input(filepath, ss=start_s)
        .output(
            "pipe:",
            t=duration,
            format="mp4",
            vcodec="copy",
            acodec="copy",
            movflags="frag_keyframe+empty_moov",
        )
        .run(capture_stdout=True, capture_stderr=True)
    )
    return out


def _generate_visual_summary(video_bytes: bytes) -> str:
    """
    Send a video clip to Gemini Flash and return a transcript + visual description.

    Gemini processes both the audio track (speech) and video frames, so the
    summary captures what was said, not just what was seen.
    Returns empty string on failure.
    """
    try:
        video_part = genai_types.Part.from_bytes(
            data=video_bytes, mime_type="video/mp4"
        )
        response = _gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[video_part, _VISUAL_SUMMARY_PROMPT],
        )
        text = "".join(
            part.text
            for part in response.candidates[0].content.parts
            if hasattr(part, "text") and part.text
        ).strip()
        return text
    except Exception as exc:
        logger.warning(f"chunk_video: visual summary generation failed: {exc}")
        return ""


def chunk_video(filepath: str) -> list[dict]:
    """
    Semantic Scene-Based Indexing: detect scenes, extract clips, generate visual
    summaries, and return two chunk dicts per scene (Chunk A + Chunk B).

    Step 1 — detect scene boundaries via PySceneDetect.
    Step 2 — force-split scenes > 120s with 5s overlap at split points.
    Step 3 — for each scene: extract raw MP4 clip via ffmpeg.
    Step 4 — send clip to Gemini Flash for speech transcription + visual summary.
    Step 5 — build Chunk A (video clip) and Chunk B (summary PDF).
    Returns a flat list; does not call embed_chunks or add_chunks.
    """
    filename = Path(filepath).name
    logger.info(f"chunk_video: starting scene detection for {filename}")

    # ── Step 1 & 2: scene detection + force-split ─────────────────────────────
    try:
        raw_scenes = _detect_scenes(filepath)
    except Exception as exc:
        logger.warning(f"chunk_video: scene detection failed for {filename}: {exc}")
        return []

    if not raw_scenes:
        logger.warning(f"chunk_video: no scenes found in {filename}")
        return []

    scenes = _split_long_scenes(raw_scenes)
    logger.info(
        f"chunk_video: {len(raw_scenes)} raw scenes → {len(scenes)} segments for {filename}"
    )

    # ── Steps 3–5: per-scene extraction and chunk building ────────────────────
    chunks: list[dict] = []
    chunk_index = 0

    for scene_index, (start_s, end_s, forced_split) in enumerate(scenes):
        logger.info(
            f"chunk_video: scene {scene_index} [{start_s:.2f}s – {end_s:.2f}s] "
            f"forced={forced_split} — {filename}"
        )
        try:
            # Step 3a: extract raw MP4 clip
            video_bytes = _extract_clip_bytes(filepath, start_s, end_s)
        except Exception as exc:
            logger.warning(
                f"chunk_video: clip extraction failed for scene {scene_index} "
                f"in {filename}: {exc} — skipping"
            )
            continue

        parent_scene_id = hashlib.sha256(video_bytes).hexdigest()

        # Step 4: generate transcript + visual summary from the clip
        visual_summary_text = _generate_visual_summary(video_bytes)

        fallback_text = f"Video: {filename} scene {scene_index}"
        chunk_a_text = visual_summary_text or fallback_text

        # Step 5a: Chunk A — native video clip
        chunks.append(
            {
                "type": "video",
                "video_bytes": video_bytes,
                "text": chunk_a_text,
                "source": filepath,
                "page": 0,
                "chunk_index": chunk_index,
                "modality": "video_clip",
                "parent_scene_id": parent_scene_id,
                "start_time_seconds": start_s,
                "end_time_seconds": end_s,
                "scene_index": scene_index,
                "forced_split": forced_split,
            }
        )
        chunk_index += 1

        # Step 5b: Chunk B — visual summary PDF (skip if no summary)
        if visual_summary_text:
            try:
                pdf_bytes = text_to_pdf(visual_summary_text)
            except Exception as exc:
                logger.warning(
                    f"chunk_video: PDF render failed for scene {scene_index} "
                    f"in {filename}: {exc} — skipping Chunk B"
                )
                continue

            chunks.append(
                {
                    "type": "document",
                    "pdf_bytes": pdf_bytes,
                    "text": visual_summary_text,
                    "source": filepath,
                    "page": 0,
                    "chunk_index": chunk_index,
                    "modality": "video_summary",
                    "parent_scene_id": parent_scene_id,
                    "start_time_seconds": start_s,
                    "end_time_seconds": end_s,
                    "scene_index": scene_index,
                }
            )
            chunk_index += 1

    logger.info(
        f"chunk_video: produced {len(chunks)} chunks from {filename}"
    )
    return chunks
