"""
Video chunking — splits a video file into 128-second segments.

Each segment is returned as a chunk with raw video bytes for
Gemini Embedding 2 (video/mp4, one per API request).

Chunk shape:
    type="video", video_bytes, text, source, page, chunk_index,
    modality="video"
"""

from loguru import logger


def chunk_video(filepath: str) -> list[dict]:
    """
    Split a video into 128-second segments and return chunk dicts.
    Each chunk contains raw video bytes ready for Gemini Embedding 2.
    """
    logger.debug(f"chunk_video called: {filepath}")
    # TODO: split video into 128s segments via ffmpeg or moviepy
    # TODO: return list of chunk dicts with type="video"
    raise NotImplementedError
