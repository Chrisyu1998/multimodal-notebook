"""
Audio chunking — passes an audio file as a single chunk.

Gemini Embedding 2 accepts audio directly (max 80s per file).
Files within the limit are passed whole; longer files will need
to be split into segments (not yet implemented).

Chunk shape:
    type="audio", audio_bytes, text, source, page, chunk_index,
    modality="audio"
"""

from loguru import logger


def chunk_audio(filepath: str) -> list[dict]:
    """
    Wrap an audio file as one or more chunk dicts ready for Gemini Embedding 2.
    """
    logger.debug(f"chunk_audio called: {filepath}")
    # TODO: validate format (mp3, wav, etc.)
    # TODO: split into <=80s segments if needed
    # TODO: return list of chunk dicts with type="audio"
    raise NotImplementedError
