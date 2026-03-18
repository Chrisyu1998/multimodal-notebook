"""
Chunking package — converts raw files into chunk dicts for embedding.

Re-exports all chunkers so callers can use:
    from backend.services import chunking
    chunking.chunk_pdf(...)
    chunking.chunk_image(...)
"""

from backend.services.chunking.chunk_image import chunk_image
from backend.services.chunking.chunk_pdf import chunk_pdf
from backend.services.chunking.chunk_video import chunk_video

__all__ = ["chunk_pdf", "chunk_image", "chunk_video"]
