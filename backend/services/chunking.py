"""
Chunking service — converts raw files into chunk dicts.

Each chunk dict has the shape:
    {
        "chunk_id": str,      # "<sha256>_<index>"
        "text": str,          # extracted/described text
        "source": str,        # original filename or GCS URI
        "chunk_type": str,    # "pdf" | "image" | "video_frame"
        "metadata": dict,     # page number, timestamp, etc.
    }

Rules:
- PDF: PyMuPDF (fitz) — 512-token chunks, 64-token overlap
- Image: passed whole; Gemini Embedding 2 handles natively (no text extraction here)
- Video: PySceneDetect extracts keyframes; each frame becomes one chunk
"""

from loguru import logger

import backend.config as config


def chunk_pdf(filepath: str) -> list[dict]:
    """
    Extract and chunk text from a PDF file.
    Uses PyMuPDF for parsing and a sliding window of
    config.CHUNK_SIZE tokens with config.CHUNK_OVERLAP overlap.
    """
    logger.debug(f"chunk_pdf called: {filepath}")
    # TODO: open PDF with fitz.open(filepath)
    # TODO: extract text page by page
    # TODO: split into token-bounded chunks (512 tokens, 64 overlap)
    #       Use LangChain RecursiveCharacterTextSplitter ONLY here
    # TODO: compute sha256 of file bytes for chunk_id prefix
    # TODO: return list of chunk dicts with page metadata
    raise NotImplementedError


def chunk_image(filepath: str) -> list[dict]:
    """
    Wrap a single image file as one chunk dict.
    The raw image bytes are stored so embeddings.embed_chunks
    can pass them directly to the Gemini multimodal embedding API.
    """
    logger.debug(f"chunk_image called: {filepath}")
    # TODO: read image bytes with Pillow to validate it's a real image
    # TODO: compute sha256 of file bytes for chunk_id
    # TODO: return single-element list with chunk_type="image"
    raise NotImplementedError


def chunk_video(filepath: str) -> list[dict]:
    """
    Extract keyframes from a video using PySceneDetect.
    Each keyframe becomes one chunk with a timestamp in its metadata.
    """
    logger.debug(f"chunk_video called: {filepath}")
    # TODO: run PySceneDetect scene detection on filepath
    # TODO: extract one representative frame per scene
    # TODO: save frames to config.TMP_UPLOAD_DIR
    # TODO: return list of chunk dicts with chunk_type="video_frame"
    raise NotImplementedError
