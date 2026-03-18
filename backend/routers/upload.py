"""
Upload router — POST /upload
Accepts a multipart file, validates it, saves it to disk, and runs the full
ingestion pipeline (chunking → GCS upload → embedding → indexing).
"""

import hashlib
import uuid
from pathlib import Path

import ffmpeg
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

import backend.config as config
from backend.models.schemas import UploadResponse
from backend.services import bm25_index, chunking, embeddings, gcs, vectorstore
from backend.services.embeddings import EmbeddingBatchError
from backend.services.vectorstore import VectorStoreUnavailableError

router = APIRouter()

_MAX_BYTES: int = config.MAX_FILE_SIZE_MB * 1024 * 1024


def _upload_chunk_media(chunk: dict, file_hash: str) -> None:
    """Upload a chunk's media bytes to GCS and stamp gcs_uri on it in-place.

    Called after chunking, before embedding.  The large byte fields
    (image_bytes, video_bytes) are only available in this window — they are
    stripped before ChromaDB / BM25 storage.

    Modality mapping:
      image_global / image_local → upload image bytes; stamp gcs_uri.
      video_clip → extract the mid-point JPEG frame from the MP4 clip bytes
                   (cheap) and upload that; stamp gcs_uri with the frame URI
                   so the reranker downloads a small JPEG instead of the
                   full video.
      everything else (pdf text, video_summary) → no GCS upload needed.
    """
    modality: str = chunk.get("modality", "")
    idx: int = chunk.get("chunk_index", 0)

    if modality in ("image_global", "image_local"):
        mime: str = chunk.get("mime_type", "image/jpeg")
        ext = "png" if mime == "image/png" else "jpg"
        uri = gcs.upload_bytes(
            chunk["image_bytes"],
            f"media/{file_hash}/img_{idx}.{ext}",
            mime,
        )
        chunk["gcs_uri"] = uri

    elif modality == "video_clip":
        video_bytes: bytes = chunk["video_bytes"]
        try:
            # Grab the first decodable frame from the piped MP4 bytes.
            # Seeking (ss=) inside a fragmented MP4 pipe is unreliable because
            # ffmpeg cannot seek backwards in a stream; taking frame 0 is always
            # safe and sufficient for a short (≤120s) clip.
            frame_bytes, _ = (
                ffmpeg
                .input("pipe:", format="mp4")
                .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
                .run(input=video_bytes, capture_stdout=True, capture_stderr=True)
            )
            uri = gcs.upload_bytes(
                frame_bytes,
                f"media/{file_hash}/video_frame_{idx}.jpg",
                "image/jpeg",
            )
            chunk["gcs_uri"] = uri
        except Exception as exc:
            logger.warning(f"Frame extraction failed for video chunk {idx}: {exc}")
            chunk["gcs_uri"] = ""

    else:
        # PDF text chunks and video_summary chunks need no GCS upload.
        chunk["gcs_uri"] = ""


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Validate and ingest an uploaded file.
    Supported extensions: .pdf, .png, .jpg, .jpeg, .webp, .mp4, .mov
    Max size: config.MAX_FILE_SIZE_MB (default 200 MB).
    """
    logger.info(f"Upload received: {file.filename!r}")

    # ---- 1. Validate extension ----
    suffix = Path(file.filename).suffix.lower()
    if suffix not in config.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Allowed types: {sorted(config.ALLOWED_FILE_TYPES)}"
            ),
        )

    # ---- 2. Read bytes + validate size ----
    contents = await file.read()
    size_bytes = len(contents)
    if size_bytes > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {size_bytes / 1_048_576:.1f} MB exceeds the "
                f"{config.MAX_FILE_SIZE_MB} MB limit."
            ),
        )

    # ---- 3. Dedup check ----
    file_hash = hashlib.sha256(contents).hexdigest()
    try:
        already_indexed = vectorstore.is_file_indexed(file_hash)
    except VectorStoreUnavailableError as exc:
        logger.error(f"Vector store unavailable during dedup check: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc

    if already_indexed:
        logger.info(f"File {file.filename!r} already indexed (hash {file_hash[:8]}…) — skipping.")
        return UploadResponse(
            file_id=file_hash[:8],
            filename=file.filename,
            size_bytes=size_bytes,
            status="already_indexed",
            num_chunks=0,
        )

    # ---- 4. Save to ./tmp/uploads/{uuid}_{original_filename} ----
    uploads_dir = Path(config.TMP_UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    dest = uploads_dir / f"{file_id}_{file.filename}"
    dest.write_bytes(contents)
    logger.info(f"Saved {size_bytes} bytes → {dest}")

    # ---- 5. Chunk ----
    try:
        if suffix == ".pdf":
            chunks = chunking.chunk_pdf(str(dest))
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            chunks = chunking.chunk_image(str(dest))
        else:  # .mp4, .mov
            chunks = chunking.chunk_video(str(dest))
    except ValueError as exc:
        logger.error(f"Chunking failed for {file.filename!r}: {exc}")
        raise HTTPException(
            status_code=422,
            detail="Could not parse this file. Please try a different PDF.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected chunking error for {file.filename!r}: {exc}")
        raise HTTPException(
            status_code=422,
            detail="Could not parse this file. Please try a different PDF.",
        ) from exc

    # Stamp original filename and content hash onto every chunk.
    for chunk in chunks:
        chunk["source"] = file.filename
        chunk["file_hash"] = file_hash

    # ---- 5b. Upload media bytes to GCS; stamp gcs_uri on each chunk ----
    # Must happen before embed_chunks because embedding strips the byte fields.
    for chunk in chunks:
        _upload_chunk_media(chunk, file_hash)

    # ---- 6. Embed ----
    try:
        chunks = embeddings.embed_chunks(chunks)
    except EmbeddingBatchError as exc:
        logger.error(f"Embedding failed for {file.filename!r}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable. Check your API key.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected embedding error for {file.filename!r}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable. Check your API key.",
        ) from exc

    # ---- 7. Index ----
    try:
        vectorstore.add_chunks(chunks)
    except VectorStoreUnavailableError as exc:
        logger.error(f"Vector store unavailable while indexing {file.filename!r}: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc
    except Exception as exc:
        logger.error(f"Unexpected vectorstore error for {file.filename!r}: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc

    # ---- 8. BM25 index ----
    bm25_index.build_index(chunks)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=size_bytes,
        status="indexed",
        num_chunks=len(chunks),
    )
