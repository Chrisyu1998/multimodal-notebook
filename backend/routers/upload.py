"""
Upload router — POST /upload
Accepts a multipart file, validates it, saves it to disk, and returns
a receipt. Full ingestion pipeline (chunking → embedding → indexing)
is wired in here once the service layer is implemented.
"""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

import backend.config as config
from backend.models.schemas import UploadResponse
from backend.services import chunking, embeddings, vectorstore

# TODO: from backend.services import bm25_index (not yet implemented)

router = APIRouter()

_MAX_BYTES: int = config.MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Validate and save an uploaded file to ./tmp/uploads/.
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

    # ---- 3. Dedup check — skip ingestion if file content already indexed ----
    file_hash = hashlib.sha256(contents).hexdigest()
    if vectorstore.is_file_indexed(file_hash):
        logger.info(f"File {file.filename!r} already indexed (hash {file_hash[:8]}…) — skipping.")
        return UploadResponse(
            file_id=file_hash[:8],
            filename=file.filename,
            size_bytes=size_bytes,
            status="already_indexed",
        )

    # ---- 4. Save to ./tmp/uploads/{uuid}_{original_filename} ----
    uploads_dir = Path(config.TMP_UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    dest = uploads_dir / f"{file_id}_{file.filename}"
    dest.write_bytes(contents)
    logger.info(f"Saved {size_bytes} bytes → {dest}")

    try:
        if suffix == ".pdf":
            chunks = chunking.chunk_pdf(str(dest))
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            chunks = chunking.chunk_image(str(dest))
        else:  # .mp4, .mov
            chunks = chunking.chunk_video(str(dest))

        # Stamp original filename and content hash onto every chunk so
        # vectorstore IDs are content-derived, not path-derived.
        for chunk in chunks:
            chunk["source"] = file.filename
            chunk["file_hash"] = file_hash

        chunks = embeddings.embed_chunks(chunks)
        vectorstore.add_chunks(chunks)
        # TODO (Week 1): bm25_index.build_index(chunks)

        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            size_bytes=size_bytes,
            status="indexed",
        )
    except Exception as exc:
        logger.error(f"Ingestion failed for {file.filename!r}: {exc}")
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            size_bytes=size_bytes,
            status="upload_failed_to_index",
        )
