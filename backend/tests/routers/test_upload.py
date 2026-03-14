"""
Tests for POST /upload — orchestration logic in backend.routers.upload.

All three services (chunking, embeddings, vectorstore) are mocked so
these tests run without any real files, Gemini calls, or ChromaDB I/O.
The file is always written to a real temp dir (patched via TMP_UPLOADS_DIR)
so we verify the disk-write path too without polluting the project tree.
"""

import os
import tempfile
from io import BytesIO
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

from backend.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CHUNKS = [{"text": "chunk text", "source": "/tmp/f.pdf", "page": 1, "chunk_index": 0}]
_FAKE_EMBEDDED = [{"embedding": [0.1, 0.2], **_FAKE_CHUNKS[0]}]


def _upload(filename: str, content: bytes = b"fake content") -> dict:
    """POST /upload with an in-memory file and return the parsed JSON."""
    return client.post(
        "/upload/",
        files={"file": (filename, BytesIO(content), "application/octet-stream")},
    )


def _patch_services(
    chunk_fn: str = "chunk_pdf",
    chunk_return=None,
    embed_raises: Optional[Exception] = None,
    add_raises: Optional[Exception] = None,
):
    """
    Context manager that patches all three service functions at once.

    chunk_fn     — which chunking function to mock ("chunk_pdf", "chunk_image", "chunk_video")
    chunk_return — return value of the chunker (defaults to _FAKE_CHUNKS)
    embed_raises — if set, embed_chunks raises this exception
    add_raises   — if set, add_chunks raises this exception
    """
    chunks_out = chunk_return if chunk_return is not None else _FAKE_CHUNKS

    mock_chunker = MagicMock(return_value=chunks_out)
    if embed_raises:
        mock_embed = MagicMock(side_effect=embed_raises)
    else:
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDED)

    if add_raises:
        mock_add = MagicMock(side_effect=add_raises)
    else:
        mock_add = MagicMock()

    return (
        patch(f"backend.routers.upload.chunking.{chunk_fn}", mock_chunker),
        patch("backend.routers.upload.embeddings.embed_chunks", mock_embed),
        patch("backend.routers.upload.vectorstore.add_chunks", mock_add),
        mock_chunker,
        mock_embed,
        mock_add,
    )


# ---------------------------------------------------------------------------
# Happy path — PDF
# ---------------------------------------------------------------------------

class TestPdfIngestion:
    def test_returns_200(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            resp = _upload("report.pdf")
        assert resp.status_code == 200

    def test_status_is_indexed(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload("report.pdf").json()
        assert body["status"] == "indexed"

    def test_response_fields_present(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload("report.pdf", b"pdfbytes").json()
        assert body["filename"] == "report.pdf"
        assert body["size_bytes"] == len(b"pdfbytes")
        assert "file_id" in body

    def test_pipeline_call_order(self, tmp_path):
        """chunk_pdf → embed_chunks → add_chunks must be called in that order."""
        call_order = []
        mock_chunk = MagicMock(side_effect=lambda *a, **kw: call_order.append("chunk") or _FAKE_CHUNKS)
        mock_embed = MagicMock(side_effect=lambda *a, **kw: call_order.append("embed") or _FAKE_EMBEDDED)
        mock_add   = MagicMock(side_effect=lambda *a, **kw: call_order.append("add"))

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf")

        assert call_order == ["chunk", "embed", "add"]

    def test_embed_receives_chunker_output(self, tmp_path):
        """embed_chunks must be called with the list returned by chunk_pdf."""
        mock_chunk = MagicMock(return_value=_FAKE_CHUNKS)
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDED)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("report.pdf")

        mock_embed.assert_called_once_with(_FAKE_CHUNKS)

    def test_add_receives_embedded_output(self, tmp_path):
        """add_chunks must be called with the list returned by embed_chunks."""
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDED)
        mock_add   = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf")

        mock_add.assert_called_once_with(_FAKE_EMBEDDED)


# ---------------------------------------------------------------------------
# Chunker dispatch — correct function called per file type
# ---------------------------------------------------------------------------

class TestChunkerDispatch:
    @pytest.mark.parametrize("filename", ["photo.png", "scan.jpg", "diagram.jpeg", "logo.webp"])
    def test_image_extensions_call_chunk_image(self, tmp_path, filename):
        mock_image = MagicMock(return_value=_FAKE_CHUNKS)
        mock_pdf   = MagicMock(return_value=_FAKE_CHUNKS)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_image", mock_image), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_pdf), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload(filename)

        mock_image.assert_called_once()
        mock_pdf.assert_not_called()

    @pytest.mark.parametrize("filename", ["clip.mp4", "recording.mov"])
    def test_video_extensions_call_chunk_video(self, tmp_path, filename):
        mock_video = MagicMock(return_value=_FAKE_CHUNKS)
        mock_pdf   = MagicMock(return_value=_FAKE_CHUNKS)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_video", mock_video), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_pdf), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload(filename)

        mock_video.assert_called_once()
        mock_pdf.assert_not_called()

    def test_pdf_extension_calls_chunk_pdf(self, tmp_path):
        mock_pdf   = MagicMock(return_value=_FAKE_CHUNKS)
        mock_image = MagicMock(return_value=_FAKE_CHUNKS)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_pdf), \
             patch("backend.routers.upload.chunking.chunk_image", mock_image), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("doc.pdf")

        mock_pdf.assert_called_once()
        mock_image.assert_not_called()


# ---------------------------------------------------------------------------
# Ingestion failure — service errors must not bubble as 500
# ---------------------------------------------------------------------------

class TestIngestionFailure:
    def test_chunker_failure_returns_200(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", side_effect=RuntimeError("boom")):
            resp = _upload("report.pdf")
        assert resp.status_code == 200

    def test_chunker_failure_status_is_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", side_effect=RuntimeError("boom")):
            body = _upload("report.pdf").json()
        assert body["status"] == "upload_failed_to_index"

    def test_embed_failure_status_is_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=ValueError("api error")):
            body = _upload("report.pdf").json()
        assert body["status"] == "upload_failed_to_index"

    def test_vectorstore_failure_status_is_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", side_effect=IOError("chroma down")):
            body = _upload("report.pdf").json()
        assert body["status"] == "upload_failed_to_index"

    def test_failure_response_preserves_metadata(self, tmp_path):
        """file_id, filename, and size_bytes must still be correct on failure."""
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", side_effect=RuntimeError("boom")):
            body = _upload("report.pdf", b"12345").json()
        assert body["filename"] == "report.pdf"
        assert body["size_bytes"] == 5
        assert "file_id" in body

    def test_add_chunks_not_called_after_embed_failure(self, tmp_path):
        """If embed_chunks raises, add_chunks must never be called."""
        mock_add = MagicMock()
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=ValueError("api error")), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf")
        mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# Validation — bad extension and oversized file
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unsupported_extension_returns_400(self):
        resp = _upload("malware.exe")
        assert resp.status_code == 400

    def test_unsupported_extension_detail_message(self):
        body = _upload("malware.exe").json()
        assert ".exe" in body["detail"]

    def test_oversized_file_returns_413(self, tmp_path):
        import backend.config as config
        oversized = b"x" * (config.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)):
            resp = _upload("big.pdf", oversized)
        assert resp.status_code == 413

    def test_services_not_called_for_bad_extension(self):
        """Chunking must never be reached if the extension is rejected."""
        mock_chunk = MagicMock()
        with patch("backend.routers.upload.chunking.chunk_pdf", mock_chunk):
            _upload("bad.exe")
        mock_chunk.assert_not_called()
