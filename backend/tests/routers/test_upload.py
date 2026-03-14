"""
Tests for POST /upload — orchestration logic in backend.routers.upload.

All three services (chunking, embeddings, vectorstore) are mocked so
these tests run without any real files, Gemini calls, or ChromaDB I/O.
The file is always written to a real temp dir (patched via TMP_UPLOADS_DIR)
so we verify the disk-write path too without polluting the project tree.
"""

import hashlib
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

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

_PDF_CONTENT = b"fake pdf content"
_PDF_HASH = hashlib.sha256(_PDF_CONTENT).hexdigest()


def _upload(filename: str, content: bytes = _PDF_CONTENT) -> dict:
    """POST /upload with an in-memory file and return the response."""
    return client.post(
        "/upload/",
        files={"file": (filename, BytesIO(content), "application/octet-stream")},
    )


# ---------------------------------------------------------------------------
# Happy path — PDF
# ---------------------------------------------------------------------------

class TestPdfIngestion:
    def test_returns_200(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            resp = _upload("report.pdf")
        assert resp.status_code == 200

    def test_status_is_indexed(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload("report.pdf").json()
        assert body["status"] == "indexed"

    def test_response_fields_present(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
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
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf")

        assert call_order == ["chunk", "embed", "add"]

    def test_chunks_stamped_with_original_filename(self, tmp_path):
        """source on each chunk must be the original filename, not the UUID temp path."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_EMBEDDED

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("report.pdf", _PDF_CONTENT)

        assert captured["chunks"][0]["source"] == "report.pdf"

    def test_chunks_stamped_with_file_hash(self, tmp_path):
        """file_hash on each chunk must equal the SHA-256 of the uploaded bytes."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_EMBEDDED

        content = b"deterministic content"
        expected_hash = hashlib.sha256(content).hexdigest()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("report.pdf", content)

        assert captured["chunks"][0]["file_hash"] == expected_hash

    def test_add_receives_embedded_output(self, tmp_path):
        """add_chunks must be called with the list returned by embed_chunks."""
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDED)
        mock_add   = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf")

        mock_add.assert_called_once_with(_FAKE_EMBEDDED)


# ---------------------------------------------------------------------------
# Deduplication — same file must not be indexed twice
# ---------------------------------------------------------------------------

class TestDedup:
    def test_already_indexed_returns_already_indexed_status(self, tmp_path):
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True):
            body = _upload("report.pdf").json()
        assert body["status"] == "already_indexed"

    def test_already_indexed_returns_200(self, tmp_path):
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True):
            resp = _upload("report.pdf")
        assert resp.status_code == 200

    def test_already_indexed_skips_chunking(self, tmp_path):
        mock_chunk = MagicMock()
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_chunk):
            _upload("report.pdf")
        mock_chunk.assert_not_called()

    def test_already_indexed_skips_embedding(self, tmp_path):
        mock_embed = MagicMock()
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed):
            _upload("report.pdf")
        mock_embed.assert_not_called()

    def test_already_indexed_skips_add_chunks(self, tmp_path):
        mock_add = MagicMock()
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf")
        mock_add.assert_not_called()

    def test_already_indexed_response_includes_filename_and_size(self):
        content = b"some bytes"
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True):
            body = _upload("report.pdf", content).json()
        assert body["filename"] == "report.pdf"
        assert body["size_bytes"] == len(content)

    def test_different_content_is_not_deduplicated(self, tmp_path):
        """Two files with different bytes must both be indexed."""
        mock_add = MagicMock()
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("a.pdf", b"content A")
            _upload("b.pdf", b"content B")
        assert mock_add.call_count == 2

    def test_dedup_check_uses_content_not_filename(self):
        """Same filename with different bytes → not a duplicate."""
        mock_add = MagicMock()
        # is_file_indexed returns False (new content), even though filename is same
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("report.pdf", b"version 1")
            _upload("report.pdf", b"version 2")
        assert mock_add.call_count == 2

    def test_is_file_indexed_called_with_sha256_of_content(self):
        """is_file_indexed must receive the SHA-256 hex digest of the uploaded bytes."""
        content = b"specific content"
        expected_hash = hashlib.sha256(content).hexdigest()
        mock_is_indexed = MagicMock(return_value=True)

        with patch("backend.routers.upload.vectorstore.is_file_indexed", mock_is_indexed):
            _upload("report.pdf", content)

        mock_is_indexed.assert_called_once_with(expected_hash)


# ---------------------------------------------------------------------------
# Chunker dispatch — correct function called per file type
# ---------------------------------------------------------------------------

class TestChunkerDispatch:
    @pytest.mark.parametrize("filename", ["photo.png", "scan.jpg", "diagram.jpeg", "logo.webp"])
    def test_image_extensions_call_chunk_image(self, tmp_path, filename):
        mock_image = MagicMock(return_value=_FAKE_CHUNKS)
        mock_pdf   = MagicMock(return_value=_FAKE_CHUNKS)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
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
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
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
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
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
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", side_effect=RuntimeError("boom")):
            resp = _upload("report.pdf")
        assert resp.status_code == 200

    def test_chunker_failure_status_is_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", side_effect=RuntimeError("boom")):
            body = _upload("report.pdf").json()
        assert body["status"] == "upload_failed_to_index"

    def test_embed_failure_status_is_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=ValueError("api error")):
            body = _upload("report.pdf").json()
        assert body["status"] == "upload_failed_to_index"

    def test_vectorstore_failure_status_is_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", side_effect=IOError("chroma down")):
            body = _upload("report.pdf").json()
        assert body["status"] == "upload_failed_to_index"

    def test_failure_response_preserves_metadata(self, tmp_path):
        """file_id, filename, and size_bytes must still be correct on failure."""
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_pdf", side_effect=RuntimeError("boom")):
            body = _upload("report.pdf", b"12345").json()
        assert body["filename"] == "report.pdf"
        assert body["size_bytes"] == 5
        assert "file_id" in body

    def test_add_chunks_not_called_after_embed_failure(self, tmp_path):
        """If embed_chunks raises, add_chunks must never be called."""
        mock_add = MagicMock()
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
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
