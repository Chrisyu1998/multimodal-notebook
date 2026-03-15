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

_PNG_CONTENT = b"fake png content"

_MP4_CONTENT = b"fake mp4 content"
_MP4_HASH = hashlib.sha256(_MP4_CONTENT).hexdigest()

# Fake video chunks mirroring the dual-stream shape (Chunk A video_clip + Chunk B video_summary)
_FAKE_VIDEO_CHUNKS = [
    {
        "type": "video",
        "video_bytes": b"fake-mp4-scene-0",
        "text": "A person explains rail transport network design.",
        "source": "/tmp/clip.mp4",
        "page": 0,
        "chunk_index": 0,
        "modality": "video_clip",
        "parent_scene_id": "deadbeef" * 8,
        "start_time_seconds": 0.0,
        "end_time_seconds": 30.0,
        "scene_index": 0,
        "forced_split": False,
    },
    {
        "type": "document",
        "pdf_bytes": b"%PDF-fake",
        "text": "A person explains rail transport network design.",
        "source": "/tmp/clip.mp4",
        "page": 0,
        "chunk_index": 1,
        "modality": "video_summary",
        "parent_scene_id": "deadbeef" * 8,
        "start_time_seconds": 0.0,
        "end_time_seconds": 30.0,
        "scene_index": 0,
    },
]
_FAKE_VIDEO_EMBEDDED = [{"embedding": [0.5, 0.6], **c} for c in _FAKE_VIDEO_CHUNKS]

# Fake video-only chunks (no Chunk B — visual summary was empty)
_FAKE_VIDEO_CHUNKS_CLIP_ONLY = [_FAKE_VIDEO_CHUNKS[0]]
_FAKE_VIDEO_EMBEDDED_CLIP_ONLY = [_FAKE_VIDEO_EMBEDDED[0]]

# Fake image chunks mirroring the dual-stream shape (global + one local)
_FAKE_IMAGE_CHUNKS = [
    {
        "type": "image",
        "image_bytes": b"raw-image-bytes",
        "text": "Image: photo.png",
        "source": "/tmp/photo.png",
        "page": 0,
        "chunk_index": 0,
        "modality": "image_global",
        "region_type": "full",
        "crop_bbox": None,
        "region_count": 1,
        "parent_image_id": "abc123",
    },
    {
        "type": "image",
        "image_bytes": b"crop-bytes",
        "text": "Image: photo.png | Region: table",
        "source": "/tmp/photo.png",
        "page": 0,
        "chunk_index": 1,
        "modality": "image_local",
        "region_type": "table",
        "crop_bbox": [0, 0, 100, 50],
        "parent_image_id": "abc123",
        "width_px": 100,
        "height_px": 50,
        "crop_bbox_normalized": [0.0, 0.0, 0.5, 0.5],
    },
]
_FAKE_IMAGE_EMBEDDED = [{"embedding": [0.3, 0.4], **c} for c in _FAKE_IMAGE_CHUNKS]


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
    @pytest.mark.parametrize("filename", ["photo.png", "diagram.jpeg"])
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
# Happy path — Image
# ---------------------------------------------------------------------------

class TestImageIngestion:
    @pytest.mark.parametrize("filename", ["photo.png", "scan.jpeg"])
    def test_returns_200(self, tmp_path, filename):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            resp = _upload(filename, _PNG_CONTENT)
        assert resp.status_code == 200

    @pytest.mark.parametrize("filename", ["photo.png", "scan.jpeg"])
    def test_status_is_indexed(self, tmp_path, filename):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload(filename, _PNG_CONTENT).json()
        assert body["status"] == "indexed"

    def test_response_fields_present(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload("photo.png", _PNG_CONTENT).json()
        assert body["filename"] == "photo.png"
        assert body["size_bytes"] == len(_PNG_CONTENT)
        assert "file_id" in body

    def test_pipeline_call_order(self, tmp_path):
        """chunk_image → embed_chunks → add_chunks must be called in that order."""
        call_order = []
        mock_chunk = MagicMock(side_effect=lambda *a, **kw: call_order.append("chunk") or _FAKE_IMAGE_CHUNKS)
        mock_embed = MagicMock(side_effect=lambda *a, **kw: call_order.append("embed") or _FAKE_IMAGE_EMBEDDED)
        mock_add   = MagicMock(side_effect=lambda *a, **kw: call_order.append("add"))

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("photo.png", _PNG_CONTENT)

        assert call_order == ["chunk", "embed", "add"]

    def test_chunk_image_called_with_temp_file_path(self, tmp_path):
        """chunk_image receives the saved temp path, not the original filename."""
        mock_chunk = MagicMock(return_value=_FAKE_IMAGE_CHUNKS)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("photo.png", _PNG_CONTENT)

        called_path = mock_chunk.call_args[0][0]
        assert called_path.startswith(str(tmp_path))
        assert "photo.png" in called_path

    def test_all_chunks_stamped_with_original_filename(self, tmp_path):
        """source on every chunk (global + local) must be the original filename."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_IMAGE_EMBEDDED

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("photo.png", _PNG_CONTENT)

        for chunk in captured["chunks"]:
            assert chunk["source"] == "photo.png"

    def test_all_chunks_stamped_with_file_hash(self, tmp_path):
        """file_hash on every chunk must equal the SHA-256 of the uploaded bytes."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_IMAGE_EMBEDDED

        expected_hash = hashlib.sha256(_PNG_CONTENT).hexdigest()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("photo.png", _PNG_CONTENT)

        for chunk in captured["chunks"]:
            assert chunk["file_hash"] == expected_hash

    def test_add_receives_embedded_output(self, tmp_path):
        """add_chunks must be called with the list returned by embed_chunks."""
        mock_add = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("photo.png", _PNG_CONTENT)

        mock_add.assert_called_once_with(_FAKE_IMAGE_EMBEDDED)

    def test_chunk_pdf_not_called_for_image(self, tmp_path):
        """chunk_pdf must never be invoked for an image upload."""
        mock_pdf = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_pdf), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("photo.png", _PNG_CONTENT)

        mock_pdf.assert_not_called()


# ---------------------------------------------------------------------------
# Image corner cases
# ---------------------------------------------------------------------------

class TestImageCornerCases:
    def test_value_error_from_chunk_image_returns_upload_failed_to_index(self, tmp_path):
        """chunk_image raises ValueError for format mismatch — must not 500."""
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image",
                   side_effect=ValueError("unsupported format 'BMP'")):
            body = _upload("photo.png", b"BM fake bmp").json()
        assert body["status"] == "upload_failed_to_index"

    def test_value_error_from_chunk_image_does_not_raise_500(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image",
                   side_effect=ValueError("unsupported format 'BMP'")):
            resp = _upload("photo.png", b"BM fake bmp")
        assert resp.status_code == 200

    def test_embed_failure_on_image_returns_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=RuntimeError("gemini down")):
            body = _upload("photo.png", _PNG_CONTENT).json()
        assert body["status"] == "upload_failed_to_index"

    def test_vectorstore_failure_on_image_returns_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_IMAGE_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", side_effect=IOError("chroma down")):
            body = _upload("photo.png", _PNG_CONTENT).json()
        assert body["status"] == "upload_failed_to_index"

    def test_failure_response_preserves_image_metadata(self, tmp_path):
        """filename and size_bytes must be correct even when chunk_image fails."""
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image",
                   side_effect=RuntimeError("boom")):
            body = _upload("photo.png", _PNG_CONTENT).json()
        assert body["filename"] == "photo.png"
        assert body["size_bytes"] == len(_PNG_CONTENT)
        assert "file_id" in body

    def test_image_dedup_returns_already_indexed(self):
        """Uploading an image with content already in the store → already_indexed."""
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True):
            body = _upload("photo.png", _PNG_CONTENT).json()
        assert body["status"] == "already_indexed"

    def test_image_dedup_skips_chunk_image(self):
        """chunk_image must not be called when the content is already indexed."""
        mock_chunk = MagicMock()
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True), \
             patch("backend.routers.upload.chunking.chunk_image", mock_chunk):
            _upload("photo.png", _PNG_CONTENT)
        mock_chunk.assert_not_called()

    def test_add_chunks_not_called_after_image_embed_failure(self, tmp_path):
        """If embed_chunks raises, add_chunks must not be called."""
        mock_add = MagicMock()
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=RuntimeError("api error")), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("photo.png", _PNG_CONTENT)
        mock_add.assert_not_called()

    def test_global_and_local_chunks_both_stamped(self, tmp_path):
        """When chunk_image returns multiple chunks, every one gets source + file_hash."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_IMAGE_EMBEDDED

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=_FAKE_IMAGE_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("photo.png", _PNG_CONTENT)

        assert len(captured["chunks"]) == 2
        for chunk in captured["chunks"]:
            assert chunk["source"] == "photo.png"
            assert "file_hash" in chunk

    def test_empty_chunk_list_from_chunk_image_still_calls_embed(self, tmp_path):
        """If chunk_image returns [], embed_chunks and add_chunks are still called
        (with empty lists) — no short-circuit in upload.py."""
        mock_embed = MagicMock(return_value=[])
        mock_add = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_image", return_value=[]), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            body = _upload("photo.png", _PNG_CONTENT).json()

        mock_embed.assert_called_once_with([])
        mock_add.assert_called_once_with([])
        assert body["status"] == "indexed"


# ---------------------------------------------------------------------------
# Happy path — Video
# ---------------------------------------------------------------------------

class TestVideoIngestion:
    @pytest.mark.parametrize("filename", ["clip.mp4", "recording.mov"])
    def test_returns_200(self, tmp_path, filename):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            resp = _upload(filename, _MP4_CONTENT)
        assert resp.status_code == 200

    @pytest.mark.parametrize("filename", ["clip.mp4", "recording.mov"])
    def test_status_is_indexed(self, tmp_path, filename):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload(filename, _MP4_CONTENT).json()
        assert body["status"] == "indexed"

    def test_response_fields_present(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload("clip.mp4", _MP4_CONTENT).json()
        assert body["filename"] == "clip.mp4"
        assert body["size_bytes"] == len(_MP4_CONTENT)
        assert "file_id" in body

    def test_pipeline_call_order(self, tmp_path):
        """chunk_video → embed_chunks → add_chunks must be called in that order."""
        call_order = []
        mock_chunk = MagicMock(side_effect=lambda *a, **kw: call_order.append("chunk") or _FAKE_VIDEO_CHUNKS)
        mock_embed = MagicMock(side_effect=lambda *a, **kw: call_order.append("embed") or _FAKE_VIDEO_EMBEDDED)
        mock_add   = MagicMock(side_effect=lambda *a, **kw: call_order.append("add"))

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("clip.mp4", _MP4_CONTENT)

        assert call_order == ["chunk", "embed", "add"]

    def test_chunk_video_called_with_temp_file_path(self, tmp_path):
        """chunk_video receives the saved temp path, not the original filename."""
        mock_chunk = MagicMock(return_value=_FAKE_VIDEO_CHUNKS)

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", mock_chunk), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("clip.mp4", _MP4_CONTENT)

        called_path = mock_chunk.call_args[0][0]
        assert called_path.startswith(str(tmp_path))
        assert "clip.mp4" in called_path

    def test_all_chunks_stamped_with_original_filename(self, tmp_path):
        """source on every chunk (video_clip + video_summary) must be the original filename."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_VIDEO_EMBEDDED

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("clip.mp4", _MP4_CONTENT)

        for chunk in captured["chunks"]:
            assert chunk["source"] == "clip.mp4"

    def test_all_chunks_stamped_with_file_hash(self, tmp_path):
        """file_hash on every chunk must equal the SHA-256 of the uploaded bytes."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_VIDEO_EMBEDDED

        expected_hash = hashlib.sha256(_MP4_CONTENT).hexdigest()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("clip.mp4", _MP4_CONTENT)

        for chunk in captured["chunks"]:
            assert chunk["file_hash"] == expected_hash

    def test_add_receives_embedded_output(self, tmp_path):
        """add_chunks must be called with the list returned by embed_chunks."""
        mock_add = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("clip.mp4", _MP4_CONTENT)

        mock_add.assert_called_once_with(_FAKE_VIDEO_EMBEDDED)

    def test_chunk_pdf_not_called_for_video(self, tmp_path):
        """chunk_pdf must never be invoked for a video upload."""
        mock_pdf = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.chunking.chunk_pdf", mock_pdf), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("clip.mp4", _MP4_CONTENT)

        mock_pdf.assert_not_called()


# ---------------------------------------------------------------------------
# Video corner cases
# ---------------------------------------------------------------------------

class TestVideoCornerCases:
    def test_chunk_video_failure_returns_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video",
                   side_effect=RuntimeError("ffmpeg error")):
            body = _upload("clip.mp4", _MP4_CONTENT).json()
        assert body["status"] == "upload_failed_to_index"

    def test_chunk_video_failure_does_not_raise_500(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video",
                   side_effect=RuntimeError("ffmpeg error")):
            resp = _upload("clip.mp4", _MP4_CONTENT)
        assert resp.status_code == 200

    def test_embed_failure_on_video_returns_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks",
                   side_effect=RuntimeError("gemini 503")):
            body = _upload("clip.mp4", _MP4_CONTENT).json()
        assert body["status"] == "upload_failed_to_index"

    def test_vectorstore_failure_on_video_returns_upload_failed_to_index(self, tmp_path):
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks",
                   side_effect=IOError("chroma down")):
            body = _upload("clip.mp4", _MP4_CONTENT).json()
        assert body["status"] == "upload_failed_to_index"

    def test_failure_response_preserves_video_metadata(self, tmp_path):
        """filename and size_bytes must be correct even when chunk_video fails."""
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video",
                   side_effect=RuntimeError("boom")):
            body = _upload("clip.mp4", _MP4_CONTENT).json()
        assert body["filename"] == "clip.mp4"
        assert body["size_bytes"] == len(_MP4_CONTENT)
        assert "file_id" in body

    def test_video_dedup_returns_already_indexed(self):
        """Uploading a video with content already in the store → already_indexed."""
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True):
            body = _upload("clip.mp4", _MP4_CONTENT).json()
        assert body["status"] == "already_indexed"

    def test_video_dedup_skips_chunk_video(self):
        """chunk_video must not be called when the content is already indexed."""
        mock_chunk = MagicMock()
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=True), \
             patch("backend.routers.upload.chunking.chunk_video", mock_chunk):
            _upload("clip.mp4", _MP4_CONTENT)
        mock_chunk.assert_not_called()

    def test_add_chunks_not_called_after_video_embed_failure(self, tmp_path):
        """If embed_chunks raises, add_chunks must not be called."""
        mock_add = MagicMock()
        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks",
                   side_effect=RuntimeError("api error")), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("clip.mp4", _MP4_CONTENT)
        mock_add.assert_not_called()

    def test_video_clip_and_summary_chunks_both_stamped(self, tmp_path):
        """Both video_clip and video_summary chunks must get source + file_hash."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_VIDEO_EMBEDDED

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            _upload("clip.mp4", _MP4_CONTENT)

        assert len(captured["chunks"]) == 2
        modalities = {c["modality"] for c in captured["chunks"]}
        assert modalities == {"video_clip", "video_summary"}
        for chunk in captured["chunks"]:
            assert chunk["source"] == "clip.mp4"
            assert "file_hash" in chunk

    def test_clip_only_chunks_stamped_when_no_summaries(self, tmp_path):
        """When chunk_video returns only Chunk A (no summaries), stamping still works."""
        captured = {}
        def capture_embed(chunks):
            captured["chunks"] = chunks
            return _FAKE_VIDEO_EMBEDDED_CLIP_ONLY

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video",
                   return_value=_FAKE_VIDEO_CHUNKS_CLIP_ONLY), \
             patch("backend.routers.upload.embeddings.embed_chunks", side_effect=capture_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks"):
            body = _upload("clip.mp4", _MP4_CONTENT).json()

        assert body["status"] == "indexed"
        assert len(captured["chunks"]) == 1
        assert captured["chunks"][0]["modality"] == "video_clip"
        assert captured["chunks"][0]["source"] == "clip.mp4"

    def test_empty_chunk_list_from_chunk_video_still_calls_embed(self, tmp_path):
        """If chunk_video returns [], embed_chunks and add_chunks are still called."""
        mock_embed = MagicMock(return_value=[])
        mock_add = MagicMock()

        with patch("backend.config.TMP_UPLOADS_DIR", str(tmp_path)), \
             patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=[]), \
             patch("backend.routers.upload.embeddings.embed_chunks", mock_embed), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            body = _upload("clip.mp4", _MP4_CONTENT).json()

        mock_embed.assert_called_once_with([])
        mock_add.assert_called_once_with([])
        assert body["status"] == "indexed"

    def test_dedup_check_uses_content_hash_not_filename_for_video(self):
        """Same video filename with different bytes → not a duplicate."""
        mock_add = MagicMock()
        with patch("backend.routers.upload.vectorstore.is_file_indexed", return_value=False), \
             patch("backend.routers.upload.chunking.chunk_video", return_value=_FAKE_VIDEO_CHUNKS), \
             patch("backend.routers.upload.embeddings.embed_chunks", return_value=_FAKE_VIDEO_EMBEDDED), \
             patch("backend.routers.upload.vectorstore.add_chunks", mock_add):
            _upload("clip.mp4", b"video version 1")
            _upload("clip.mp4", b"video version 2")
        assert mock_add.call_count == 2

    def test_is_file_indexed_called_with_sha256_of_video_content(self):
        """is_file_indexed must receive the SHA-256 of the video bytes."""
        expected_hash = hashlib.sha256(_MP4_CONTENT).hexdigest()
        mock_is_indexed = MagicMock(return_value=True)

        with patch("backend.routers.upload.vectorstore.is_file_indexed", mock_is_indexed):
            _upload("clip.mp4", _MP4_CONTENT)

        mock_is_indexed.assert_called_once_with(expected_hash)


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

    @pytest.mark.parametrize("filename", ["photo.jpg", "logo.webp"])
    def test_removed_image_extensions_return_400(self, filename):
        """.jpg and .webp are no longer supported — must return 400."""
        resp = _upload(filename)
        assert resp.status_code == 400
