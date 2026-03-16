"""
Tests for backend.services.gcs — upload_bytes and download_bytes.

The Google Cloud Storage client is patched in every test so no real
GCS bucket is touched.  Module-level init is patched before import.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

# Patch at import time so the module-level _client / _bucket don't
# attempt a real connection when the module is first loaded.
with (
    patch("google.cloud.storage.Client"),
):
    import backend.services.gcs as gcs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_bucket():
    """Return a fresh mock bucket wired into the gcs module."""
    bucket = MagicMock()
    with patch.object(gcs, "_bucket", bucket):
        yield bucket


# ---------------------------------------------------------------------------
# upload_bytes
# ---------------------------------------------------------------------------

class TestUploadBytes:
    def test_returns_gs_uri(self, mock_bucket):
        """upload_bytes must return a gs:// URI for the given blob name."""
        blob = MagicMock()
        mock_bucket.blob.return_value = blob

        uri = gcs.upload_bytes(b"hello", "media/abc/img_0.jpg", "image/jpeg")

        assert uri == "gs://test-bucket/media/abc/img_0.jpg"

    def test_calls_blob_with_correct_name(self, mock_bucket):
        """_bucket.blob must be called with the exact blob_name passed in."""
        blob = MagicMock()
        mock_bucket.blob.return_value = blob

        gcs.upload_bytes(b"data", "media/hash/audio_0.mp3", "audio/mp3")

        mock_bucket.blob.assert_called_once_with("media/hash/audio_0.mp3")

    def test_calls_upload_from_string_with_data_and_content_type(self, mock_bucket):
        """upload_from_string must receive the raw bytes and the MIME type."""
        blob = MagicMock()
        mock_bucket.blob.return_value = blob
        data = b"\xff\xd8\xff raw jpeg"

        gcs.upload_bytes(data, "media/x/img_1.jpg", "image/jpeg")

        blob.upload_from_string.assert_called_once_with(data, content_type="image/jpeg")

    def test_uri_uses_bucket_name_from_config(self, mock_bucket):
        """The bucket name in the returned URI must match GCS_BUCKET_NAME."""
        blob = MagicMock()
        mock_bucket.blob.return_value = blob

        with patch.object(gcs.config, "GCS_BUCKET_NAME", "my-prod-bucket"):
            uri = gcs.upload_bytes(b"x", "some/path.jpg", "image/jpeg")

        assert uri.startswith("gs://my-prod-bucket/")

    def test_empty_bytes_are_uploaded(self, mock_bucket):
        """upload_bytes must not short-circuit on empty data."""
        blob = MagicMock()
        mock_bucket.blob.return_value = blob

        gcs.upload_bytes(b"", "media/x/empty.jpg", "image/jpeg")

        blob.upload_from_string.assert_called_once_with(b"", content_type="image/jpeg")

    def test_propagates_storage_exception(self, mock_bucket):
        """If GCS raises, upload_bytes must not swallow the error."""
        blob = MagicMock()
        blob.upload_from_string.side_effect = Exception("GCS unavailable")
        mock_bucket.blob.return_value = blob

        with pytest.raises(Exception, match="GCS unavailable"):
            gcs.upload_bytes(b"data", "media/x/img.jpg", "image/jpeg")


# ---------------------------------------------------------------------------
# download_bytes
# ---------------------------------------------------------------------------

class TestDownloadBytes:
    def test_returns_raw_bytes(self, mock_bucket):
        """download_bytes must return exactly what download_as_bytes produces."""
        blob = MagicMock()
        blob.download_as_bytes.return_value = b"binary content"
        mock_bucket.blob.return_value = blob

        result = gcs.download_bytes("gs://test-bucket/media/abc/img_0.jpg")

        assert result == b"binary content"

    def test_strips_bucket_prefix_before_calling_blob(self, mock_bucket):
        """The blob name passed to _bucket.blob must have the gs://<bucket>/ prefix removed."""
        blob = MagicMock()
        blob.download_as_bytes.return_value = b""
        mock_bucket.blob.return_value = blob

        gcs.download_bytes("gs://test-bucket/media/abc/audio_1.mp3")

        mock_bucket.blob.assert_called_once_with("media/abc/audio_1.mp3")

    def test_calls_download_as_bytes(self, mock_bucket):
        """download_as_bytes must be called once on the blob object."""
        blob = MagicMock()
        blob.download_as_bytes.return_value = b"bytes"
        mock_bucket.blob.return_value = blob

        gcs.download_bytes("gs://test-bucket/some/path.jpg")

        blob.download_as_bytes.assert_called_once()

    def test_nested_path_is_preserved(self, mock_bucket):
        """Blob names with multiple path components are kept intact."""
        blob = MagicMock()
        blob.download_as_bytes.return_value = b""
        mock_bucket.blob.return_value = blob

        gcs.download_bytes("gs://test-bucket/media/deadbeef/video_frame_3.jpg")

        mock_bucket.blob.assert_called_once_with("media/deadbeef/video_frame_3.jpg")

    def test_propagates_storage_exception(self, mock_bucket):
        """If GCS raises, download_bytes must not swallow the error."""
        blob = MagicMock()
        blob.download_as_bytes.side_effect = Exception("blob not found")
        mock_bucket.blob.return_value = blob

        with pytest.raises(Exception, match="blob not found"):
            gcs.download_bytes("gs://test-bucket/media/x/img.jpg")
