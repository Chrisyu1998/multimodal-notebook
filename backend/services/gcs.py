"""
GCS helpers — upload and download media blobs.

All other modules must import from here; no other file touches
google.cloud.storage directly.

Blob naming convention:
    media/{file_hash}/img_{chunk_index}.jpg        — image global/local chunks
    media/{file_hash}/video_frame_{chunk_index}.jpg — mid-point frame for reranking
    media/{file_hash}/audio_{chunk_index}.mp3       — audio clip chunks
"""

from google.cloud import storage
from loguru import logger

import backend.config as config

_client = storage.Client()
_bucket = _client.bucket(config.GCS_BUCKET_NAME)


def upload_bytes(data: bytes, blob_name: str, content_type: str) -> str:
    """Upload *data* to GCS and return the gs:// URI.

    Args:
        data:         Raw bytes to upload.
        blob_name:    Path within the bucket, e.g. "media/abc123/img_0.jpg".
        content_type: MIME type stored on the object, e.g. "image/jpeg".

    Returns:
        gs://<bucket>/<blob_name>
    """
    blob = _bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    uri = f"gs://{config.GCS_BUCKET_NAME}/{blob_name}"
    logger.debug(f"gcs: uploaded {len(data):,} bytes → {uri}")
    return uri


def download_bytes(gcs_uri: str) -> bytes:
    """Download and return the raw bytes at *gcs_uri*.

    Args:
        gcs_uri: Full gs:// URI, e.g. "gs://my-bucket/media/abc123/img_0.jpg".

    Returns:
        Raw bytes of the object.
    """
    prefix = f"gs://{config.GCS_BUCKET_NAME}/"
    blob_name = gcs_uri.removeprefix(prefix)
    blob = _bucket.blob(blob_name)
    data: bytes = blob.download_as_bytes()
    logger.debug(f"gcs: downloaded {len(data):,} bytes ← {gcs_uri}")
    return data
