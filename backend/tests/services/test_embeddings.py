"""
Tests for backend.services.embeddings.

All Gemini API calls are mocked — no real network requests are made.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

import backend.services.embeddings as emb
from backend.services.embeddings import EmbeddingBatchError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 768


def fake_vector(seed: float = 1.0) -> list[float]:
    val = seed / math.sqrt(DIM)
    return [val] * DIM


def embed_response(*vectors: list[float]):
    """Build a mock EmbedContentResponse with the given embedding vectors."""
    embeddings = [SimpleNamespace(values=v) for v in vectors]
    return SimpleNamespace(embeddings=embeddings)


def make_image_chunk(idx: int, png: bool = False) -> dict:
    magic = b"\x89PNG\r\n\x1a\n" if png else b"\xff\xd8\xff\xe0"
    return {"type": "image", "image_bytes": magic + bytes([idx]), "source": f"img_{idx}", "chunk_index": idx}


def make_media_chunk(chunk_type: str, idx: int) -> dict:
    bytes_field = emb._MEDIA_BYTES_FIELD[chunk_type]
    return {"type": chunk_type, bytes_field: b"\x00\x01" + bytes([idx]), "source": f"file_{idx}", "chunk_index": idx}


def capture_parts_call(model, contents):  # noqa: ARG001
    n = len(contents) if isinstance(contents, list) else 1
    return embed_response(*[fake_vector(float(i + 1)) for i in range(n)])


def google_api_error(status_code: int) -> Exception:
    """Fake an API exception with a status_code attribute."""
    exc = Exception(f"API error {status_code}")
    exc.status_code = status_code
    return exc


def plain_error(msg: str = "network blip") -> Exception:
    """A plain Exception with no status code — should be treated as transient."""
    return Exception(msg)


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_returns_list_of_floats(self):
        with patch.object(emb._client.models, "embed_content", return_value=embed_response(fake_vector())):
            result = emb.embed_text("What is machine learning?")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_calls_correct_model(self):
        mock_fn = MagicMock(return_value=embed_response(fake_vector()))
        with patch.object(emb._client.models, "embed_content", mock_fn):
            emb.embed_text("test")
        assert mock_fn.call_args.kwargs["model"] == emb.config.EMBEDDING_MODEL

    def test_passes_text_as_contents(self):
        mock_fn = MagicMock(return_value=embed_response(fake_vector()))
        with patch.object(emb._client.models, "embed_content", mock_fn):
            emb.embed_text("hello world")
        assert mock_fn.call_args.kwargs["contents"] == "hello world"

    def test_retries_transient_error(self):
        good = embed_response(fake_vector())
        mock_fn = MagicMock(side_effect=[plain_error(), plain_error(), good])
        with patch.object(emb._client.models, "embed_content", mock_fn), \
             patch("backend.services.embeddings.time.sleep"):
            result = emb.embed_text("retry me")
        assert mock_fn.call_count == 3
        assert result == fake_vector()

    def test_raises_embedding_batch_error_after_max_retries(self):
        mock_fn = MagicMock(side_effect=plain_error())
        with patch.object(emb._client.models, "embed_content", mock_fn), \
             patch("backend.services.embeddings.time.sleep"):
            with pytest.raises(EmbeddingBatchError):
                emb.embed_text("doomed")
        assert mock_fn.call_count == emb._MAX_RETRIES

    def test_permanent_error_fails_immediately(self):
        mock_fn = MagicMock(side_effect=google_api_error(400))
        with patch.object(emb._client.models, "embed_content", mock_fn), \
             patch("backend.services.embeddings.time.sleep"):
            with pytest.raises(EmbeddingBatchError) as exc_info:
                emb.embed_text("bad input")
        assert mock_fn.call_count == 1
        assert exc_info.value.attempts == 1

    def test_exponential_backoff_sleep_durations(self):
        good = embed_response(fake_vector())
        mock_fn = MagicMock(side_effect=[plain_error(), plain_error(), good])
        with patch.object(emb._client.models, "embed_content", mock_fn), \
             patch("backend.services.embeddings.time.sleep") as mock_sleep:
            emb.embed_text("backoff")
        assert mock_sleep.call_args_list == [call(1), call(2)]


# ---------------------------------------------------------------------------
# EmbeddingBatchError
# ---------------------------------------------------------------------------

class TestEmbeddingBatchError:
    def test_carries_chunk_type_indices_cause_and_attempts(self):
        cause = Exception("boom")
        err = EmbeddingBatchError("image", [0, 1, 2], cause, attempts=2)
        assert err.chunk_type == "image"
        assert err.indices == [0, 1, 2]
        assert err.cause is cause
        assert err.attempts == 2

    def test_str_includes_context(self):
        err = EmbeddingBatchError("audio", [5], Exception("timeout"), attempts=3)
        assert "audio" in str(err)
        assert "5" in str(err)
        assert "3" in str(err)

    def test_permanent_error_reports_attempt_1(self):
        err = EmbeddingBatchError("document", [0], Exception("400"), attempts=1)
        assert "1 attempt" in str(err)


# ---------------------------------------------------------------------------
# _extract_status_code
# ---------------------------------------------------------------------------

class TestExtractStatusCode:
    def test_reads_status_code_attr(self):
        assert emb._extract_status_code(google_api_error(429)) == 429

    def test_returns_none_for_plain_exception(self):
        assert emb._extract_status_code(Exception("plain")) is None

    def test_does_not_read_errno_from_oserror(self):
        assert emb._extract_status_code(ConnectionRefusedError(111, "refused")) is None
        assert emb._extract_status_code(TimeoutError(110, "timed out")) is None


# ---------------------------------------------------------------------------
# _with_retry — permanent vs transient
# ---------------------------------------------------------------------------

class TestWithRetry:
    def test_does_not_retry_permanent_4xx(self):
        mock_fn = MagicMock(side_effect=google_api_error(400))
        with patch("backend.services.embeddings.time.sleep"):
            with pytest.raises(EmbeddingBatchError) as exc_info:
                emb._with_retry(mock_fn, chunk_type="image", indices=[0])
        assert mock_fn.call_count == 1
        assert exc_info.value.attempts == 1

    def test_retries_429(self):
        good = fake_vector()
        mock_fn = MagicMock(side_effect=[google_api_error(429), good])
        with patch("backend.services.embeddings.time.sleep"):
            result = emb._with_retry(mock_fn, chunk_type="image", indices=[0])
        assert result == good

    def test_retries_5xx(self):
        good = fake_vector()
        mock_fn = MagicMock(side_effect=[google_api_error(503), good])
        with patch("backend.services.embeddings.time.sleep"):
            result = emb._with_retry(mock_fn, chunk_type="image", indices=[0])
        assert result == good

    def test_unknown_status_is_retried(self):
        good = fake_vector()
        mock_fn = MagicMock(side_effect=[plain_error(), good])
        with patch("backend.services.embeddings.time.sleep"):
            result = emb._with_retry(mock_fn, chunk_type="image", indices=[0])
        assert mock_fn.call_count == 2
        assert result == good

    def test_raises_immediately_if_max_retries_is_zero(self):
        with patch.object(emb, "_MAX_RETRIES", 0):
            with pytest.raises(EmbeddingBatchError) as exc_info:
                emb._with_retry(lambda: None, chunk_type="image", indices=[0])
        assert exc_info.value.attempts == 0


# ---------------------------------------------------------------------------
# embed_chunks — edge cases
# ---------------------------------------------------------------------------

class TestEmbedChunksEdgeCases:
    def test_empty_list_is_noop(self):
        mock_fn = MagicMock()
        with patch.object(emb._client.models, "embed_content", mock_fn):
            result = emb.embed_chunks([])
        assert result == []
        mock_fn.assert_not_called()

    def test_returns_same_list_object(self):
        chunks = [make_image_chunk(0)]
        with patch.object(emb._client.models, "embed_content", side_effect=capture_parts_call):
            result = emb.embed_chunks(chunks)
        assert result is chunks

    def test_unknown_type_is_skipped(self):
        chunk = {"type": "unknown", "source": "x", "chunk_index": 0}
        with patch.object(emb._client.models, "embed_content", MagicMock()):
            result = emb.embed_chunks([chunk])
        assert "embedding" not in result[0]

    def test_raises_embedding_batch_error_with_context_on_failure(self):
        chunks = [make_image_chunk(0), make_image_chunk(1)]
        mock_fn = MagicMock(side_effect=plain_error())
        with patch.object(emb._client.models, "embed_content", mock_fn), \
             patch("backend.services.embeddings.time.sleep"):
            with pytest.raises(EmbeddingBatchError) as exc_info:
                emb.embed_chunks(chunks)
        err = exc_info.value
        assert err.chunk_type == "image"
        assert len(err.indices) > 0

    def test_permanent_error_in_embed_chunks_fails_immediately(self):
        chunks = [make_image_chunk(0)]
        mock_fn = MagicMock(side_effect=google_api_error(400))
        with patch.object(emb._client.models, "embed_content", mock_fn), \
             patch("backend.services.embeddings.time.sleep"):
            with pytest.raises(EmbeddingBatchError) as exc_info:
                emb.embed_chunks(chunks)
        assert mock_fn.call_count == 1
        assert exc_info.value.attempts == 1


# ---------------------------------------------------------------------------
# embed_chunks — images
# ---------------------------------------------------------------------------

class TestEmbedChunksImages:
    def test_image_chunks_get_embeddings(self):
        chunks = [make_image_chunk(i) for i in range(3)]
        with patch.object(emb._client.models, "embed_content", side_effect=capture_parts_call):
            result = emb.embed_chunks(chunks)
        assert all("embedding" in c for c in result)

    def test_embeddings_written_to_correct_indices(self):
        chunks = [make_image_chunk(i) for i in range(6)]
        with patch.object(emb._client.models, "embed_content", side_effect=capture_parts_call):
            result = emb.embed_chunks(chunks)
        assert all("embedding" in c for c in result)

    def test_images_batched_in_groups_of_image_batch_size(self):
        chunks = [make_image_chunk(i) for i in range(13)]
        call_sizes: list[int] = []

        def capture(model, contents):  # noqa: ARG001
            n = len(contents)
            call_sizes.append(n)
            return embed_response(*[fake_vector() for _ in range(n)])

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks(chunks)

        assert sorted(call_sizes) == [1, 6, 6]

    def test_sends_raw_bytes_not_base64(self):
        """New SDK uses Part.from_bytes — no base64 encoding should occur."""
        from google.genai import types as genai_types
        chunk = make_image_chunk(0)
        received_parts = []

        def capture(model, contents):  # noqa: ARG001
            received_parts.extend(contents)
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks([chunk])

        assert len(received_parts) == 1
        assert isinstance(received_parts[0], genai_types.Part)

    def test_jpeg_uses_jpeg_mime(self):
        chunk = make_image_chunk(0, png=False)
        received = []

        def capture(model, contents):  # noqa: ARG001
            received.append(contents)
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks([chunk])

        assert received[0][0].inline_data.mime_type == "image/jpeg"

    def test_png_uses_png_mime(self):
        chunk = make_image_chunk(0, png=True)
        received = []

        def capture(model, contents):  # noqa: ARG001
            received.append(contents)
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks([chunk])

        assert received[0][0].inline_data.mime_type == "image/png"


# ---------------------------------------------------------------------------
# _embed_media_batch — guards
# ---------------------------------------------------------------------------

class TestEmbedMediaBatch:
    def test_raises_on_mixed_types(self):
        items = [(0, make_image_chunk(0)), (1, make_media_chunk("video", 1))]
        with pytest.raises(ValueError, match="mixed types"):
            emb._embed_media_batch(items)

    def test_empty_items_returns_empty(self):
        assert emb._embed_media_batch([]) == []


# ---------------------------------------------------------------------------
# embed_chunks — single-file types (video, audio) — 1 call per chunk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_type,expected_mime", [
    ("video", "video/mp4"),
    ("audio", "audio/mp3"),
])
class TestEmbedChunksSingleFileTypes:
    def test_embedding_added(self, chunk_type, expected_mime):
        chunk = make_media_chunk(chunk_type, 0)
        with patch.object(emb._client.models, "embed_content", side_effect=capture_parts_call):
            result = emb.embed_chunks([chunk])
        assert "embedding" in result[0]

    def test_correct_mime_type_sent(self, chunk_type, expected_mime):
        chunk = make_media_chunk(chunk_type, 0)
        received = []

        def capture(model, contents):  # noqa: ARG001
            received.append(contents)
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks([chunk])

        assert received[0][0].inline_data.mime_type == expected_mime

    def test_one_api_call_per_chunk(self, chunk_type, expected_mime):
        chunks = [make_media_chunk(chunk_type, i) for i in range(3)]
        call_sizes: list[int] = []

        def capture(model, contents):  # noqa: ARG001
            call_sizes.append(len(contents))
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks(chunks)

        assert sorted(call_sizes) == [1, 1, 1]


# ---------------------------------------------------------------------------
# embed_chunks — document chunks — batched up to 100 per call
# ---------------------------------------------------------------------------

class TestEmbedChunksDocument:
    def test_embedding_added(self):
        chunk = make_media_chunk("document", 0)
        with patch.object(emb._client.models, "embed_content", side_effect=capture_parts_call):
            result = emb.embed_chunks([chunk])
        assert "embedding" in result[0]

    def test_correct_content_type_sent(self):
        # Documents are embedded as text strings (not PDF bytes) for exact
        # text-to-text alignment with query embeddings.
        chunk = {"type": "document", "text": "hello world", "source": "file_0", "chunk_index": 0}
        received = []

        def capture(model, contents):  # noqa: ARG001
            received.append(contents)
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks([chunk])

        assert isinstance(received[0], list)
        assert isinstance(received[0][0], str)

    def test_batched_in_single_call(self):
        """3 document chunks must be sent in one API call, not three."""
        chunks = [make_media_chunk("document", i) for i in range(3)]
        call_sizes: list[int] = []

        def capture(model, contents):  # noqa: ARG001
            call_sizes.append(len(contents))
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks(chunks)

        assert call_sizes == [3]

    def test_batch_splits_at_document_batch_size(self):
        """batch_size+1 document chunks must produce exactly 2 API calls."""
        n = emb._DOCUMENT_BATCH_SIZE + 1
        chunks = [make_media_chunk("document", i) for i in range(n)]
        call_sizes: list[int] = []

        def capture(model, contents):  # noqa: ARG001
            call_sizes.append(len(contents))
            return embed_response(fake_vector())

        with patch.object(emb._client.models, "embed_content", side_effect=capture):
            emb.embed_chunks(chunks)

        assert sorted(call_sizes) == [1, emb._DOCUMENT_BATCH_SIZE]


# ---------------------------------------------------------------------------
# embed_chunks — mixed modalities
# ---------------------------------------------------------------------------

class TestEmbedChunksMixed:
    def test_all_modalities_get_embeddings(self):
        chunks = [
            make_image_chunk(0),
            make_media_chunk("video", 1),
            make_media_chunk("audio", 2),
            make_media_chunk("document", 3),
        ]

        with patch.object(emb._client.models, "embed_content", side_effect=capture_parts_call):
            result = emb.embed_chunks(chunks)

        assert all("embedding" in c for c in result)
