"""
Tests for POST /query — orchestration logic in backend.routers.query.

All three services (embeddings, vectorstore, generation) are mocked so
these tests run without any real Gemini calls or ChromaDB I/O.

Covers:
  - Happy path: question → embedding → chunks → answer
  - Empty vectorstore: no chunks → canned "no information" response
  - Service failure: embed_text raises → 502
  - Service failure: vectorstore.search raises → 502
  - Service failure: generate_answer raises → 502
  - Input validation: missing question field → 422
  - Input validation: empty string question → passes (model decides)
  - Response shape: all required fields present and correctly typed
  - chunks_used matches sources list length
  - model field reflects config.GENERATION_MODEL
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

from backend.main import app
import backend.config as config

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING = [0.1, 0.2, 0.3]

_FAKE_CHUNKS = [
    {"text": "Paris is the capital of France.", "source": "geography.pdf", "page": 1, "score": 0.95},
    {"text": "France is in Western Europe.",    "source": "geography.pdf", "page": 2, "score": 0.88},
]

_FAKE_GENERATION_RESULT = {
    "answer": "Paris is the capital of France.",
    "sources": [
        {"filename": "geography.pdf", "page": 1, "score": 0.95},
        {"filename": "geography.pdf", "page": 2, "score": 0.88},
    ],
    "chunks_used": 2,
    "model": config.GENERATION_MODEL,
}


def _patches(embed_return=None, search_return=None, gen_return=None):
    """Return a context-manager stack that patches all three service calls."""
    return (
        patch("backend.routers.query.embeddings.embed_text",
              return_value=embed_return if embed_return is not None else _FAKE_EMBEDDING),
        patch("backend.routers.query.vectorstore.search_balanced",
              return_value=search_return if search_return is not None else _FAKE_CHUNKS),
        patch("backend.routers.query.generation.generate_answer",
              return_value=gen_return if gen_return is not None else _FAKE_GENERATION_RESULT),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestQueryHappyPath:
    def test_returns_200(self):
        with _patches()[0], _patches()[1], _patches()[2]:
            with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
                 patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
                 patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
                resp = client.post("/query/", json={"question": "What is the capital of France?"})
        assert resp.status_code == 200

    def test_response_contains_answer(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            resp = client.post("/query/", json={"question": "What is the capital of France?"})
        data = resp.json()
        assert data["answer"] == "Paris is the capital of France."

    def test_response_shape(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            resp = client.post("/query/", json={"question": "What is the capital of France?"})
        data = resp.json()
        assert set(data.keys()) == {"answer", "sources", "chunks_used", "model"}

    def test_sources_list_shape(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            resp = client.post("/query/", json={"question": "What is the capital of France?"})
        sources = resp.json()["sources"]
        assert len(sources) == 2
        for s in sources:
            assert set(s.keys()) == {"filename", "page", "score"}
            assert isinstance(s["filename"], str)
            assert isinstance(s["page"], int)
            assert isinstance(s["score"], float)

    def test_chunks_used_matches_sources_count(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            resp = client.post("/query/", json={"question": "What is the capital of France?"})
        data = resp.json()
        assert data["chunks_used"] == len(data["sources"])

    def test_model_field_matches_config(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            resp = client.post("/query/", json={"question": "What is the capital of France?"})
        assert resp.json()["model"] == config.GENERATION_MODEL

    def test_embed_text_called_with_question(self):
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDING)
        with patch("backend.routers.query.embeddings.embed_text", mock_embed), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            client.post("/query/", json={"question": "What is the capital of France?"})
        mock_embed.assert_called_once_with("What is the capital of France?")

    def test_search_balanced_called_with_embedding(self):
        mock_search = MagicMock(return_value=_FAKE_CHUNKS)
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", mock_search), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            client.post("/query/", json={"question": "What is the capital of France?"})
        mock_search.assert_called_once_with(_FAKE_EMBEDDING)

    def test_generate_answer_called_with_question_and_chunks(self):
        mock_gen = MagicMock(return_value=_FAKE_GENERATION_RESULT)
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", mock_gen):
            client.post("/query/", json={"question": "What is the capital of France?"})
        mock_gen.assert_called_once_with("What is the capital of France?", _FAKE_CHUNKS)


# ---------------------------------------------------------------------------
# Edge case: no chunks found (empty vectorstore)
# ---------------------------------------------------------------------------

class TestQueryNoChunks:
    def test_returns_200_when_no_chunks(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=[]):
            resp = client.post("/query/", json={"question": "Who invented the internet?"})
        assert resp.status_code == 200

    def test_canned_answer_when_no_chunks(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=[]):
            resp = client.post("/query/", json={"question": "Who invented the internet?"})
        data = resp.json()
        assert "don't have enough information" in data["answer"]

    def test_empty_sources_when_no_chunks(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=[]):
            resp = client.post("/query/", json={"question": "Who invented the internet?"})
        assert resp.json()["sources"] == []

    def test_chunks_used_zero_when_no_chunks(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=[]):
            resp = client.post("/query/", json={"question": "Who invented the internet?"})
        assert resp.json()["chunks_used"] == 0

    def test_generation_not_called_when_no_chunks(self):
        mock_gen = MagicMock()
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=[]), \
             patch("backend.routers.query.generation.generate_answer", mock_gen):
            client.post("/query/", json={"question": "Who invented the internet?"})
        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Service failure: 502 error paths
# ---------------------------------------------------------------------------

class TestQueryServiceFailures:
    def test_503_when_embed_text_raises(self):
        with patch("backend.routers.query.embeddings.embed_text",
                   side_effect=RuntimeError("Gemini down")):
            resp = client.post("/query/", json={"question": "test"})
        assert resp.status_code == 503
        assert "embed" in resp.json()["detail"].lower()

    def test_502_when_vectorstore_search_raises(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced",
                   side_effect=RuntimeError("ChromaDB unavailable")):
            resp = client.post("/query/", json={"question": "test"})
        assert resp.status_code == 502
        assert "search" in resp.json()["detail"].lower()

    def test_502_when_generation_raises(self):
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer",
                   side_effect=RuntimeError("LLM quota exceeded")):
            resp = client.post("/query/", json={"question": "test"})
        assert resp.status_code == 502
        assert "generation" in resp.json()["detail"].lower()

    def test_vectorstore_not_called_when_embed_fails(self):
        mock_search = MagicMock()
        with patch("backend.routers.query.embeddings.embed_text",
                   side_effect=RuntimeError("Gemini down")), \
             patch("backend.routers.query.vectorstore.search_balanced", mock_search):
            client.post("/query/", json={"question": "test"})
        mock_search.assert_not_called()

    def test_generation_not_called_when_search_fails(self):
        mock_gen = MagicMock()
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced",
                   side_effect=RuntimeError("ChromaDB unavailable")), \
             patch("backend.routers.query.generation.generate_answer", mock_gen):
            client.post("/query/", json={"question": "test"})
        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestQueryInputValidation:
    def test_422_when_question_field_missing(self):
        resp = client.post("/query/", json={})
        assert resp.status_code == 422

    def test_422_when_body_is_empty(self):
        resp = client.post("/query/", content=b"", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_422_when_question_is_not_a_string(self):
        resp = client.post("/query/", json={"question": 42})
        assert resp.status_code == 422

    def test_empty_string_question_passes_validation(self):
        """Pydantic accepts an empty string — the model handles it gracefully."""
        empty_result = {**_FAKE_GENERATION_RESULT, "answer": "I don't have enough information to answer this."}
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=empty_result):
            resp = client.post("/query/", json={"question": ""})
        assert resp.status_code == 200

    def test_whitespace_question_is_passed_as_is(self):
        """Whitespace questions are forwarded verbatim — no silent stripping."""
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDING)
        with patch("backend.routers.query.embeddings.embed_text", mock_embed), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            client.post("/query/", json={"question": "   "})
        mock_embed.assert_called_once_with("   ")

    def test_question_with_special_characters(self):
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDING)
        with patch("backend.routers.query.embeddings.embed_text", mock_embed), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            client.post("/query/", json={"question": "What is 2+2? ∑ ñoño 日本語"})
        mock_embed.assert_called_once_with("What is 2+2? ∑ ñoño 日本語")

    def test_long_question_is_accepted(self):
        long_q = "word " * 500
        mock_embed = MagicMock(return_value=_FAKE_EMBEDDING)
        with patch("backend.routers.query.embeddings.embed_text", mock_embed), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=_FAKE_CHUNKS), \
             patch("backend.routers.query.generation.generate_answer", return_value=_FAKE_GENERATION_RESULT):
            resp = client.post("/query/", json={"question": long_q})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Single-chunk result
# ---------------------------------------------------------------------------

class TestQuerySingleChunk:
    def test_single_chunk_response(self):
        single_chunk = [_FAKE_CHUNKS[0]]
        single_result = {
            "answer": "Paris.",
            "sources": [{"filename": "geography.pdf", "page": 1, "score": 0.95}],
            "chunks_used": 1,
            "model": config.GENERATION_MODEL,
        }
        with patch("backend.routers.query.embeddings.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("backend.routers.query.vectorstore.search_balanced", return_value=single_chunk), \
             patch("backend.routers.query.generation.generate_answer", return_value=single_result):
            resp = client.post("/query/", json={"question": "Capital?"})
        data = resp.json()
        assert data["chunks_used"] == 1
        assert len(data["sources"]) == 1
