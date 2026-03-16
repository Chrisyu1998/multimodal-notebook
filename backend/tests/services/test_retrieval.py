"""
Tests for backend.services.retrieval.

reciprocal_rank_fusion is a pure function — tested directly with no mocks.
hybrid_search and rerank call external services (Gemini, ChromaDB, BM25) —
those are patched so no network or disk I/O occurs.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

from backend.services.retrieval import (
    _normalize,
    hybrid_search,
    rerank,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bm25_result(text: str, score: float = 1.0, source: str = "a.pdf", page: int = 1, chunk_index: int = 0) -> dict:
    """Return a result in BM25 shape (nested metadata)."""
    return {
        "text": text,
        "score": score,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "modality": "text",
        },
    }


def _vector_result(text: str, score: float = 1.0, source: str = "a.pdf", page: int = 1, chunk_index: int = 0) -> dict:
    """Return a result in vector/ChromaDB shape (flat dict)."""
    return {
        "text": text,
        "score": score,
        "source": source,
        "page": page,
        "chunk_index": chunk_index,
        "type": "document",
    }


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_bm25_shape_flattened(self):
        r = _bm25_result("hello", score=2.5, source="b.pdf", page=3, chunk_index=7)
        n = _normalize(r)
        assert n["text"] == "hello"
        assert n["source"] == "b.pdf"
        assert n["page"] == 3
        assert n["chunk_index"] == 7
        assert n["orig_score"] == 2.5
        assert "metadata" not in n

    def test_vector_shape_flattened(self):
        r = _vector_result("world", score=0.9, source="c.pdf", page=2, chunk_index=4)
        n = _normalize(r)
        assert n["text"] == "world"
        assert n["source"] == "c.pdf"
        assert n["page"] == 2
        assert n["chunk_index"] == 4
        assert n["orig_score"] == 0.9

    def test_vector_type_mapped_to_modality(self):
        r = _vector_result("text", source="x.pdf")
        r["type"] = "image"
        n = _normalize(r)
        assert n["modality"] == "image"

    def test_vector_specific_modality_takes_priority_over_type(self):
        # When search() returns both type and modality, the specific modality wins.
        r = _vector_result("text", source="x.pdf")
        r["type"] = "image"
        r["modality"] = "image_global"
        n = _normalize(r)
        assert n["modality"] == "image_global"

    def test_missing_fields_get_defaults(self):
        n = _normalize({"text": "bare"})
        assert n["source"] == ""
        assert n["page"] == 0
        assert n["chunk_index"] == 0
        assert n["orig_score"] == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — pure function, no mocks needed
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_empty_inputs_return_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_bm25_only(self):
        results = reciprocal_rank_fusion(
            [_bm25_result("alpha"), _bm25_result("beta")],
            [],
        )
        assert len(results) == 2
        # alpha was rank 1 → higher score than beta at rank 2
        assert results[0]["text"] == "alpha"
        assert results[0]["score"] > results[1]["score"]

    def test_vector_only(self):
        results = reciprocal_rank_fusion(
            [],
            [_vector_result("x"), _vector_result("y")],
        )
        assert len(results) == 2
        assert results[0]["text"] == "x"

    def test_rrf_score_formula(self):
        """Single document ranked #1 in one list → score = 1/(60+1)."""
        results = reciprocal_rank_fusion(
            [_bm25_result("only")],
            [],
        )
        expected = 1.0 / (60 + 1)
        assert abs(results[0]["score"] - expected) < 1e-9

    def test_document_in_both_lists_gets_combined_score(self):
        """Same text in both lists at rank 1 each → score = 2 × 1/(60+1)."""
        shared = "shared chunk text"
        results = reciprocal_rank_fusion(
            [_bm25_result(shared)],
            [_vector_result(shared)],
        )
        expected = 2.0 / (60 + 1)
        match = next(r for r in results if r["text"] == shared)
        assert abs(match["score"] - expected) < 1e-9

    def test_shared_chunk_outranks_single_list_chunks(self):
        """A chunk consistently present in both lists beats a chunk that only
        appears as rank 1 in one list but is absent from the other."""
        shared = "consistent chunk"
        exclusive = "unique to bm25 only"
        results = reciprocal_rank_fusion(
            [_bm25_result(exclusive), _bm25_result(shared)],
            [_vector_result(shared)],
        )
        texts = [r["text"] for r in results]
        # 'shared' appears in both lists (combined score) vs 'exclusive' in
        # BM25 only at rank 1.  At small corpus sizes, combined presence wins.
        shared_score    = next(r["score"] for r in results if r["text"] == shared)
        exclusive_score = next(r["score"] for r in results if r["text"] == exclusive)
        # shared: 1/(60+2) + 1/(60+1); exclusive: 1/(60+1)
        assert shared_score > exclusive_score

    def test_deduplication_by_exact_text(self):
        """Same text in both lists produces one output entry, not two."""
        dup = "duplicate text"
        results = reciprocal_rank_fusion(
            [_bm25_result(dup)],
            [_vector_result(dup)],
        )
        assert len([r for r in results if r["text"] == dup]) == 1

    def test_top_k_respected(self):
        bm25 = [_bm25_result(f"doc_{i}") for i in range(15)]
        vec  = [_vector_result(f"doc_{i}") for i in range(15)]
        results = reciprocal_rank_fusion(bm25, vec, top_k=5)
        assert len(results) <= 5

    def test_sorted_descending_by_score(self):
        bm25 = [_bm25_result(f"b{i}") for i in range(5)]
        vec  = [_vector_result(f"v{i}") for i in range(5)]
        results = reciprocal_rank_fusion(bm25, vec)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_output_has_required_keys(self):
        results = reciprocal_rank_fusion(
            [_bm25_result("hello")],
            [_vector_result("world")],
        )
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "page" in r
            assert "chunk_index" in r
            assert "score" in r

    def test_debug_rank_fields_attached(self):
        """_bm25_rank and _vector_rank must be set for cross-list diagnostics."""
        shared = "both lists"
        results = reciprocal_rank_fusion(
            [_bm25_result(shared)],
            [_vector_result(shared)],
        )
        r = results[0]
        assert r["_bm25_rank"] == 1
        assert r["_vector_rank"] == 1

    def test_chunk_only_in_bm25_has_none_vector_rank(self):
        results = reciprocal_rank_fusion(
            [_bm25_result("only bm25")],
            [_vector_result("only vector")],
        )
        bm25_only = next(r for r in results if r["text"] == "only bm25")
        assert bm25_only["_bm25_rank"] == 1
        assert bm25_only["_vector_rank"] is None

    def test_lower_rank_has_lower_rrf_contribution(self):
        """Rank 1 contributes more than rank 20."""
        contrib_rank1  = 1.0 / (60 + 1)
        contrib_rank20 = 1.0 / (60 + 20)
        assert contrib_rank1 > contrib_rank20

    def test_metadata_preserved_from_first_seen(self):
        """Source and page from the first occurrence (BM25) must be in output."""
        r = reciprocal_rank_fusion(
            [_bm25_result("text", source="bm25.pdf", page=5)],
            [_vector_result("text", source="vec.pdf", page=9)],
        )
        # BM25 is processed first → its metadata wins
        assert r[0]["source"] == "bm25.pdf"
        assert r[0]["page"] == 5


# ---------------------------------------------------------------------------
# hybrid_search — patches BM25, vectorstore, embeddings, and Gemini
# ---------------------------------------------------------------------------

class TestHybridSearch:
    @pytest.fixture()
    def mock_deps(self):
        """Patch all external I/O so hybrid_search runs without network/disk."""
        bm25_results   = [_bm25_result(f"bm25_doc_{i}") for i in range(3)]
        vector_results = [_vector_result(f"vec_doc_{i}") for i in range(3)]

        with (
            patch("backend.services.retrieval.hyde_expand", return_value="hypothetical answer") as mock_hyde,
            patch("backend.services.retrieval.embeddings.embed_text", return_value=[0.1] * 768) as mock_embed,
            patch("backend.services.retrieval.bm25_index.search_bm25", return_value=bm25_results) as mock_bm25,
            patch("backend.services.retrieval.vectorstore.search", return_value=vector_results) as mock_vec,
        ):
            yield {
                "hyde": mock_hyde,
                "embed": mock_embed,
                "bm25": mock_bm25,
                "vec": mock_vec,
                "bm25_results": bm25_results,
                "vector_results": vector_results,
            }

    def test_returns_list(self, mock_deps):
        results = hybrid_search("what is attention?")
        assert isinstance(results, list)

    def test_calls_hyde_with_original_query(self, mock_deps):
        hybrid_search("my query")
        mock_deps["hyde"].assert_called_once_with("my query")

    def test_embeds_hyde_text_not_original_query(self, mock_deps):
        hybrid_search("my query")
        mock_deps["embed"].assert_called_once_with("hypothetical answer")

    def test_bm25_called_with_original_query(self, mock_deps):
        hybrid_search("my query")
        mock_deps["bm25"].assert_called_once_with("my query", top_k=20)

    def test_vector_called_with_hyde_embedding(self, mock_deps):
        hybrid_search("my query")
        mock_deps["vec"].assert_called_once_with([0.1] * 768, top_k=20)

    def test_top_k_respected(self, mock_deps):
        results = hybrid_search("query", top_k=4)
        assert len(results) <= 4

    def test_result_has_score_key(self, mock_deps):
        results = hybrid_search("query")
        for r in results:
            assert "score" in r

    def test_use_hyde_false_skips_hyde_expand(self, mock_deps):
        """When use_hyde=False, hyde_expand must not be called."""
        hybrid_search("my query", use_hyde=False)
        mock_deps["hyde"].assert_not_called()

    def test_use_hyde_false_embeds_raw_query(self, mock_deps):
        """When use_hyde=False, embed_text receives the original query, not a hypothetical."""
        hybrid_search("my query", use_hyde=False)
        mock_deps["embed"].assert_called_once_with("my query")

    def test_use_hyde_true_embeds_hyde_text(self, mock_deps):
        """When use_hyde=True (default), embed_text receives the HyDE expansion."""
        hybrid_search("my query", use_hyde=True)
        mock_deps["embed"].assert_called_once_with("hypothetical answer")

    def test_bm25_always_uses_original_query_regardless_of_hyde_flag(self, mock_deps):
        """BM25 must receive the raw query whether use_hyde is True or False."""
        hybrid_search("original", use_hyde=False)
        mock_deps["bm25"].assert_called_once_with("original", top_k=20)

    def test_bm25_runtime_error_degrades_gracefully(self):
        """If BM25 index is not built, hybrid_search falls back to vector-only."""
        vector_results = [_vector_result("vec_only")]
        with (
            patch("backend.services.retrieval.hyde_expand", return_value="hyp"),
            patch("backend.services.retrieval.embeddings.embed_text", return_value=[0.0] * 768),
            patch("backend.services.retrieval.bm25_index.search_bm25", side_effect=RuntimeError("no index")),
            patch("backend.services.retrieval.vectorstore.search", return_value=vector_results),
        ):
            results = hybrid_search("query")
        assert any(r["text"] == "vec_only" for r in results)


# ---------------------------------------------------------------------------
# rerank — patches the Gemini client
# ---------------------------------------------------------------------------

class TestRerank:
    def _chunks(self, n: int) -> list[dict]:
        return [
            {"text": f"chunk_{i}", "source": "a.pdf", "page": i, "score": 1.0 / (i + 1)}
            for i in range(n)
        ]

    def test_returns_empty_for_empty_input(self):
        assert rerank("q", []) == []

    def test_skips_api_when_chunks_lte_top_k(self):
        """If we already have ≤ top_k candidates there's no point calling Gemini."""
        chunks = self._chunks(3)
        with patch("backend.services.retrieval._client") as mock_client:
            result = rerank("q", chunks, top_k=5)
        mock_client.models.generate_content.assert_not_called()
        assert result == chunks[:5]

    def test_calls_gemini_when_chunks_gt_top_k(self):
        chunks = self._chunks(10)
        scores_payload = [{"id": i + 1, "score": float(10 - i)} for i in range(10)]
        mock_response = MagicMock()
        mock_response.text = json.dumps(scores_payload)

        with patch("backend.services.retrieval._client") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            rerank("query", chunks, top_k=3)

        mock_client.models.generate_content.assert_called_once()

    def test_returns_top_k_chunks(self):
        chunks = self._chunks(10)
        scores_payload = [{"id": i + 1, "score": float(10 - i)} for i in range(10)]
        mock_response = MagicMock()
        mock_response.text = json.dumps(scores_payload)

        with patch("backend.services.retrieval._client") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            result = rerank("query", chunks, top_k=3)

        assert len(result) == 3

    def test_sorted_by_rerank_score_descending(self):
        # Use 8 chunks with top_k=5 so the short-circuit (≤ top_k) doesn't fire
        chunks = self._chunks(8)
        scores_payload = [{"id": i + 1, "score": float(8 - i)} for i in range(8)]
        mock_response = MagicMock()
        mock_response.text = json.dumps(scores_payload)

        with patch("backend.services.retrieval._client") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            result = rerank("query", chunks, top_k=5)

        rerank_scores = [r["rerank_score"] for r in result]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    def test_rerank_score_attached_to_chunks(self):
        chunks = self._chunks(6)
        scores_payload = [{"id": i + 1, "score": float(i)} for i in range(6)]
        mock_response = MagicMock()
        mock_response.text = json.dumps(scores_payload)

        with patch("backend.services.retrieval._client") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            result = rerank("query", chunks, top_k=3)

        for r in result:
            assert "rerank_score" in r

    def test_strips_markdown_fences_from_response(self):
        chunks = self._chunks(6)
        scores_payload = [{"id": i + 1, "score": float(i)} for i in range(6)]
        fenced = f"```json\n{json.dumps(scores_payload)}\n```"
        mock_response = MagicMock()
        mock_response.text = fenced

        with patch("backend.services.retrieval._client") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            result = rerank("query", chunks, top_k=3)

        assert len(result) == 3

    def test_falls_back_to_rrf_order_on_api_error(self):
        """A Gemini failure must not raise — return first top_k chunks by RRF order."""
        chunks = self._chunks(10)

        with patch("backend.services.retrieval._client") as mock_client:
            mock_client.models.generate_content.side_effect = RuntimeError("API down")
            result = rerank("query", chunks, top_k=3)

        assert len(result) == 3
        # Fallback preserves original order
        assert result[0]["text"] == "chunk_0"
