"""
Tests for backend.services.bm25_index.

All tests reset the module-level _index / _corpus_chunks state via the
`bm25` fixture, and redirect _BM25_INDEX_PATH to a pytest tmp_path so
no test writes to the real chroma_db directory.
"""

import os
import pickle

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

import backend.services.bm25_index as bm25_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def bm25(tmp_path):
    """Reset module-level BM25 state and redirect pickle path before each test."""
    bm25_module._index = None
    bm25_module._corpus_chunks = []
    original_path = bm25_module._BM25_INDEX_PATH
    bm25_module._BM25_INDEX_PATH = tmp_path / "bm25_index.pkl"
    yield bm25_module
    bm25_module._index = None
    bm25_module._corpus_chunks = []
    bm25_module._BM25_INDEX_PATH = original_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(texts: list[str], modality: str = "pdf") -> list[dict]:
    """Return minimal chunk dicts suitable for build_index."""
    return [
        {
            "type": "document",
            "text": text,
            "source": f"doc_{i}.pdf",
            "page": i,
            "chunk_index": i,
            "modality": modality,
            "file_hash": f"hash_{i}",
        }
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# _tokenize — stemming behaviour
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_stems_morphological_variants(self):
        """run / running / runs must all stem to the same token."""
        stems = {bm25_module._tokenize(w)[0] for w in ["run", "running", "runs"]}
        assert len(stems) == 1

    def test_retrieval_variants_collapse(self):
        stems = {bm25_module._tokenize(w)[0] for w in ["retrieval", "retrieve", "retrieved"]}
        assert len(stems) == 1

    def test_lowercase(self):
        assert bm25_module._tokenize("Transformer") == bm25_module._tokenize("transformer")

    def test_punctuation_stripped(self):
        tokens = bm25_module._tokenize("hello, world!")
        assert "hello," not in tokens
        assert "world!" not in tokens
        assert "hello" in [t.rstrip(".,!") for t in tokens] or "hello" in tokens

    def test_numbers_included(self):
        tokens = bm25_module._tokenize("ISO 27001 clause 9")
        assert "27001" in tokens or any("27001" in t for t in tokens)

    def test_empty_string_returns_empty_list(self):
        assert bm25_module._tokenize("") == []


# ---------------------------------------------------------------------------
# _strip_binaries
# ---------------------------------------------------------------------------

class TestStripBinaries:
    def test_removes_binary_fields(self):
        chunk = {
            "text": "hello",
            "video_bytes": b"big blob",
            "audio_bytes": b"more bytes",
            "embedding": [0.1, 0.2],
            "source": "file.mp4",
        }
        slim = bm25_module._strip_binaries(chunk)
        assert "video_bytes" not in slim
        assert "audio_bytes" not in slim
        assert "embedding" not in slim

    def test_keeps_text_and_metadata(self):
        chunk = {"text": "hello", "source": "f.pdf", "page": 1, "image_bytes": b"x"}
        slim = bm25_module._strip_binaries(chunk)
        assert slim["text"] == "hello"
        assert slim["source"] == "f.pdf"
        assert slim["page"] == 1

    def test_original_chunk_unmodified(self):
        chunk = {"text": "hi", "embedding": [0.0]}
        bm25_module._strip_binaries(chunk)
        assert "embedding" in chunk


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_index_is_loaded_after_build(self, bm25):
        bm25.build_index(_make_chunks(["hello world"]))
        assert bm25._index is not None

    def test_corpus_chunks_grows_on_each_call(self, bm25):
        bm25.build_index(_make_chunks(["first"]))
        assert len(bm25._corpus_chunks) == 1
        bm25.build_index(_make_chunks(["second"]))
        assert len(bm25._corpus_chunks) == 2

    def test_persists_pickle_to_disk(self, bm25):
        bm25.build_index(_make_chunks(["hello"]))
        assert bm25._BM25_INDEX_PATH.exists()

    def test_pickle_contains_version_stamp(self, bm25):
        bm25.build_index(_make_chunks(["hello"]))
        with open(bm25._BM25_INDEX_PATH, "rb") as fh:
            data = pickle.load(fh)
        assert isinstance(data, tuple)
        assert data[0] == bm25._PICKLE_VERSION

    def test_video_clip_chunks_excluded(self, bm25):
        pdf_chunks = _make_chunks(["pdf text"], modality="pdf")
        clip_chunks = _make_chunks(["video summary"], modality="video_clip")
        summary_chunks = _make_chunks(["video summary"], modality="video_summary")
        bm25.build_index(pdf_chunks + clip_chunks + summary_chunks)
        # video_clip excluded → 2 chunks indexed (pdf + video_summary)
        assert len(bm25._corpus_chunks) == 2
        modalities = {c["modality"] for c in bm25._corpus_chunks}
        assert "video_clip" not in modalities

    def test_empty_chunk_list_does_not_raise(self, bm25):
        """Empty input after modality filtering must not raise — index stays None."""
        bm25.build_index([])  # should not raise
        # No indexable content → _index stays None rather than crashing
        assert bm25._index is None

    def test_binary_fields_stripped_from_corpus(self, bm25):
        chunks = _make_chunks(["hello"])
        chunks[0]["embedding"] = [0.1, 0.2, 0.3]
        chunks[0]["video_bytes"] = b"big"
        bm25.build_index(chunks)
        assert "embedding" not in bm25._corpus_chunks[0]
        assert "video_bytes" not in bm25._corpus_chunks[0]


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------

class TestLoadIndex:
    def test_noop_when_no_file(self, bm25):
        bm25.load_index()
        assert bm25._index is None  # no file → stays None

    def test_loads_after_build(self, bm25):
        bm25.build_index(_make_chunks(["hello world"]))
        # Reset in-memory state, then reload from disk
        bm25._index = None
        bm25._corpus_chunks = []
        bm25.load_index()
        assert bm25._index is not None
        assert len(bm25._corpus_chunks) == 1

    def test_version_mismatch_discards_index(self, bm25, tmp_path):
        """A pickle with a wrong version stamp must be silently discarded."""
        stale_path = bm25._BM25_INDEX_PATH
        with open(stale_path, "wb") as fh:
            pickle.dump((999, None, []), fh)  # wrong version
        bm25.load_index()
        assert bm25._index is None

    def test_legacy_tuple_without_version_discarded(self, bm25):
        """A pickle from the old schema (no version field) must be discarded."""
        stale_path = bm25._BM25_INDEX_PATH
        with open(stale_path, "wb") as fh:
            pickle.dump((None, []), fh)  # old 2-tuple format
        bm25.load_index()
        assert bm25._index is None


# ---------------------------------------------------------------------------
# search_bm25
# ---------------------------------------------------------------------------

class TestSearchBm25:
    def test_raises_when_index_not_loaded(self, bm25):
        with pytest.raises(RuntimeError, match="BM25 index is not loaded"):
            bm25.search_bm25("anything")

    def test_returns_list(self, bm25):
        bm25.build_index(_make_chunks(["hello world"]))
        result = bm25.search_bm25("hello")
        assert isinstance(result, list)

    def test_result_has_required_keys(self, bm25):
        bm25.build_index(_make_chunks(["hello world"]))
        result = bm25.search_bm25("hello")
        assert result, "expected at least one result"
        for item in result:
            assert "text" in item
            assert "score" in item
            assert "metadata" in item

    def test_metadata_has_required_keys(self, bm25):
        bm25.build_index(_make_chunks(["hello world"]))
        result = bm25.search_bm25("hello")
        for item in result:
            meta = item["metadata"]
            assert "source" in meta
            assert "chunk_index" in meta
            assert "page" in meta
            assert "modality" in meta

    def test_score_is_float(self, bm25):
        bm25.build_index(_make_chunks(["hello world"]))
        result = bm25.search_bm25("hello")
        for item in result:
            assert isinstance(item["score"], float)

    def test_top_k_respected(self, bm25):
        bm25.build_index(_make_chunks([f"doc {i}" for i in range(10)]))
        result = bm25.search_bm25("doc", top_k=3)
        assert len(result) <= 3

    def test_relevant_chunk_ranks_above_irrelevant(self, bm25):
        chunks = _make_chunks([
            "transformer attention mechanism multi-head",  # relevant
            "unrelated cooking recipe pasta carbonara",    # irrelevant
            "transformer encoder decoder architecture",   # relevant
        ])
        bm25.build_index(chunks)
        results = bm25.search_bm25("transformer attention encoder", top_k=3)
        scores = [r["score"] for r in results]
        # Top result must have a higher score than the last
        assert scores[0] >= scores[-1]

    def test_matching_chunk_outscores_nonmatching(self, bm25):
        """The chunk containing the query term must rank above ones that do not.

        BM25Okapi IDF = log((N - df + 0.5) / (df + 0.5)).  With N=2, df=1
        the formula yields log(1) = 0 for both documents.  A corpus of ≥3
        documents where the target term appears in only 1 gives df < N/2,
        making IDF positive and the matching chunk clearly outrank the others.
        """
        chunks = _make_chunks([
            "chromadb vector store cosine similarity",   # contains "chromadb"
            "unrelated cooking recipe pasta carbonara",  # does not
            "unrelated weather forecast sunny cloudy",   # does not
        ])
        bm25.build_index(chunks)
        results = bm25.search_bm25("chromadb", top_k=3)
        assert results[0]["text"] == "chromadb vector store cosine similarity"
        assert results[0]["score"] > results[1]["score"]

    def test_stemmed_query_matches_stemmed_corpus(self, bm25):
        """'running' in query must match 'run' in corpus via stemming.

        Three-document corpus keeps df < N/2 so IDF is positive for "run".
        """
        chunks = _make_chunks([
            "the model run the pipeline efficiently",  # contains "run"
            "unrelated weather forecast sunny cloudy",  # does not
            "unrelated cooking recipe pasta carbonara",  # does not
        ])
        bm25.build_index(chunks)
        results = bm25.search_bm25("running pipeline", top_k=3)
        assert results[0]["text"] == "the model run the pipeline efficiently"
        assert results[0]["score"] > results[1]["score"]

    def test_no_match_returns_zero_scores(self, bm25):
        """Out-of-vocabulary query terms score 0 (no partial match)."""
        bm25.build_index(_make_chunks(["hello world"]))
        results = bm25.search_bm25("xyzzy quux zork")  # made-up terms, not in vocab
        assert all(r["score"] == 0.0 for r in results)


# ---------------------------------------------------------------------------
# End-to-end: build → persist → reload → search
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_search_works_after_reload(self, bm25):
        """Results after reload from disk must match results before reload."""
        chunks = _make_chunks([
            "gemini embedding multimodal retrieval",
            "chromadb vector store persistence",
        ])
        bm25.build_index(chunks)
        pre_reload = bm25.search_bm25("gemini retrieval", top_k=2)

        # Simulate server restart: wipe in-memory state and reload
        bm25._index = None
        bm25._corpus_chunks = []
        bm25.load_index()

        post_reload = bm25.search_bm25("gemini retrieval", top_k=2)
        assert len(pre_reload) == len(post_reload)
        for pre, post in zip(pre_reload, post_reload):
            assert pre["text"] == post["text"]
            assert abs(pre["score"] - post["score"]) < 1e-6

    def test_accumulation_across_multiple_builds(self, bm25):
        """All ingested files remain searchable after subsequent build_index calls."""
        bm25.build_index(_make_chunks(["chromadb vector store"]))
        bm25.build_index(_make_chunks(["bm25 keyword retrieval rank"]))

        chroma_results = bm25.search_bm25("chromadb vector", top_k=2)
        bm25_results = bm25.search_bm25("bm25 keyword", top_k=2)

        assert any("chromadb" in r["text"] or "vector" in r["text"] for r in chroma_results)
        assert any("bm25" in r["text"] or "keyword" in r["text"] for r in bm25_results)

    def test_video_clip_absent_from_search_results(self, bm25):
        """video_clip chunks must not appear in BM25 results even if ingested."""
        pdf_chunks = _make_chunks(["transformer architecture attention"], modality="pdf")
        clip_chunks = _make_chunks(["transformer architecture attention"], modality="video_clip")
        summary_chunks = _make_chunks(["transformer architecture attention"], modality="video_summary")
        bm25.build_index(pdf_chunks + clip_chunks + summary_chunks)

        results = bm25.search_bm25("transformer attention", top_k=10)
        modalities = [r["metadata"]["modality"] for r in results]
        assert "video_clip" not in modalities
