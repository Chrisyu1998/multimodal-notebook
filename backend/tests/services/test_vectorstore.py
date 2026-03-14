"""
Tests for backend.services.vectorstore.

Each test gets a fresh in-memory ChromaDB collection via the
`store` fixture — completely isolated from the production DB.
"""

import hashlib
import math

import chromadb
import pytest

import backend.services.vectorstore as vs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_DIM = 768


def unit_vector() -> list[float]:
    """Return a unit vector of FAKE_DIM dimensions (cosine sim = 1.0 with itself)."""
    val = 1.0 / math.sqrt(FAKE_DIM)
    return [val] * FAKE_DIM


def orthogonal_vector() -> list[float]:
    """Return a vector orthogonal to unit_vector (first half +, second half -)."""
    val = 1.0 / math.sqrt(FAKE_DIM)
    half = FAKE_DIM // 2
    return [val] * half + [-val] * half


def make_chunk(text: str, source: str, page: int, chunk_index: int, embedding: list[float]) -> dict:
    return {"text": text, "source": source, "page": page, "chunk_index": chunk_index, "embedding": embedding}


def chunk_id(source: str, chunk_index: int) -> str:
    """Mirror of the private _chunk_id used in vectorstore.py."""
    return hashlib.sha256(f"{source}:{chunk_index}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixture: isolated collection
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    """
    Replace the module-level _collection with a fresh temporary one.
    The production DB is never touched.
    """
    client = chromadb.PersistentClient(path=str(tmp_path))
    isolated = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(vs, "_collection", isolated)
    return isolated


# ---------------------------------------------------------------------------
# add_chunks tests
# ---------------------------------------------------------------------------

def test_add_chunks_inserts_new_chunks(store):
    chunks = [
        make_chunk("Q1 revenue was $4.2M.", "report.pdf", 1, 0, unit_vector()),
        make_chunk("Operating costs rose 5%.", "report.pdf", 1, 1, unit_vector()),
    ]
    vs.add_chunks(chunks)
    assert store.count() == 2


def test_add_chunks_uses_correct_ids(store):
    chunks = [make_chunk("Some text.", "doc.pdf", 1, 0, unit_vector())]
    vs.add_chunks(chunks)

    expected_id = chunk_id("doc.pdf", 0)
    result = store.get(ids=[expected_id], include=[])
    assert result["ids"] == [expected_id]


def test_add_chunks_skips_duplicates(store):
    chunk = make_chunk("Duplicate text.", "doc.pdf", 1, 0, unit_vector())
    vs.add_chunks([chunk])
    vs.add_chunks([chunk])  # second call — should be a no-op
    assert store.count() == 1


def test_add_chunks_partial_dedup(store):
    """If 1 of 2 chunks already exists, only the new one is inserted."""
    chunk_a = make_chunk("First chunk.", "doc.pdf", 1, 0, unit_vector())
    chunk_b = make_chunk("Second chunk.", "doc.pdf", 2, 1, unit_vector())

    vs.add_chunks([chunk_a])
    assert store.count() == 1

    vs.add_chunks([chunk_a, chunk_b])  # chunk_a is a dup, chunk_b is new
    assert store.count() == 2


def test_add_chunks_empty_list_is_noop(store):
    vs.add_chunks([])
    assert store.count() == 0


def test_add_chunks_stores_correct_metadata(store):
    chunk = make_chunk("Some text.", "report.pdf", 3, 7, unit_vector())
    vs.add_chunks([chunk])

    result = store.get(include=["metadatas"])
    meta = result["metadatas"][0]
    assert meta["source"] == "report.pdf"
    assert meta["page"] == 3
    assert meta["chunk_index"] == 7


def test_add_chunks_stores_correct_document(store):
    chunk = make_chunk("Exact text content.", "doc.pdf", 1, 0, unit_vector())
    vs.add_chunks([chunk])

    result = store.get(include=["documents"])
    assert result["documents"][0] == "Exact text content."


# ---------------------------------------------------------------------------
# search tests
# ---------------------------------------------------------------------------

def test_search_returns_correct_fields(store):
    vs.add_chunks([make_chunk("Some text.", "doc.pdf", 1, 0, unit_vector())])
    results = vs.search(unit_vector(), top_k=1)

    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) == {"text", "source", "page", "chunk_index", "score"}


def test_search_score_is_near_one_for_identical_vector(store):
    """Searching with the same vector that was indexed should return score ≈ 1.0."""
    vs.add_chunks([make_chunk("Identical vector.", "doc.pdf", 1, 0, unit_vector())])
    results = vs.search(unit_vector(), top_k=1)

    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_search_score_is_near_zero_for_orthogonal_vector(store):
    """Searching with an orthogonal vector should return score ≈ 0.5 (cosine sim = 0)."""
    vs.add_chunks([make_chunk("Unrelated chunk.", "doc.pdf", 1, 0, unit_vector())])
    results = vs.search(orthogonal_vector(), top_k=1)

    # cosine sim of orthogonal vectors = 0 → distance = 1 → score = 0
    assert results[0]["score"] == pytest.approx(0.0, abs=0.05)


def test_search_respects_top_k(store):
    chunks = [
        make_chunk(f"Chunk {i}.", "doc.pdf", i, i, unit_vector())
        for i in range(5)
    ]
    vs.add_chunks(chunks)

    results = vs.search(unit_vector(), top_k=3)
    assert len(results) == 3


def test_search_returns_correct_text_and_metadata(store):
    chunk = make_chunk("RAG improves accuracy.", "paper.pdf", 2, 4, unit_vector())
    vs.add_chunks([chunk])
    results = vs.search(unit_vector(), top_k=1)

    r = results[0]
    assert r["text"] == "RAG improves accuracy."
    assert r["source"] == "paper.pdf"
    assert r["page"] == 2
    assert r["chunk_index"] == 4
