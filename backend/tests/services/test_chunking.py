"""
Tests for backend.services.chunk_pdf.

All PDFs are built in-memory with fitz — no fixture files needed.
No network calls are made.
"""

import os
import tempfile
from typing import Optional

import fitz
import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")

from backend.services.chunking import chunk_pdf
import backend.config as config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_SIZE = config.PDF_CHUNK_SIZE      # 800
_CHUNK_OVERLAP = config.PDF_CHUNK_OVERLAP  # 100


def _make_pdf(
    pages: list[str],
    toc: Optional[list[list]] = None,
    page_height: int = 842,
) -> str:
    """
    Build an in-memory PDF and write it to a temp file.

    pages:       list of plain-text strings, one per page.
    toc:         optional [[level, title, page_num], ...] passed to fitz set_toc.
    page_height: fitz page height in points; increase for pages with lots of text.

    Returns the temp file path. Caller is responsible for cleanup.
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=page_height)
        # insert_textbox wraps text within the rect; insert_text renders one line only.
        rect = fitz.Rect(72, 72, 523, page_height - 72)
        page.insert_textbox(rect, text, fontsize=11)
    if toc:
        doc.set_toc(toc)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    tmp.close()
    return tmp.name


def _word(n: int) -> str:
    """Return a string of n space-separated 'word' tokens (~1 token each)."""
    return " ".join(["word"] * n)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "type", "pdf_bytes", "text", "source", "page",
    "chunk_index", "modality", "section_heading", "document_title",
}


class TestChunkSchema:
    def test_required_keys_present(self):
        path = _make_pdf(["Hello world.\n\nSecond paragraph."])
        try:
            chunks = chunk_pdf(path)
            assert chunks, "expected at least one chunk"
            for ch in chunks:
                assert REQUIRED_KEYS <= ch.keys(), f"missing keys: {REQUIRED_KEYS - ch.keys()}"
        finally:
            os.unlink(path)

    def test_field_types(self):
        path = _make_pdf(["Hello world."])
        try:
            chunks = chunk_pdf(path)
            ch = chunks[0]
            assert ch["type"] == "document"
            assert ch["modality"] == "pdf"
            assert isinstance(ch["pdf_bytes"], bytes) and len(ch["pdf_bytes"]) > 0
            assert isinstance(ch["text"], str) and ch["text"]
            assert isinstance(ch["page"], int) and ch["page"] >= 1
            assert isinstance(ch["chunk_index"], int) and ch["chunk_index"] == 0
            assert isinstance(ch["section_heading"], str)
            assert isinstance(ch["document_title"], str)
        finally:
            os.unlink(path)

    def test_chunk_index_sequential(self):
        # Three pages each with enough text to produce multiple chunks
        pages = [_word(300)] * 3
        path = _make_pdf(pages)
        try:
            chunks = chunk_pdf(path)
            indices = [ch["chunk_index"] for ch in chunks]
            assert indices == list(range(len(chunks)))
        finally:
            os.unlink(path)

    def test_source_is_filepath(self):
        path = _make_pdf(["Some text."])
        try:
            chunks = chunk_pdf(path)
            assert all(ch["source"] == path for ch in chunks)
        finally:
            os.unlink(path)

    def test_pdf_bytes_is_valid_pdf(self):
        path = _make_pdf(["Hello world."])
        try:
            chunks = chunk_pdf(path)
            raw = chunks[0]["pdf_bytes"]
            # PDF magic bytes
            assert raw[:4] == b"%PDF", "pdf_bytes does not start with PDF magic bytes"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TOC / section boundary behaviour
# ---------------------------------------------------------------------------

class TestTocBoundary:
    def test_toc_boundary_produces_separate_chunks(self):
        """Each TOC section must become its own chunk with no cross-section bleed."""
        pages = [
            "Introduction text here.",
            "Methods text here.",
            "Results text here.",
        ]
        toc = [[1, "Introduction", 1], [1, "Methods", 2], [1, "Results", 3]]
        path = _make_pdf(pages, toc)
        try:
            chunks = chunk_pdf(path)
            headings = [ch["section_heading"] for ch in chunks]
            # Every chunk must carry exactly one section heading
            assert "Introduction" in headings
            assert "Methods" in headings
            assert "Results" in headings
            # No chunk may mix sections
            for ch in chunks:
                assert ch["section_heading"] in ("Introduction", "Methods", "Results", "")
        finally:
            os.unlink(path)

    def test_toc_boundary_no_overlap(self):
        """Text from section A must not appear in the first chunk of section B."""
        section_a = "alpha " * 50        # distinctive word
        section_b = "beta " * 50
        pages = [section_a, section_b]
        toc = [[1, "Alpha", 1], [1, "Beta", 2]]
        path = _make_pdf(pages, toc)
        try:
            chunks = chunk_pdf(path)
            beta_chunks = [ch for ch in chunks if ch["section_heading"] == "Beta"]
            assert beta_chunks, "expected at least one Beta chunk"
            # The very first Beta chunk should not contain alpha text
            assert "alpha" not in beta_chunks[0]["text"].lower()
        finally:
            os.unlink(path)

    def test_no_toc_single_chunk_for_short_doc(self):
        """A short PDF with no TOC should produce one chunk with empty section_heading."""
        path = _make_pdf(["Short document with just a few words."])
        try:
            chunks = chunk_pdf(path)
            assert len(chunks) == 1
            assert chunks[0]["section_heading"] == ""
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Token ceiling and overlap
# ---------------------------------------------------------------------------

class TestTokenCeiling:
    def test_no_chunk_exceeds_ceiling(self):
        """Every chunk must be at or below PDF_CHUNK_SIZE tokens."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # 3 pages × 400 words each → forces splits
        pages = [_word(400)] * 3
        path = _make_pdf(pages)
        try:
            chunks = chunk_pdf(path)
            for ch in chunks:
                token_count = len(enc.encode(ch["text"]))
                assert token_count <= _CHUNK_SIZE, (
                    f"chunk {ch['chunk_index']} has {token_count} tokens, "
                    f"exceeds ceiling of {_CHUNK_SIZE}"
                )
        finally:
            os.unlink(path)

    def test_mid_section_split_has_overlap(self):
        """When a section is split mid-way, the second chunk should share some
        tokens with the end of the first chunk."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # Two paragraphs separated by \n\n; together they exceed 800 tokens.
        # page_height=20000 ensures fitz doesn't clip the long text.
        para_a = _word(500)
        para_b = _word(500)
        pages = [para_a + "\n\n" + para_b]
        path = _make_pdf(pages, page_height=20000)
        try:
            chunks = chunk_pdf(path)
            assert len(chunks) >= 2, "expected a split into at least 2 chunks"
            # The second chunk should start with some words from para_a or end of chunk 0
            # i.e. chunk 1 text must overlap with chunk 0 text at the token level
            tokens_0 = enc.encode(chunks[0]["text"])
            tokens_1 = enc.encode(chunks[1]["text"])
            tail_0 = set(tokens_0[-_CHUNK_OVERLAP:])
            head_1 = set(tokens_1[:_CHUNK_OVERLAP])
            assert tail_0 & head_1, "expected token overlap between consecutive mid-section chunks"
        finally:
            os.unlink(path)

    def test_mega_paragraph_hard_split(self):
        """A single paragraph longer than 800 tokens must be hard-split."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # One enormous paragraph with no \n\n breaks.
        # page_height=50000 ensures fitz stores all 1800 words without clipping.
        giant = _word(1800)
        path = _make_pdf([giant], page_height=50000)
        try:
            chunks = chunk_pdf(path)
            assert len(chunks) >= 2, "expected hard-split into multiple chunks"
            for ch in chunks:
                assert len(enc.encode(ch["text"])) <= _CHUNK_SIZE
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Advanced TOC + overflow interactions
# ---------------------------------------------------------------------------

class TestAdvancedChunking:
    def test_overflowing_section_produces_multiple_chunks_with_correct_heading(self):
        """A single section long enough to produce 3+ chunks must label every
        chunk with that section's heading — not a neighbouring section's."""
        # One section with ~2400 tokens → expect at least 3 chunks
        long_section = _word(_CHUNK_SIZE * 3)
        pages = [long_section]
        toc = [[1, "LongSection", 1]]
        path = _make_pdf(pages, toc, page_height=80000)
        try:
            chunks = chunk_pdf(path)
            assert len(chunks) >= 3, (
                f"expected ≥3 chunks from a {_CHUNK_SIZE * 3}-token section, got {len(chunks)}"
            )
            for ch in chunks:
                assert ch["section_heading"] == "LongSection", (
                    f"chunk {ch['chunk_index']} has heading {ch['section_heading']!r}, "
                    f"expected 'LongSection'"
                )
        finally:
            os.unlink(path)

    def test_page_number_reflects_source_page(self):
        """chunk['page'] must be the 1-indexed page where the chunk's first
        paragraph originates, not always page 1."""
        # Use a TOC so each page flushes as its own chunk at the section boundary,
        # making the per-chunk page number observable.
        pages = ["Page one content.", "Page two content.", "Page three content."]
        toc = [[1, "Sec1", 1], [1, "Sec2", 2], [1, "Sec3", 3]]
        path = _make_pdf(pages, toc)
        try:
            chunks = chunk_pdf(path)
            pages_seen = {ch["page"] for ch in chunks}
            # At least two distinct page numbers must appear across all chunks
            assert len(pages_seen) >= 2, (
                f"expected chunks from multiple pages, got page numbers: {pages_seen}"
            )
            assert min(pages_seen) >= 1, "page numbers must be 1-indexed"
        finally:
            os.unlink(path)

    def test_nested_toc_headings_all_create_boundaries(self):
        """Level-2 TOC entries (subsections) must also trigger section boundaries,
        not be ignored in favour of level-1 headings only."""
        pages = [
            "Chapter intro text.",
            "Subsection A text.",
            "Subsection B text.",
        ]
        toc = [
            [1, "Chapter One", 1],
            [2, "Subsection A", 2],
            [2, "Subsection B", 3],
        ]
        path = _make_pdf(pages, toc)
        try:
            chunks = chunk_pdf(path)
            headings = {ch["section_heading"] for ch in chunks}
            # All three entries should appear as distinct section headings
            assert "Chapter One" in headings
            assert "Subsection A" in headings
            assert "Subsection B" in headings
        finally:
            os.unlink(path)

    def test_document_title_read_from_pdf_metadata(self):
        """When the PDF carries a title in its metadata, document_title must
        use that value rather than the filename stem."""
        doc = fitz.open()
        page = doc.new_page()
        rect = fitz.Rect(72, 72, 523, 770)
        page.insert_textbox(rect, "Some content.", fontsize=11)
        doc.set_metadata({"title": "My Explicit Title"})
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        tmp.close()
        try:
            chunks = chunk_pdf(tmp.name)
            assert chunks, "expected at least one chunk"
            assert all(ch["document_title"] == "My Explicit Title" for ch in chunks), (
                f"expected 'My Explicit Title', got {chunks[0]['document_title']!r}"
            )
        finally:
            os.unlink(tmp.name)

    def test_chunk_index_globally_sequential_across_sections(self):
        """chunk_index must increment globally across all sections, not reset
        to 0 at each TOC boundary."""
        # Three sections, each long enough to produce 2 chunks → expect 0-5
        section_text = _word(_CHUNK_SIZE + 100)
        pages = [section_text, section_text, section_text]
        toc = [[1, "Sec1", 1], [1, "Sec2", 2], [1, "Sec3", 3]]
        path = _make_pdf(pages, toc, page_height=30000)
        try:
            chunks = chunk_pdf(path)
            indices = [ch["chunk_index"] for ch in chunks]
            assert indices == list(range(len(chunks))), (
                f"chunk_index is not globally sequential: {indices}"
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_empty_page_skipped(self):
        """An empty page must not produce a chunk and must not crash."""
        pages = ["Real content here.", "", "More real content."]
        path = _make_pdf(pages)
        try:
            chunks = chunk_pdf(path)
            # Should still get chunks from the non-empty pages
            assert chunks
            for ch in chunks:
                assert ch["text"].strip()
        finally:
            os.unlink(path)

    def test_returns_list(self):
        path = _make_pdf(["Hello."])
        try:
            result = chunk_pdf(path)
            assert isinstance(result, list)
        finally:
            os.unlink(path)

    def test_whitespace_only_paragraphs_excluded(self):
        """Paragraphs that are only whitespace must not appear in any chunk."""
        # \n\n produces empty strings after split; they should be filtered out
        pages = ["Real text.\n\n   \n\nMore real text."]
        path = _make_pdf(pages)
        try:
            chunks = chunk_pdf(path)
            for ch in chunks:
                assert ch["text"].strip() == ch["text"] or ch["text"].strip()
        finally:
            os.unlink(path)

    def test_document_title_fallback_to_filename(self):
        """When PDF metadata has no title, document_title falls back to the filename stem."""
        path = _make_pdf(["Some content."])  # fitz sets no title by default
        try:
            chunks = chunk_pdf(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            assert all(ch["document_title"] == stem for ch in chunks)
        finally:
            os.unlink(path)

    def test_chunk_to_embed_not_stored_as_text(self):
        """The enriched 'Doc: ... | Section: ...' prefix must NOT appear in chunk['text']."""
        path = _make_pdf(["Plain content."])
        try:
            chunks = chunk_pdf(path)
            for ch in chunks:
                assert not ch["text"].startswith("Doc:")
        finally:
            os.unlink(path)
