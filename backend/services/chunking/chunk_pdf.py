"""
PDF chunking — converts a PDF file into chunk dicts.

Uses PyMuPDF (fitz) for parsing and tiktoken for token counting.

Chunk shape:
    type="document", text, source, page, chunk_index,
    modality="pdf", section_heading, document_title
"""

from pathlib import Path
from typing import Optional

import fitz
import tiktoken
from loguru import logger

import backend.config as config

_enc = tiktoken.get_encoding("cl100k_base")
_PDF_CHUNK_SIZE: int = config.PDF_CHUNK_SIZE
_PDF_CHUNK_OVERLAP: int = config.PDF_CHUNK_OVERLAP



def _get_overlap_paras(
    paras: list[tuple[int, str, str]],
    overlap_tokens: int,
) -> list[tuple[int, str, str]]:
    """Return the trailing paragraphs whose tokens sum to <= overlap_tokens."""
    result: list[tuple[int, str, str]] = []
    count = 0
    for item in reversed(paras):
        pt = len(_enc.encode(item[1]))
        if count + pt > overlap_tokens:
            break
        result.insert(0, item)
        count += pt
    return result


def chunk_pdf(filepath: str) -> list[dict]:
    """
    Parse a PDF and return chunk dicts ready for embed_chunks / add_chunks.

    Step 1 — Parse: extract title, TOC, and per-page text via PyMuPDF.
    Step 2 — Chunk: sliding window over paragraphs; reset at TOC boundaries,
      100-token overlap at mid-section splits, hard ceiling of 800 tokens.
    Step 3 — Return chunk dicts with all required keys.
    """
    doc = fitz.open(filepath)
    title: str = (doc.metadata.get("title") or "").strip() or Path(filepath).stem

    # Build sorted (page_num, heading) list from TOC (1-indexed page numbers)
    toc = doc.get_toc()  # list of [level, title, page_num]
    toc_entries: list[tuple[int, str]] = sorted(
        [(pn, t) for _, t, pn in toc], key=lambda x: x[0]
    )

    def get_heading(page_num: int) -> str:
        """Return the most recent TOC heading at or before page_num."""
        heading = ""
        for p, h in toc_entries:
            if p <= page_num:
                heading = h
            else:
                break
        return heading

    # ── Collect (page_num, paragraph, heading) tuples from all pages ─────────
    all_paras: list[tuple[int, str, str]] = []
    num_pages = len(doc)

    for page_idx in range(num_pages):
        page_num = page_idx + 1
        heading = get_heading(page_num)
        try:
            page_text = doc[page_idx].get_text()
        except Exception as exc:
            logger.warning(
                f"chunk_pdf: failed to extract page {page_num} from {filepath}: {exc}"
            )
            continue

        logger.info(f"chunk_pdf: page {page_num}/{num_pages} — {Path(filepath).name}")

        if not page_text.strip():
            logger.warning(f"chunk_pdf: page {page_num} is empty, skipping")
            continue

        for raw_para in page_text.split("\n\n"):
            para = raw_para.strip()
            if para:
                all_paras.append((page_num, para, heading))

    doc.close()

    # ── Greedy chunking with TOC-aware boundaries ─────────────────────────────
    chunks: list[dict] = []
    chunk_index = 0
    current: list[tuple[int, str, str]] = []  # (page_num, para_text, heading)
    current_tokens = 0
    prev_heading: Optional[str] = None

    # These tokens never appear in real prose — any occurrence means the chunk
    # is from a diagram or attention visualization, not readable text.
    _JUNK_TOKENS: frozenset[str] = frozenset({"<EOS>", "<pad>", "<unk>", "<s>", "</s>", "<mask>"})

    def flush(paras: list[tuple[int, str, str]]) -> None:
        """Build a chunk dict from paras and append to chunks."""
        nonlocal chunk_index
        if not paras:
            return
        chunk_text = "\n\n".join(p for _, p, _ in paras)

        # Skip chunks containing any special tokens (e.g. attention visualizations)
        if any(tok in chunk_text for tok in _JUNK_TOKENS):
            logger.warning(
                f"chunk_pdf: skipping chunk {chunk_index} — "
                f"contains special tokens (page {paras[0][0]})"
            )
            chunk_index += 1
            return

        page_num = paras[0][0]
        section_heading = paras[0][2]
        chunks.append(
            {
                "type": "document",
                "text": chunk_text,
                "source": filepath,
                "page": page_num,
                "chunk_index": chunk_index,
                "modality": "pdf",
                "section_heading": section_heading,
                "document_title": title,
            }
        )
        chunk_index += 1

    for page_num, para, heading in all_paras:
        para_token_ids = _enc.encode(para)
        para_token_count = len(para_token_ids)

        # Clean TOC section boundary → flush without overlap
        is_new_section = prev_heading is not None and heading != prev_heading
        if is_new_section and current:
            flush(current)
            current = []
            current_tokens = 0

        prev_heading = heading

        # Mega-paragraph: exceeds hard ceiling on its own → hard-split with overlap
        if para_token_count > _PDF_CHUNK_SIZE:
            if current:
                flush(current)
                current = []
                current_tokens = 0
            start = 0
            while start < para_token_count:
                end = min(start + _PDF_CHUNK_SIZE, para_token_count)
                sub_text = _enc.decode(para_token_ids[start:end])
                flush([(page_num, sub_text, heading)])
                next_start = end - _PDF_CHUNK_OVERLAP if end < para_token_count else end
                start = max(next_start, start + 1)  # guard against infinite loop
            continue

        # Adding this paragraph would overflow → flush with overlap
        if current_tokens + para_token_count > _PDF_CHUNK_SIZE and current:
            flush(current)
            current = _get_overlap_paras(current, _PDF_CHUNK_OVERLAP)
            current_tokens = sum(len(_enc.encode(p)) for _, p, _ in current)

        current.append((page_num, para, heading))
        current_tokens += para_token_count

    if current:
        flush(current)

    logger.info(
        f"chunk_pdf: produced {len(chunks)} chunks from {Path(filepath).name}"
    )
    return chunks
