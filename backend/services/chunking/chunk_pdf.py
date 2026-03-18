"""
PDF chunking — converts a PDF file into chunk dicts.

Uses PyMuPDF (fitz) for parsing and tiktoken for token counting.

Chunk shape:
    type="document", text, source, page, chunk_index,
    modality="pdf", section_heading, document_title
"""

import re
from pathlib import Path
from typing import Optional

import fitz
import tiktoken
from loguru import logger

import backend.config as config

_enc = tiktoken.get_encoding("cl100k_base")
_PDF_CHUNK_SIZE: int = config.PDF_CHUNK_SIZE
_PDF_CHUNK_OVERLAP: int = config.PDF_CHUNK_OVERLAP



def _clean_pdf_text(text: str) -> str:
    """Clean common PyMuPDF extraction artifacts from PDF table text."""
    replacements = [
        # Superscript exponent artifacts from ReportLab standard fonts
        ("10II",  "10^-9"),
        ("10■■",  "10^-9"),
        ("ε=",    "epsilon="),
        ("ɛ=",    "epsilon="),
        # Whitespace artifacts
        ("\x00",  ""),
        ("\xa0",  " "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


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
    try:
        doc = fitz.open(filepath)
    except Exception as exc:
        logger.error(f"chunk_pdf: fitz failed to open {filepath!r}: {exc}")
        raise ValueError(f"Could not open PDF '{Path(filepath).name}': {exc}") from exc

    title: str = (doc.metadata.get("title") or "").strip() or Path(filepath).stem

    # Build TOC title set for paragraph-level heading detection.
    # Using a set allows O(1) lookup per paragraph.
    toc = doc.get_toc()  # list of [level, title, page_num]
    toc_title_set: set[str] = {t for _, t, _ in toc}

    # ── Collect (page_num, paragraph, heading) tuples from all pages ─────────
    # Track current_heading paragraph-by-paragraph so that two sections on the
    # same page are distinguished correctly: when a paragraph's text exactly
    # matches a TOC title, it marks a section boundary and is itself skipped
    # (it is a label, not body content).
    all_paras: list[tuple[int, str, str]] = []
    num_pages = len(doc)
    current_heading: str = ""

    for page_idx in range(num_pages):
        page_num = page_idx + 1
        try:
            # Use "blocks" mode so each text block is a discrete unit.
            # In "text" mode fitz joins blocks with single \n, causing heading
            # blocks to merge into surrounding content when we split on \n\n.
            raw_blocks: list = doc[page_idx].get_text("blocks")
        except Exception as exc:
            logger.warning(
                f"chunk_pdf: failed to extract page {page_num} from {filepath}: {exc}"
            )
            continue

        logger.info(f"chunk_pdf: page {page_num}/{num_pages} — {Path(filepath).name}")

        if not raw_blocks:
            logger.warning(f"chunk_pdf: page {page_num} is empty, skipping")
            continue

        for block in raw_blocks:
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image — skip image blocks
            if block[6] != 0:
                continue
            para = _clean_pdf_text(block[4]).strip()
            if not para:
                continue
            # If this block IS a TOC heading title, update current_heading
            # and skip it — it is a section label, not body content.
            if para in toc_title_set:
                current_heading = para
                continue
            all_paras.append((page_num, para, current_heading))

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
        # Use word-boundary check so "<s>elf-attention" is not falsely matched.
        if any(re.search(rf'(?<!\w){re.escape(tok)}(?!\w)', chunk_text)
               for tok in _JUNK_TOKENS):
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

        # Clean TOC section boundary → flush, then carry overlap into new section
        is_new_section = prev_heading is not None and heading != prev_heading
        if is_new_section and current:
            flush(current)
            # Keep overlap but correct their heading to the NEW section
            overlap = _get_overlap_paras(current, _PDF_CHUNK_OVERLAP)
            current = [(pn, para, heading) for pn, para, _ in overlap]
            current_tokens = sum(len(_enc.encode(p)) for _, p, _ in current)

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
