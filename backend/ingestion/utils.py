"""
Shared ingestion helpers used across chunking modules.
"""

import io

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


def text_to_pdf(text: str) -> bytes:
    """Render plain text into a minimal single-page PDF using reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 72.0
    max_width = width - 2 * margin
    y = height - margin
    line_height = 14.0
    c.setFont("Helvetica", 10)
    for line in text.split("\n"):
        wrapped = simpleSplit(line, "Helvetica", 10, max_width) or [""]
        for wl in wrapped:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin
            c.drawString(margin, y, wl)
            y -= line_height
    c.save()
    return buf.getvalue()
