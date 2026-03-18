"""
Generate transformer_technical_report.pdf with proper ReportLab bookmarks
so fitz.get_toc() returns all 7 section entries.

All content matches the original report; only the structure (TOC bookmarks)
and epsilon/superscript rendering are changed.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable


# ─── Bookmark flowable ────────────────────────────────────────────────────────

class _Bookmark(Flowable):
    """Invisible flowable that registers a PDF outline/bookmark entry."""

    def __init__(self, title: str, key: str, level: int = 0) -> None:
        super().__init__()
        self.title = title
        self.key = key
        self.level = level
        self.width = 0
        self.height = 0

    def draw(self) -> None:
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, self.level, closed=False)


# ─── Document setup ───────────────────────────────────────────────────────────

def _make_doc(output_path: str) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        title="Transformer Technical Report",
        author="Research Team",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    return doc


# ─── Styles ───────────────────────────────────────────────────────────────────

def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "rpt_title",
            parent=base["Title"],
            fontSize=20,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "rpt_h1",
            parent=base["Heading1"],
            fontSize=14,
            spaceBefore=18,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "rpt_body",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "rpt_caption",
            parent=base["BodyText"],
            fontSize=9,
            leading=12,
            spaceAfter=6,
            textColor=colors.HexColor("#444444"),
        ),
    }


# ─── Table style helper ───────────────────────────────────────────────────────

_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
    ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
    ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",   (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
])


# ─── Content builders ─────────────────────────────────────────────────────────

def _build_story(s: dict) -> list:
    story = []

    # ── Cover / title ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Transformer Technical Report", s["title"]))
    story.append(Paragraph(
        "Attention Is All You Need — An Engineering Deep Dive", s["body"]
    ))
    story.append(Spacer(1, 0.25 * inch))

    # ── Section 1: Background and Motivation ──────────────────────────────────
    story.append(_Bookmark("Background and Motivation", "sec_background", level=0))
    story.append(Paragraph("Background and Motivation", s["h1"]))
    story.append(Paragraph(
        "Sequence transduction models have traditionally relied on recurrent or "
        "convolutional architectures. Recurrent models process tokens sequentially, "
        "which fundamentally limits parallelisation during training and creates "
        "difficulty modelling long-range dependencies due to vanishing gradients. "
        "Convolutional approaches address some parallelism concerns but require "
        "O(log n) or O(n) operations to relate distant positions depending on the "
        "kernel structure.",
        s["body"],
    ))
    story.append(Paragraph(
        "The Transformer architecture was introduced to address both limitations "
        "simultaneously. By replacing recurrence entirely with attention mechanisms, "
        "the model achieves O(1) path length between any two positions while "
        "supporting fully parallel computation across the sequence during training. "
        "This design choice proved transformative: models trained with the Transformer "
        "architecture achieved state-of-the-art results on machine translation "
        "benchmarks in a fraction of the training time previously required.",
        s["body"],
    ))
    story.append(Paragraph(
        "Prior work on attention mechanisms had focused on using them as supplements "
        "to recurrent networks — to allow the decoder to attend back to encoder "
        "hidden states. The key insight in the Transformer is that attention alone, "
        "without any recurrence or convolution, is sufficient to learn powerful "
        "sequence representations.",
        s["body"],
    ))

    # ── Section 2: The Transformer Architecture ───────────────────────────────
    story.append(PageBreak())
    story.append(_Bookmark("The Transformer Architecture", "sec_architecture", level=0))
    story.append(Paragraph("The Transformer Architecture", s["h1"]))
    story.append(Paragraph(
        "The Transformer follows an encoder-decoder structure. The encoder maps an "
        "input sequence (x_1, ..., x_n) to a sequence of continuous representations "
        "z = (z_1, ..., z_n). Given z, the decoder generates an output sequence "
        "(y_1, ..., y_m) one symbol at a time, consuming previously generated symbols "
        "as additional input at each step (auto-regressive).",
        s["body"],
    ))
    story.append(Paragraph(
        "Both encoder and decoder are composed of N=6 identical layers. Each encoder "
        "layer has two sub-layers: (1) a multi-head self-attention mechanism, and "
        "(2) a position-wise fully connected feed-forward network. Each decoder layer "
        "adds a third sub-layer — multi-head cross-attention over the encoder output. "
        "Residual connections and layer normalisation are applied around each sub-layer.",
        s["body"],
    ))
    story.append(Paragraph(
        "Positional encodings are added to the input embeddings to inject sequence "
        "order information since the architecture contains no recurrence or "
        "convolution. The encodings use sine and cosine functions of different "
        "frequencies, allowing the model to generalise to longer sequences than "
        "those seen during training.",
        s["body"],
    ))
    story.append(Paragraph(
        "The feed-forward sub-layers use a two-layer network with a ReLU activation "
        "and an inner dimensionality of d_ff = 2048, giving each position its own "
        "independently parameterised non-linear transformation. Weight sharing does "
        "not occur across positions, but the same parameters are used across layers "
        "of the same type.",
        s["body"],
    ))

    # ── Section 3: Attention Mechanism ────────────────────────────────────────
    story.append(PageBreak())
    story.append(_Bookmark("Attention Mechanism", "sec_attention", level=0))
    story.append(Paragraph("Attention Mechanism", s["h1"]))
    story.append(Paragraph(
        "The core operation is Scaled Dot-Product Attention. Given queries Q, keys K, "
        "and values V, the output is computed as:",
        s["body"],
    ))
    story.append(Paragraph(
        "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V",
        s["body"],
    ))
    story.append(Paragraph(
        "The scaling factor 1/sqrt(d_k) prevents the dot products from growing large "
        "in magnitude and pushing the softmax into regions with very small gradients. "
        "For large d_k the unscaled dot products can have variance d_k, so dividing "
        "by sqrt(d_k) normalises the variance to 1.",
        s["body"],
    ))
    story.append(Paragraph(
        "Multi-Head Attention projects Q, K, V into h=8 parallel lower-dimensional "
        "spaces (d_k = d_v = d_model/h = 64), computes attention in each head "
        "independently, then concatenates and projects the results. This allows the "
        "model to jointly attend to information from different representation "
        "subspaces at different positions, something a single attention head cannot "
        "do without averaging.",
        s["body"],
    ))
    story.append(Paragraph(
        "Three types of attention are used in the Transformer: encoder self-attention "
        "(all keys, values, and queries come from the encoder output), "
        "decoder self-attention (masked to prevent attending to subsequent positions, "
        "preserving the auto-regressive property), and encoder-decoder cross-attention "
        "(queries from the decoder, keys and values from the encoder).",
        s["body"],
    ))

    # ── Section 4: Training Configuration ────────────────────────────────────
    story.append(PageBreak())
    story.append(_Bookmark("Training Configuration", "sec_training", level=0))
    story.append(Paragraph("Training Configuration", s["h1"]))
    story.append(Paragraph(
        "Models were trained on the WMT 2014 English-German and English-French "
        "datasets. For English-German the training set consists of approximately "
        "4.5 million sentence pairs encoded using byte-pair encoding with a shared "
        "source-target vocabulary of approximately 37,000 tokens. For English-French "
        "a significantly larger dataset of 36 million sentences is used with a 32,000 "
        "word-piece vocabulary.",
        s["body"],
    ))
    story.append(Paragraph(
        "Training used 8 NVIDIA P100 GPUs. The base model was trained for 100,000 "
        "steps (approximately 12 hours) and the big model for 300,000 steps "
        "(3.5 days). A custom learning rate schedule was used: the rate increases "
        "linearly for the first warmup_steps=4000 training steps, then decreases "
        "proportionally to the inverse square root of the step number.",
        s["body"],
    ))
    story.append(Paragraph(
        "The optimizer is Adam with beta1=0.9, beta2=0.98, and epsilon=1e-9. "
        "Residual dropout (P_drop=0.1) was applied to the output of each sub-layer "
        "before adding the residual connection, as well as to the sums of embeddings "
        "and positional encodings in both encoder and decoder stacks.",
        s["body"],
    ))

    # ── Table 1: Training config ───────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Table 1: Hyperparameter Configuration", s["caption"]))
    table1_data = [
        ["Parameter",         "Base Model",   "Big Model"],
        ["d_model",           "512",          "1024"],
        ["d_ff",              "2048",         "4096"],
        ["Num heads (h)",     "8",            "16"],
        ["d_k = d_v",         "64",           "64"],
        ["Encoder layers (N)", "6",           "6"],
        ["Decoder layers (N)", "6",           "6"],
        ["Dropout",           "0.1",          "0.3"],
        ["Optimizer",         "Adam",         "Adam"],
        ["beta1",             "0.9",          "0.9"],
        ["beta2",             "0.98",         "0.98"],
        ["epsilon",           "epsilon=1e-9", "epsilon=1e-9"],
        ["warmup_steps",      "4000",         "4000"],
        ["Training steps",    "100,000",      "300,000"],
        ["Training time",     "~12 hours",    "~3.5 days"],
        ["GPUs",              "8x P100",      "8x P100"],
    ]
    t1 = Table(table1_data, colWidths=[2.2 * inch, 2 * inch, 2 * inch])
    t1.setStyle(_TABLE_STYLE)
    story.append(t1)
    story.append(Spacer(1, 10))

    # ── Section 5: Benchmark Results ─────────────────────────────────────────
    story.append(PageBreak())
    story.append(_Bookmark("Benchmark Results", "sec_benchmarks", level=0))
    story.append(Paragraph("Benchmark Results", s["h1"]))
    story.append(Paragraph(
        "The Transformer big model achieves 28.4 BLEU on the WMT 2014 "
        "English-to-German translation task, surpassing all previously reported "
        "models including ensembles, by more than 2 BLEU. On WMT 2014 "
        "English-to-French, the big model achieves 41.0 BLEU, outperforming all "
        "previously published single models, at less than one-quarter the training "
        "cost of the previous state-of-the-art.",
        s["body"],
    ))
    story.append(Paragraph(
        "The base model also surpasses all previous single models at a fraction of "
        "the training cost: 27.3 BLEU on EN-DE and 38.1 BLEU on EN-FR. These results "
        "demonstrate that the Transformer achieves better quality while being "
        "significantly more efficient to train.",
        s["body"],
    ))

    # ── Table 2: Benchmark BLEU ────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Table 2: Machine Translation Benchmark Results", s["caption"]))
    table2_data = [
        ["Model",                 "EN-DE BLEU", "EN-FR BLEU", "Train Cost (FLOPs)"],
        ["ByteNet",               "23.75",      "—",          "—"],
        ["Deep-Att + PosUnk Ens", "—",          "39.2",       "—"],
        ["GNMT + RL (Ensemble)",  "26.30",      "41.16",      "1.4 x 10^20"],
        ["ConvS2S (Ensemble)",    "26.36",      "41.29",      "1.5 x 10^20"],
        ["MoE",                   "26.03",      "40.56",      "1.2 x 10^20"],
        ["Transformer (base)",    "27.3",       "38.1",       "3.3 x 10^18"],
        ["Transformer (big)",     "28.4",       "41.0",       "2.3 x 10^19"],
    ]
    t2 = Table(table2_data, colWidths=[2.2 * inch, 1.3 * inch, 1.3 * inch, 1.6 * inch])
    t2.setStyle(_TABLE_STYLE)
    story.append(t2)
    story.append(Spacer(1, 10))

    # ── Section 6: Ablation Study ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_Bookmark("Ablation Study", "sec_ablation", level=0))
    story.append(Paragraph("Ablation Study", s["h1"]))
    story.append(Paragraph(
        "A systematic ablation study was conducted to evaluate the contribution of "
        "different architectural components. All ablation experiments use the base "
        "model trained on WMT 2014 EN-DE and evaluated on newstest2013 development "
        "set. Results are reported as BLEU scores.",
        s["body"],
    ))
    story.append(Paragraph(
        "Varying the number of attention heads shows that h=8 performs best. "
        "Single-head attention (h=1) achieves only 23.3 BLEU, while too many heads "
        "(h=32) also degrades performance to 25.6 BLEU, likely due to the reduced "
        "d_k=16 dimension per head. The optimal d_k=64 with h=8 achieves 25.8 BLEU.",
        s["body"],
    ))
    story.append(Paragraph(
        "Replacing positional encodings (sinusoidal) with learned positional "
        "embeddings produces nearly identical results (25.7 vs 25.8 BLEU), "
        "suggesting the specific form of positional encoding matters less than its "
        "presence. Removing positional information entirely drops performance "
        "substantially.",
        s["body"],
    ))

    # ── Table 3: Ablation ──────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Table 3: Ablation Results on EN-DE newstest2013", s["caption"]))
    table3_data = [
        ["Variant",                           "d_k", "d_v", "Heads", "BLEU"],
        ["Base model",                        "64",  "64",  "8",     "25.8"],
        ["h=1 (single head)",                 "512", "512", "1",     "23.3"],
        ["h=4",                               "128", "128", "4",     "25.5"],
        ["h=16",                              "32",  "32",  "16",    "25.7"],
        ["h=32",                              "16",  "16",  "32",    "25.6"],
        ["d_k=16 (reduced key dim)",          "16",  "64",  "8",     "25.3"],
        ["No positional encoding",            "64",  "64",  "8",     "22.1"],
        ["Learned positional embeddings",     "64",  "64",  "8",     "25.7"],
        ["No dropout",                        "64",  "64",  "8",     "25.3"],
        ["Dropout=0.2",                       "64",  "64",  "8",     "25.6"],
        ["No label smoothing",                "64",  "64",  "8",     "25.4"],
        ["Label smoothing eps=0.2",           "64",  "64",  "8",     "25.7"],
    ]
    t3 = Table(table3_data, colWidths=[2.6 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch])
    t3.setStyle(_TABLE_STYLE)
    story.append(t3)
    story.append(Spacer(1, 10))

    # ── Section 7: Impact and Legacy ─────────────────────────────────────────
    story.append(PageBreak())
    story.append(_Bookmark("Impact and Legacy", "sec_impact", level=0))
    story.append(Paragraph("Impact and Legacy", s["h1"]))
    story.append(Paragraph(
        "The Transformer architecture has become the dominant foundation for "
        "language models across virtually all NLP tasks. BERT, GPT, T5, and their "
        "successors all build directly on the original Transformer design, "
        "demonstrating that the core architectural decisions — multi-head "
        "self-attention, positional encoding, and layer normalisation — generalise "
        "far beyond machine translation.",
        s["body"],
    ))
    story.append(Paragraph(
        "The attention mechanism's quadratic complexity O(n^2) in sequence length "
        "remains the primary scalability bottleneck, spurring a significant line of "
        "research into efficient attention variants: sparse attention (Longformer, "
        "BigBird), linear attention approximations (Performer, Linformer), and "
        "hardware-optimised implementations (FlashAttention).",
        s["body"],
    ))
    story.append(Paragraph(
        "The model's strong inductive bias toward learning global dependencies "
        "from scratch — rather than incorporating locality priors through convolution "
        "or recurrence — also proved surprisingly effective in vision (ViT), "
        "audio, and multimodal settings, cementing the Transformer as a truly "
        "general-purpose sequence modelling primitive.",
        s["body"],
    ))
    story.append(Paragraph(
        "As of 2024, the largest language models contain hundreds of billions of "
        "parameters and are trained on trillions of tokens, yet their core "
        "computation remains the scaled dot-product attention introduced in the "
        "original paper — a testament to the elegance and scalability of the design.",
        s["body"],
    ))

    return story


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    import os
    output = os.path.join(os.path.dirname(__file__), "transformer_technical_report.pdf")
    doc = _make_doc(output)
    styles = _styles()
    story = _build_story(styles)
    doc.build(story)
    print(f"Written: {output}")

    # Verify TOC with fitz
    import fitz  # type: ignore
    d = fitz.open(output)
    toc = d.get_toc()
    d.close()
    print(f"fitz.get_toc() returned {len(toc)} entries:")
    for entry in toc:
        print(" ", entry)
    if not toc:
        raise RuntimeError("TOC is empty — bookmarks were not written correctly!")


if __name__ == "__main__":
    main()
