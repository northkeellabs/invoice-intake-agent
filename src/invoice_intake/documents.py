"""Files in, text out: validation (extension + magic bytes + size + page cap), PDF text layer, page rendering
(pypdfium2, permissive licence), and OCR (RapidOCR, ONNX) for images and PDF pages without a text layer."""
from __future__ import annotations

import io
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from .normalize import alnum

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 10
MAX_PIXELS = 40_000_000
RENDER_DPI = 150
MIN_TEXT_LAYER_CHARS = 20  # fewer letters/digits than this on a PDF page = no usable text layer -> OCR
MAGIC = {".pdf": (b"%PDF-",), ".png": (b"\x89PNG\r\n\x1a\n",), ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",)}
KIND = {".pdf": "pdf", ".png": "image", ".jpg": "image", ".jpeg": "image"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class FileRejected(Exception):
    pass


@dataclass
class OcrLine:
    text: str
    score: float


@dataclass
class PageText:
    page: int
    text: str
    source: str  # "text_layer" | "ocr"
    confidence: float  # 1.0 for a text layer; length-weighted mean RapidOCR score for OCR
    lines: list[OcrLine] = field(default_factory=list)


def validate_bytes(data: bytes, filename: str) -> tuple[str, int]:
    """Return (kind, pages) or raise FileRejected. Used for CLI files and webhook uploads alike."""
    ext = Path(filename).suffix.lower()
    if ext not in MAGIC:
        raise FileRejected(f"extension {ext or '(none)'} not allowed (pdf, png, jpg)")
    if len(data) > MAX_BYTES:
        raise FileRejected(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    if len(data) == 0:
        raise FileRejected("empty file")
    if not any(data.startswith(m) for m in MAGIC[ext]):
        raise FileRejected(f"content does not match the {ext} extension (magic bytes)")
    if KIND[ext] == "pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise FileRejected("encrypted PDF")
            pages = len(reader.pages)
        except FileRejected:
            raise
        except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
            raise FileRejected(f"unreadable PDF ({type(exc).__name__})") from None
        if pages == 0:
            raise FileRejected("PDF has no pages")
        if pages > MAX_PAGES:
            raise FileRejected(f"PDF has {pages} pages (limit {MAX_PAGES})")
        return "pdf", pages
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise FileRejected("image too large")
            if getattr(im, "n_frames", 1) > 1:
                raise FileRejected("multi-frame images are not accepted")
    except FileRejected:
        raise
    except Exception as exc:  # Pillow raises many types for broken images
        raise FileRejected(f"unreadable image ({type(exc).__name__})") from None
    return "image", 1


def validate_file(path: Path) -> tuple[str, int]:
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise FileRejected(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    return validate_bytes(path.read_bytes(), path.name)


def render_pages(path: Path, kind: str, dpi: int = RENDER_DPI) -> list[Image.Image]:
    if kind == "image":
        with Image.open(path) as im:
            return [ImageOps.exif_transpose(im).convert("RGB")]
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(min(len(pdf), MAX_PAGES))]
    finally:
        pdf.close()


class OcrEngine(Protocol):
    name: str

    def lines(self, image: Image.Image) -> list[tuple]:
        """(text, score, x_left, y_center, height[, x_center, angle_radians]) per detected text box."""
        ...


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self):
        self._engine = None

    def lines(self, image: Image.Image):
        import numpy as np
        if self._engine is None:
            import logging
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        r = self._engine(np.array(image.convert("RGB")))
        if r.boxes is None or not r.txts:
            return []
        out = []
        for box, txt, score in zip(r.boxes, r.txts, r.scores):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            # box points: top-left, top-right, bottom-right, bottom-left; the top edge gives the text slope
            angle = math.atan2(box[1][1] - box[0][1], (box[1][0] - box[0][0]) or 1e-6)
            height = math.hypot(box[3][0] - box[0][0], box[3][1] - box[0][1])
            out.append((str(txt), float(score), float(min(xs)), float(sum(ys) / 4), float(height),
                        float(sum(xs) / 4), float(angle)))
        return out


def ocr_page(engine: OcrEngine, image: Image.Image, page: int) -> PageText:
    raw = engine.lines(image)
    if not raw:
        return PageText(page=page, text="", source="ocr", confidence=0.0)
    boxes = [tuple(b) + ((b[2], 0.0) if len(b) == 5 else ()) for b in raw]  # (text, score, x0, yc, h, xc, angle)
    # Deskew: on a page photographed or scanned at a slight angle, one text row drifts up or down across the
    # page. Project every box centre onto the page's median text slope before grouping boxes into rows.
    slope = math.tan(statistics.median(b[6] for b in boxes))
    boxes = [(*b[:3], b[3] - b[5] * slope, *b[4:]) for b in boxes]
    med_h = statistics.median(b[4] for b in boxes) or 10.0
    boxes.sort(key=lambda b: (b[3], b[2]))
    rows: list[list[tuple]] = []
    for b in boxes:  # group boxes into reading-order rows by (deskewed) vertical centre
        if rows and abs(rows[-1][0][3] - b[3]) < 0.5 * med_h:
            rows[-1].append(b)
        else:
            rows.append([b])
    lines = []
    for row in rows:
        row.sort(key=lambda b: b[2])
        lines.append(OcrLine(text="   ".join(b[0] for b in row),
                             score=round(sum(b[1] for b in row) / len(row), 4)))
    total = sum(len(b[0]) for b in boxes) or 1
    conf = sum(b[1] * len(b[0]) for b in boxes) / total
    return PageText(page=page, text="\n".join(l.text for l in lines), source="ocr", confidence=round(conf, 4),
                    lines=lines)


def pdf_text_layer(path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    out = []
    for p in reader.pages[:MAX_PAGES]:
        try:
            t = p.extract_text(extraction_mode="layout") or ""
        except Exception:  # layout mode can fail on odd PDFs; plain mode is the fallback
            t = p.extract_text() or ""
        out.append(t)
    return out


def get_text(path: Path, kind: str, engine: OcrEngine, images: list[Image.Image] | None = None) -> list[PageText]:
    """Per-page text: the PDF text layer where it exists, otherwise OCR of the rendered page."""
    images = images if images is not None else render_pages(path, kind)
    if kind == "image":
        return [ocr_page(engine, images[0], 1)]
    pages: list[PageText] = []
    for i, t in enumerate(pdf_text_layer(path), start=1):
        if len(alnum(t)) >= MIN_TEXT_LAYER_CHARS:
            pages.append(PageText(page=i, text=t, source="text_layer", confidence=1.0))
        else:
            pages.append(ocr_page(engine, images[i - 1], i))
    return pages
