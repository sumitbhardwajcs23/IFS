"""Phase 1: PDF text extraction with font-aware Kruti Dev decoding and Tesseract OCR fallback."""

from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image
import pytesseract
from tqdm import tqdm

from src.config import MIN_PAGE_CHARS_FOR_NATIVE_TEXT, OCR_DPI, OCR_LANGUAGES
from src.krutidev import krutidev_to_unicode

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
PUA_OR_REPLACEMENT_RE = re.compile(r"[\uE000-\uF8FF\uFFFD]")

# Strong Kruti Dev 010 byte/character indicators (never occur in normal English legal prose)
_KD_MARKER_CHARS_RE = re.compile(r"[¼½åŸçè¡¿À¾ØÝØÜÑÒÓÔÖ]")
_KD_COMMON_TOKENS_RE = re.compile(
    r"(?:^|\s)(?:ds|esa|dh|dk|ls|gS|gSa|vkSj|mŸkj|mÙkj|izns'k|xtV|Hkkx|foKfIr|fu;qfDr|"
    r"vf/kdkj|la\[;k|la0|iz;kxjkt|i`\"B|jftLVMZ|ykblsUl|vuqHkkx|fu;e|vkns'k)(?=\s|$|[,\.\]\):-])"
)
_ENGLISH_COMMON_WORDS_RE = re.compile(
    r"\b(?:the|and|for|with|that|this|from|under|section|rule|rules|act|order|"
    r"notification|department|government|court|district|india|uttar|pradesh|"
    r"gazette|dated|shall|hereby|appointed|secretary|general|september|october|"
    r"november|december|january|february|march|april|august)\b",
    re.IGNORECASE,
)


def find_tesseract_binary() -> str | None:
    """Locate Tesseract OCR executable if installed on the system."""
    in_path = shutil.which("tesseract")
    if in_path:
        return in_path

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def is_krutidev_sample(text: str) -> bool:
    """Heuristically decide whether a font's aggregate text is Kruti Dev encoded."""
    if not text or DEVANAGARI_RE.search(text):
        return False

    name_lower = text.strip()
    if not name_lower:
        return False

    kd_markers = len(_KD_MARKER_CHARS_RE.findall(text))
    kd_tokens = len(_KD_COMMON_TOKENS_RE.findall(text))
    en_words = len(_ENGLISH_COMMON_WORDS_RE.findall(text))

    # Punctuation embedded inside alphabetic sequences (e.g. la[;k, izns'k, dk;Z, i`"B)
    midword_punct = len(re.findall(r"[A-Za-z][\[\];'\"@~>][A-Za-z]", text))

    kd_score = kd_markers * 3 + kd_tokens * 4 + midword_punct * 2
    en_score = en_words * 3
    return kd_score > en_score and kd_score >= 3


def classify_document_fonts(doc: pymupdf.Document, max_sample_chars_per_font: int = 8000) -> dict[str, str]:
    """Classify each embedded font in a PDF as 'krutidev' or 'latin_or_unicode'."""
    font_buffers: dict[str, list[str]] = defaultdict(list)
    font_lengths: dict[str, int] = defaultdict(int)

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_name = span.get("font", "")
                    if font_lengths[font_name] >= max_sample_chars_per_font:
                        continue
                    txt = span.get("text", "")
                    if txt.strip():
                        font_buffers[font_name].append(txt)
                        font_lengths[font_name] += len(txt)

    classification: dict[str, str] = {}
    for font_name, chunks in font_buffers.items():
        combined = " ".join(chunks)
        name_hint = font_name.lower()
        if "kruti" in name_hint or "devlys" in name_hint or is_krutidev_sample(combined):
            classification[font_name] = "krutidev"
        else:
            classification[font_name] = "latin_or_unicode"
    return classification


def _needs_ocr(text: str) -> bool:
    """Return True if a page lacks usable native text (likely scanned or unmapped glyphs)."""
    compact = re.sub(r"\s+", "", text)
    if len(compact) < MIN_PAGE_CHARS_FOR_NATIVE_TEXT:
        return True
    garbled = len(PUA_OR_REPLACEMENT_RE.findall(compact))
    return (garbled / max(1, len(compact))) > 0.15


def _extract_native_page_text(
    page: pymupdf.Page,
    font_classes: dict[str, str],
) -> tuple[str, bool]:
    """Extract reading-order text from a page, decoding Kruti Dev spans per font."""
    page_dict = page.get_text("dict")
    text_blocks = [
        b for b in page_dict.get("blocks", []) if b.get("type", 0) == 0 and b.get("lines")
    ]
    if not text_blocks:
        return page.get_text("text").strip(), False

    # Sort blocks in reading order: top-to-bottom (4pt bands), then left-to-right
    text_blocks.sort(key=lambda b: (round(b["bbox"][1] / 4.0), b["bbox"][0]))

    used_krutidev = False
    block_strings: list[str] = []

    for block in text_blocks:
        line_strings: list[str] = []
        for line in block.get("lines", []):
            span_parts: list[str] = []
            for span in line.get("spans", []):
                raw_span = span.get("text", "")
                if not raw_span:
                    continue
                font_name = span.get("font", "")
                if font_classes.get(font_name) == "krutidev":
                    used_krutidev = True
                    span_parts.append(krutidev_to_unicode(raw_span))
                else:
                    span_parts.append(raw_span)
            joined_line = "".join(span_parts).strip()
            if joined_line:
                line_strings.append(joined_line)
        if line_strings:
            block_strings.append("\n".join(line_strings))

    return "\n".join(block_strings).strip(), used_krutidev


def _ocr_page(page: pymupdf.Page, dpi: int = OCR_DPI, languages: str = OCR_LANGUAGES) -> str:
    """Render a PDF page to an image and run Tesseract OCR."""
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    try:
        return pytesseract.image_to_string(image, lang=languages).strip()
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(image, lang="eng").strip()


def extract_pdf(
    pdf_path: Path,
    doc_id: str | None = None,
    tesseract_cmd: str | None = None,
    enable_ocr: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract page-level text and provenance from a single PDF."""
    resolved_id = doc_id or pdf_path.stem
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    ocr_available = enable_ocr and bool(tesseract_cmd)
    pages_out: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {
        "pymupdf": 0,
        "pymupdf_krutidev": 0,
        "ocr_tesseract": 0,
        "pymupdf_low_text_no_ocr": 0,
        "empty": 0,
    }

    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        pdf_metadata = doc.metadata or {}
        font_classes = classify_document_fonts(doc)

        for idx, page in enumerate(doc, start=1):
            native_text, used_krutidev = _extract_native_page_text(page, font_classes)
            image_count = len(page.get_images(full=False))
            method = "pymupdf_krutidev" if used_krutidev else "pymupdf"
            final_text = native_text

            if _needs_ocr(native_text):
                if ocr_available:
                    ocr_text = _ocr_page(page)
                    if len(ocr_text.strip()) > len(native_text.strip()):
                        final_text = ocr_text
                        method = "ocr_tesseract"
                    elif not native_text.strip():
                        method = "empty"
                else:
                    method = "empty" if not native_text.strip() else "pymupdf_low_text_no_ocr"

            method_counts[method] = method_counts.get(method, 0) + 1
            pages_out.append(
                {
                    "doc_id": resolved_id,
                    "filename": pdf_path.name,
                    "page_number": idx,
                    "extraction_method": method,
                    "raw_char_count": len(final_text),
                    "image_count": image_count,
                    "has_devanagari": bool(DEVANAGARI_RE.search(final_text)),
                    "raw_text": final_text,
                }
            )

    krutidev_fonts = sorted(k for k, v in font_classes.items() if v == "krutidev")
    english_fonts = sorted(k for k, v in font_classes.items() if v != "krutidev")

    summary = {
        "doc_id": resolved_id,
        "filename": pdf_path.name,
        "page_count": page_count,
        "pdf_title": (pdf_metadata.get("title") or "").strip(),
        "pdf_author": (pdf_metadata.get("author") or "").strip(),
        "krutidev_fonts": krutidev_fonts,
        "english_fonts": english_fonts,
        "method_counts": method_counts,
        "total_raw_chars": sum(p["raw_char_count"] for p in pages_out),
        "devanagari_pages": sum(1 for p in pages_out if p["has_devanagari"]),
    }
    return pages_out, summary


def extract_corpus(
    pdf_paths: list[Path],
    enable_ocr: bool = True,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract all PDFs in `pdf_paths` and return combined page records and summary report."""
    tesseract_cmd = find_tesseract_binary() if enable_ocr else None
    all_pages: list[dict[str, Any]] = []
    doc_summaries: list[dict[str, Any]] = []

    iterator = tqdm(pdf_paths, desc="Extracting PDFs", unit="pdf") if show_progress else pdf_paths
    for pdf_path in iterator:
        pages, summary = extract_pdf(
            pdf_path=pdf_path,
            doc_id=pdf_path.stem,
            tesseract_cmd=tesseract_cmd,
            enable_ocr=enable_ocr,
        )
        all_pages.extend(pages)
        doc_summaries.append(summary)

    report = {
        "total_pdfs": len(pdf_paths),
        "total_pages": len(all_pages),
        "tesseract_available": bool(tesseract_cmd),
        "tesseract_path": tesseract_cmd or "",
        "documents": doc_summaries,
    }
    return all_pages, report
