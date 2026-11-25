"""Phase 1: Gazette text cleaning, metadata extraction, and clause/section structuring."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from src.config import MIN_CLEAN_PAGE_CHARS, REPEATED_LINE_MIN_PAGE_RATIO
from src.phase1_dataset import infer_date_from_filename

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")

# Known running headers/footers & postal registration lines in UP Gazette PDFs
_HEADER_LINE_PATTERNS = [
    re.compile(r"^उत्तर प्रदेश गजट\s*,.*(?:शक संवत्|ई[0०])[\)\s\[\]\w\-कखगघ]*$"),
    re.compile(r"^\[?\s*भाग\s+[0-9०-९]+(?:\-[कखगघ])?\s*\]?$"),
    re.compile(r"^रजिस्टर्ड नं[0०]\-[^\n]+$"),
    re.compile(r"^लाइसेन्स सं[0०]\-[^\n]+$"),
    re.compile(r"^\(लाइसेन्स्ड टू पोस्ट बिदाउट प्रीपेमेन्ट\)$"),
    re.compile(r"^\d{1,4}$"),  # bare page number
]

# Hindi month names -> numeric month
_HINDI_MONTHS = {
    "जनवरी": 1,
    "फरवरी": 2,
    "मार्च": 3,
    "अप्रैल": 4,
    "मई": 5,
    "जून": 6,
    "जुलाई": 7,
    "अगस्त": 8,
    "सितम्बर": 9,
    "सितंबर": 9,
    "अक्टूबर": 10,
    "अक्तूबर": 10,
    "नवम्बर": 11,
    "नवंबर": 11,
    "दिसम्बर": 12,
    "दिसंबर": 12,
}

_DATE_HINDI_RE = re.compile(
    r"(\d{1,2})\s+(" + "|".join(_HINDI_MONTHS.keys()) + r")\s*,\s*(20\d{2})"
)
_VOLUME_ISSUE_RE = re.compile(r"खण्ड\s+(\d+).*?संख्या\s+(\d+)", re.DOTALL)

_NOTIFICATION_NO_RE = re.compile(
    r"((?:No\.|संख्या|सं[0०])\s*(?=[A-Za-z0-9I/\.\-\(\)\u0900-\u097F]*\d)[A-Za-z0-9I/\.\-\(\)\u0900-\u097F]{4,60})"
)

_DEPARTMENT_RE = re.compile(
    r"^([A-Z][A-Z\s&\(\),\-/]{4,80}(?:DEPARTMENT|VIBHAG|COMMISSION|AUTHORITY|COURT[^\n]{0,40}|UTTAR PRADESH))\s*$",
    re.MULTILINE,
)

# Boundary patterns for structured Gazette sections/clauses
_SECTION_BOUNDARY_RE = re.compile(
    r"^(?:"
    r"अनुभाग[\-\s]*[0-9]+"
    r"|Anubhag[\-\s]*[0-9]+"
    r"|SECTION[\-\s]*[0-9]+"
    r"|PART[\-\s]*[IVX0-9]+"
    r"|NOTIFICATION|अधिसूचना|शासनादेश|विज्ञप्ति"
    r"|(?:No\.|संख्या|सं[0०])\s*[A-Za-z0-9I/\.\-\(\)\u0900-\u097F]{4,}"
    r")",
    re.IGNORECASE,
)

_SUBCLAUSE_BOUNDARY_RE = re.compile(
    r"^(?:\d{1,3}[\.\-]\s+|\(\d{1,3}\)[\.\-]?\s+|\([ivxlcdm]+\)[\.\-]?\s+)",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Recruitment Notification", ["भर्ती", "चयन", "नियुक्ति", "पदोन्नति", "recruitment", "selection", "appointed", "promotion"]),
    ("Rules and Regulations", ["नियमावली", "नियम", "विनियमन", "sub-rule", "rules", "regulations", "amendment"]),
    ("Ministry Order", ["शासनादेश", "आदेश", "कार्य-विधियां", "order", "ordered", "directed"]),
    ("Government Notification", ["अधिसूचना", "notification", "act", "धारा", "section", "article-348", "land acquisition"]),
    ("Other Circular", ["परिपत्र", "विज्ञप्ति", "नोटिस", "circular", "notice", "tender", "निविदा"]),
]


def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalize Unicode to NFC, fix common Hindi abbreviations, and clean whitespace."""
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Normalize ASCII '0' used as Hindi abbreviation sign after common abbreviations
    cleaned = re.sub(r"(ई|सं|नं|रु|उ|प्र)0", r"\1०", cleaned)
    # Rejoin English words hyphenated across line breaks
    cleaned = re.sub(r"([A-Za-z]{2,})-\n([a-z]{2,})", r"\1\2", cleaned)
    # Collapse horizontal runs of spaces/tabs
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Strip trailing/leading spaces per line
    lines = [line.strip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def _detect_repeated_lines(page_texts: list[str]) -> set[str]:
    """Find short lines that repeat across >= REPEATED_LINE_MIN_PAGE_RATIO of pages."""
    if len(page_texts) < 4:
        return set()
    line_page_counts: Counter[str] = Counter()
    for text in page_texts:
        unique_lines = {
            line.strip()
            for line in text.split("\n")
            if 2 <= len(line.strip()) <= 120
        }
        line_page_counts.update(unique_lines)

    threshold = max(3, int(len(page_texts) * REPEATED_LINE_MIN_PAGE_RATIO))
    return {line for line, cnt in line_page_counts.items() if cnt >= threshold}


def clean_page_text(raw_text: str, repeated_lines: set[str] | None = None) -> str:
    """Clean a single page's extracted text by removing noise and boilerplate lines."""
    normalized = normalize_unicode_and_whitespace(raw_text)
    if not normalized:
        return ""

    repeated = repeated_lines or set()
    kept_lines: list[str] = []

    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        if stripped in repeated:
            continue
        if any(pattern.match(stripped) for pattern in _HEADER_LINE_PATTERNS):
            continue
        if stripped in {"-", "--", "---"}:
            continue
        kept_lines.append(stripped)

    result = "\n".join(kept_lines).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def infer_category(text: str) -> str:
    """Classify a Gazette document into one of the 5 project categories."""
    sample = text[:30000].lower()
    scores: dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS:
        scores[category] = sum(sample.count(kw.lower()) for kw in keywords)
    best_cat, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_cat if best_score > 0 else "Government Notification"


def extract_document_metadata(
    doc_id: str,
    filename: str,
    raw_pages: list[dict[str, Any]],
    clean_text: str,
) -> dict[str, Any]:
    """Extract structured Gazette metadata (date, issue, department, notification no., category)."""
    first_pages_raw = "\n".join(p["raw_text"] for p in raw_pages[:3])
    gazette_date = ""
    date_match = _DATE_HINDI_RE.search(first_pages_raw)
    if date_match:
        day, month_hi, year = date_match.groups()
        month_num = _HINDI_MONTHS[month_hi]
        gazette_date = f"{year}-{month_num:02d}-{int(day):02d}"
    else:
        gazette_date = infer_date_from_filename(filename)

    vol_match = _VOLUME_ISSUE_RE.search(first_pages_raw)
    if vol_match:
        vol, issue = vol_match.groups()
        title = f"उत्तर प्रदेश गजट — खण्ड {vol}, संख्या {issue} ({gazette_date})"
    else:
        first_line = next((ln.strip() for ln in clean_text.split("\n") if len(ln.strip()) > 10), filename)
        title = first_line[:160]

    dept_match = _DEPARTMENT_RE.search(clean_text[:40000])
    department = re.sub(r"\s+", " ", dept_match.group(1)).strip() if dept_match else "Government of Uttar Pradesh"

    notif_match = _NOTIFICATION_NO_RE.search(clean_text[:15000])
    notification_no = notif_match.group(1).strip() if notif_match else ""

    has_hi = bool(DEVANAGARI_RE.search(clean_text))
    has_en = bool(LATIN_WORD_RE.search(clean_text))
    if has_hi and has_en:
        language_mix = "bilingual"
    elif has_hi:
        language_mix = "hindi"
    else:
        language_mix = "english"

    return {
        "title": title,
        "category": infer_category(clean_text),
        "gazette_date": gazette_date,
        "ministry": department,
        "department": department,
        "notification_no": notification_no,
        "language_mix": language_mix,
    }


def segment_clauses(
    doc_id: str,
    cleaned_pages: list[dict[str, Any]],
    max_clause_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Split a cleaned document's pages into structured notification/clause units."""
    clauses: list[dict[str, Any]] = []
    buf_lines: list[str] = []
    cur_label = ""
    cur_heading = ""
    page_start = cleaned_pages[0]["page_number"] if cleaned_pages else 1
    page_end = page_start

    def flush() -> None:
        nonlocal buf_lines, cur_label, cur_heading, page_start, page_end
        text = "\n".join(buf_lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) >= MIN_CLEAN_PAGE_CHARS:
            idx = len(clauses) + 1
            clauses.append(
                {
                    "clause_id": f"{doc_id}__c{idx:04d}",
                    "doc_id": doc_id,
                    "clause_index": idx,
                    "clause_label": cur_label or f"Clause {idx}",
                    "heading": cur_heading or (text.split("\n")[0][:120]),
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": text,
                    "char_count": len(text),
                }
            )
        buf_lines = []

    for page in cleaned_pages:
        pno = page["page_number"]
        ptext = page["clean_text"]
        if not ptext:
            continue
        for line in ptext.split("\n"):
            stripped = line.strip()
            if not stripped:
                if buf_lines and buf_lines[-1] != "":
                    buf_lines.append("")
                continue

            buf_len = sum(len(x) + 1 for x in buf_lines)
            is_major_boundary = bool(_SECTION_BOUNDARY_RE.match(stripped))
            is_sub_boundary = buf_len >= max_clause_chars and bool(_SUBCLAUSE_BOUNDARY_RE.match(stripped))
            is_hard_split = buf_len >= max_clause_chars * 2

            if buf_lines and (is_major_boundary or is_sub_boundary or is_hard_split):
                flush()
                page_start = pno
                cur_label = stripped[:80] if is_major_boundary else ""
                cur_heading = stripped[:140]

            if not buf_lines:
                page_start = pno
                if not cur_heading:
                    cur_heading = stripped[:140]
                if is_major_boundary and not cur_label:
                    cur_label = stripped[:80]

            buf_lines.append(stripped)
            page_end = pno

    flush()
    return clauses


def clean_corpus(
    raw_pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean all extracted pages and return (cleaned_documents, clauses, cleaned_pages)."""
    pages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in raw_pages:
        pages_by_doc[page["doc_id"]].append(page)

    cleaned_documents: list[dict[str, Any]] = []
    all_clauses: list[dict[str, Any]] = []
    all_cleaned_pages: list[dict[str, Any]] = []

    for doc_id in sorted(pages_by_doc.keys()):
        doc_pages = sorted(pages_by_doc[doc_id], key=lambda p: p["page_number"])
        normalized_pages = [normalize_unicode_and_whitespace(p["raw_text"]) for p in doc_pages]
        repeated = _detect_repeated_lines(normalized_pages)

        doc_cleaned_pages: list[dict[str, Any]] = []
        for page in doc_pages:
            cleaned_text = clean_page_text(page["raw_text"], repeated_lines=repeated)
            clean_page_rec = {
                "doc_id": doc_id,
                "page_number": page["page_number"],
                "extraction_method": page["extraction_method"],
                "raw_char_count": page["raw_char_count"],
                "clean_char_count": len(cleaned_text),
                "has_devanagari": 1 if DEVANAGARI_RE.search(cleaned_text) else 0,
                "clean_text": cleaned_text,
            }
            doc_cleaned_pages.append(clean_page_rec)
            all_cleaned_pages.append(clean_page_rec)

        full_clean_text = "\n\n".join(
            p["clean_text"] for p in doc_cleaned_pages if len(p["clean_text"]) >= MIN_CLEAN_PAGE_CHARS
        )
        metadata = extract_document_metadata(
            doc_id=doc_id,
            filename=doc_pages[0]["filename"],
            raw_pages=doc_pages,
            clean_text=full_clean_text,
        )

        native_pages = sum(1 for p in doc_pages if p["extraction_method"].startswith("pymupdf"))
        ocr_pages = sum(1 for p in doc_pages if p["extraction_method"] == "ocr_tesseract")
        empty_pages = sum(1 for p in doc_pages if p["extraction_method"] == "empty")

        cleaned_documents.append(
            {
                "doc_id": doc_id,
                "filename": doc_pages[0]["filename"],
                "title": metadata["title"],
                "category": metadata["category"],
                "gazette_date": metadata["gazette_date"],
                "ministry": metadata["ministry"],
                "department": metadata["department"],
                "notification_no": metadata["notification_no"],
                "language_mix": metadata["language_mix"],
                "page_count": len(doc_pages),
                "native_text_pages": native_pages,
                "ocr_pages": ocr_pages,
                "empty_pages": empty_pages,
                "char_count": len(full_clean_text),
                "clean_text": full_clean_text,
            }
        )

        doc_clauses = segment_clauses(doc_id=doc_id, cleaned_pages=doc_cleaned_pages)
        all_clauses.extend(doc_clauses)

    return cleaned_documents, all_clauses, all_cleaned_pages
