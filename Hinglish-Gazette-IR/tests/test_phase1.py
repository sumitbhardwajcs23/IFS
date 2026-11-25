"""Tests for Phase 1: Kruti Dev decoding, dataset manifest, extraction, cleaning, and SQLite storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pymupdf

from src.krutidev import krutidev_to_unicode
from src.phase1_cleaner import clean_corpus, clean_page_text, normalize_unicode_and_whitespace
from src.phase1_dataset import dataset_target_progress, infer_date_from_filename, sync_manifest
from src.phase1_extractor import extract_pdf, is_krutidev_sample
from src.storage import save_to_sqlite


def test_krutidev_to_unicode_gazette_phrases() -> None:
    assert krutidev_to_unicode("mŸkj izns'k xtV") == "उत्तर प्रदेश गजट"
    assert krutidev_to_unicode("iz;kxjkt") == "प्रयागराज"
    assert krutidev_to_unicode("jftLVMZ") == "रजिस्टर्ड"
    assert krutidev_to_unicode("foKfIr") == "विज्ञप्ति"
    assert krutidev_to_unicode(">k¡lh y[kuÅ") == "झाँसी लखनऊ"
    assert krutidev_to_unicode("535&540") == "535-540"
    assert krutidev_to_unicode("0-01") == "0.01"


def test_is_krutidev_sample_distinguishes_english() -> None:
    assert is_krutidev_sample("mŸkj izns'k xtV] 12 flrEcj] 2026 bZ0 ¼Hkknzin 21] 1948 'kd laor~½")
    assert not is_krutidev_sample(
        "In pursuance of the provisions of clause (3) of Article-348 of the Constitution"
    )


def test_clean_page_text_strips_headers_and_normalizes() -> None:
    raw = (
        "536\n"
        "उत्तर प्रदेश गजट, 19 सितम्बर, 2026 ई0 (भाद्रपद 28, 1948 शक संवत्)\n"
        "[ भाग 1\n"
        "अनुभाग-1\n"
        "पदोन्नति adminis-\n"
        "tration order issued.\n"
    )
    cleaned = clean_page_text(raw)
    assert "536" not in cleaned
    assert "उत्तर प्रदेश गजट" not in cleaned
    assert "administration order issued." in cleaned
    assert "अनुभाग-1" in cleaned


def test_normalize_hindi_abbreviations() -> None:
    assert normalize_unicode_and_whitespace("सं0 12 ई0 उ0प्र0") == "सं० 12 ई० उ०प्र०"


def test_metadata_ignores_table_header_and_parses_multiline_issue() -> None:
    from src.phase1_cleaner import extract_document_metadata

    raw_pages = [
        {
            "page_number": 1,
            "raw_text": "खण्ड 80]\nप्रयागराज, शनिवार, 07 नवम्बर, 2026 ई०\n[संख्या 45",
        }
    ]
    clean_text = "संख्या वार्षिक\nसं० 861/96-आयुष-1-2024-66/2011\nFINANCE (GENERAL) DEPARTMENT\n"
    meta = extract_document_metadata("doc1", "doc1.pdf", raw_pages, clean_text)
    assert meta["title"] == "उत्तर प्रदेश गजट — खण्ड 80, संख्या 45 (2026-11-07)"
    assert meta["notification_no"] == "सं० 861/96-आयुष-1-2024-66/2011"
    assert meta["department"] == "FINANCE (GENERAL) DEPARTMENT"


def test_infer_date_from_filename() -> None:
    assert infer_date_from_filename("C_202609141739155022.pdf") == "2026-09-14"
    assert infer_date_from_filename("sample.pdf") == ""


def test_end_to_end_synthetic_pdf_to_sqlite(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf_path = raw_dir / "C_20261015000000.pdf"

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text(
        (72, 72),
        "GOVERNMENT OF UTTAR PRADESH\nFINANCE DEPARTMENT\nNOTIFICATION\n"
        "No. 101/2026 - Maternity benefit leave rules shall apply to all employees.\n",
    )
    p2 = doc.new_page()
    p2.insert_text(
        (72, 72),
        "2. Employees shall be entitled to 180 days of paid maternity leave subject to conditions.",
    )
    doc.save(pdf_path)
    doc.close()

    manifest = sync_manifest(raw_dir=raw_dir, manifest_path=tmp_path / "manifest.csv", project_root=tmp_path)
    assert len(manifest) == 1
    assert manifest.iloc[0]["inferred_date"] == "2026-10-15"

    pages, summary = extract_pdf(pdf_path, enable_ocr=False)
    assert summary["page_count"] == 2
    assert summary["method_counts"]["pymupdf"] == 2

    docs, clauses, clean_pages = clean_corpus(pages)
    assert len(docs) == 1
    assert docs[0]["gazette_date"] == "2026-10-15"
    assert docs[0]["language_mix"] == "english"
    assert len(clauses) >= 1
    assert any("maternity" in c["text"].lower() for c in clauses)

    db_path = tmp_path / "gazettes.db"
    save_to_sqlite(db_path, docs, clauses, clean_pages)
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0] == len(clauses)
    assert conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 2
    conn.close()

    progress = dataset_target_progress(manifest)
    assert progress["total_target"] == 100
    assert progress["total_current"] == 1
