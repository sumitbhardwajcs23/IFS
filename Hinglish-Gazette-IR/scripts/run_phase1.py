"""End-to-end Phase 1 CLI runner: Dataset Manifest -> PDF Extraction (+OCR) -> Cleaning -> Storage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLAUSES_JSONL_PATH,
    DOCUMENTS_JSONL_PATH,
    EXTRACTION_SUMMARY_PATH,
    MANIFEST_PATH,
    PAGES_JSONL_PATH,
    RAW_PDF_DIR,
    SQLITE_DB_PATH,
)
from src.phase1_cleaner import clean_corpus
from src.phase1_dataset import (
    dataset_target_progress,
    sync_manifest,
    update_manifest_with_cleaned_docs,
)
from src.phase1_extractor import extract_corpus
from src.storage import save_to_sqlite, write_json, write_jsonl


def run_phase1(
    raw_dir: Path = RAW_PDF_DIR,
    manifest_path: Path = MANIFEST_PATH,
    enable_ocr: bool = True,
) -> dict[str, int]:
    """Execute the full Phase 1 pipeline and return summary counts."""
    manifest_df = sync_manifest(raw_dir=raw_dir, manifest_path=manifest_path, project_root=PROJECT_ROOT)
    pdf_paths = sorted(raw_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    print(f"[Phase 1] Found {len(pdf_paths)} PDF(s) in {raw_dir}")

    raw_pages, extraction_report = extract_corpus(pdf_paths=pdf_paths, enable_ocr=enable_ocr)
    write_jsonl(PAGES_JSONL_PATH, raw_pages)
    write_json(EXTRACTION_SUMMARY_PATH, extraction_report)
    print(
        f"[Phase 1] Extracted {extraction_report['total_pages']} pages "
        f"(Tesseract OCR available: {extraction_report['tesseract_available']})"
    )

    cleaned_docs, clauses, cleaned_pages = clean_corpus(raw_pages)
    write_jsonl(DOCUMENTS_JSONL_PATH, cleaned_docs)
    write_jsonl(CLAUSES_JSONL_PATH, clauses)
    save_to_sqlite(SQLITE_DB_PATH, documents=cleaned_docs, clauses=clauses, pages=cleaned_pages)

    updated_manifest = update_manifest_with_cleaned_docs(cleaned_docs, manifest_path=manifest_path)
    progress = dataset_target_progress(updated_manifest)

    print(
        f"[Phase 1] Cleaned {len(cleaned_docs)} documents and segmented {len(clauses)} clauses.\n"
        f"[Phase 1] Saved JSONL + SQLite DB to {SQLITE_DB_PATH}\n"
        f"[Phase 1] Dataset progress toward 100-PDF target: "
        f"{progress['total_current']} / {progress['total_target']}"
    )
    return {
        "pdfs": len(pdf_paths),
        "pages": len(raw_pages),
        "documents": len(cleaned_docs),
        "clauses": len(clauses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 (Dataset, Extraction, Cleaning).")
    parser.add_argument("--raw-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--no-ocr", action="store_true", help="Disable Tesseract OCR fallback.")
    args = parser.parse_args()

    run_phase1(raw_dir=args.raw_dir, manifest_path=args.manifest, enable_ocr=not args.no_ocr)


if __name__ == "__main__":
    main()
