"""Central configuration for Phases 1-7 (Dataset, Extraction, Cleaning, Chunking, Retrieval, UI)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Phase 1 artifacts
PAGES_JSONL_PATH = EXTRACTED_DIR / "pages.jsonl"
EXTRACTION_SUMMARY_PATH = EXTRACTED_DIR / "extraction_summary.json"
DOCUMENTS_JSONL_PATH = CLEANED_DIR / "documents.jsonl"
CLAUSES_JSONL_PATH = CLEANED_DIR / "clauses.jsonl"
SQLITE_DB_PATH = CLEANED_DIR / "gazettes.db"

# Phase 2-6 artifacts
CHUNKS_JSONL_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
SEMANTIC_META_PATH = INDEX_DIR / "semantic_meta.json"

# Phase 8 evaluation artifacts
EVAL_QUERIES_PATH = EVAL_DIR / "queries_200.jsonl"
EVAL_RESULTS_PATH = EVAL_DIR / "eval_results.json"
EVAL_COMPARISON_CSV_PATH = EVAL_DIR / "comparison_table.csv"
EVAL_CATEGORY_CSV_PATH = EVAL_DIR / "per_category_metrics.csv"

# Dataset planning targets from project spec (100 PDFs total)
CATEGORY_TARGETS = {
    "Government Notification": 30,
    "Rules and Regulations": 20,
    "Ministry Order": 20,
    "Recruitment Notification": 15,
    "Other Circular": 15,
}

# Phase 1: Extraction, OCR & Cleaning thresholds
MIN_PAGE_CHARS_FOR_NATIVE_TEXT = 40
OCR_DPI = 250
OCR_LANGUAGES = "eng+hin"
MIN_CLEAN_PAGE_CHARS = 15
REPEATED_LINE_MIN_PAGE_RATIO = 0.35

# Phase 2: Clause-aware chunking parameters
MIN_CHUNK_CHARS = 140
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 180

# Phase 4-6: Retrieval & RRF parameters
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_BATCH_SIZE = 32
RRF_K = 60
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_SEMANTIC_WEIGHT = 1.0
