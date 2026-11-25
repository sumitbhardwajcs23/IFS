# Hinglish-Gazette-IR

Phonetic and Dialect-Aware Hybrid Information Retrieval System for Code-Mixed (Hinglish) Queries over Formal Government Gazettes.

## Project Structure

- `data/raw/` — Raw Government Gazette PDFs (10 initial PDFs; target 100 across 5 categories).
- `data/manifest.csv` — Dataset catalog tracking document IDs, categories, sources, page counts, and processing status.
- `data/extracted/` — Page-level raw text & OCR provenance JSONL files (`pages.jsonl`, `extraction_summary.json`).
- `data/cleaned/` — Cleaned Gazette documents (`documents.jsonl`), parsed clauses (`clauses.jsonl`), and SQLite database (`gazettes.db`).
- `data/processed/` — Clause-aware retrieval chunks (`chunks.jsonl`).
- `data/index/` — Persisted BM25 (`bm25.pkl`) and Multilingual-E5 FAISS (`faiss.index`, `semantic_meta.json`) indices.
- `data/eval/` — 200-query Hinglish evaluation benchmark (`queries_200.jsonl`) and ablation reports (`eval_results.json`, `comparison_table.csv`, `per_category_metrics.csv`).
- `src/` — Pipeline modules (`config.py`, `krutidev.py`, `phase1_dataset.py`, `phase1_extractor.py`, `phase1_cleaner.py`, `phase2_chunker.py`, `phase3_normalizer.py`, `phase4_bm25.py`, `phase5_semantic.py`, `phase6_hybrid.py`, `phase8_dataset.py`, `phase8_eval.py`, `api.py`, `storage.py`).
- `scripts/` — CLI entrypoints (`run_phase1.py`, `add_gazette_pdfs.py`, `build_indices.py`, `run_phase8_eval.py`).
- `app.py` — Streamlit search, comparison & benchmark UI.
- `tests/` — Automated tests (`test_phase1.py`, `test_phases_2_to_7.py`, `test_phase8.py`).

## Quickstart (Phases 1–8)

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 1. Phase 1: Extract, decode Kruti Dev, clean, and structure Gazettes
python scripts/run_phase1.py

# 2. Phases 2-5: Clause-aware chunking + BM25 + Multilingual-E5 FAISS indices
python scripts/build_indices.py

# 3. Phase 7: Start FastAPI backend (Terminal 1) and Streamlit frontend (Terminal 2)
uvicorn src.api:app --host 127.0.0.1 --port 8000
streamlit run app.py

# 4. Phase 8: Build 200 Hinglish queries & run IR benchmark (P@10, R@10, MRR@10, nDCG@10)
python scripts/run_phase8_eval.py

# 5. Run all automated tests
pytest
```
