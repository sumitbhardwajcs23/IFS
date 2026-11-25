"""Build Phase 2 chunks and Phases 4-5 retrieval indices (BM25 + Multilingual-E5 FAISS)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BM25_INDEX_PATH,
    CHUNKS_JSONL_PATH,
    CLAUSES_JSONL_PATH,
    DOCUMENTS_JSONL_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    SEMANTIC_META_PATH,
)
from src.phase2_chunker import run_chunking
from src.phase4_bm25 import BM25Retriever
from src.phase5_semantic import SemanticRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 chunks, BM25 index, and FAISS index.")
    parser.add_argument("--model", default=EMBEDDING_MODEL_NAME, help="SentenceTransformer model name.")
    parser.add_argument("--skip-semantic", action="store_true", help="Only build chunks and BM25 index.")
    args = parser.parse_args()

    print("[Phase 2] Running clause-aware chunking...")
    chunks = run_chunking(
        documents_path=DOCUMENTS_JSONL_PATH,
        clauses_path=CLAUSES_JSONL_PATH,
        output_path=CHUNKS_JSONL_PATH,
    )
    print(f"[Phase 2] Wrote {len(chunks)} chunks to {CHUNKS_JSONL_PATH}")

    print("[Phase 4] Building BM25 index...")
    bm25 = BM25Retriever.build_from_chunks(chunks)
    bm25.save(BM25_INDEX_PATH)
    print(f"[Phase 4] Saved BM25 index to {BM25_INDEX_PATH}")

    if not args.skip_semantic:
        print(f"[Phase 5] Encoding {len(chunks)} chunks with {args.model} and building FAISS index...")
        semantic = SemanticRetriever.build_from_chunks(chunks, model_name=args.model)
        semantic.save(index_path=FAISS_INDEX_PATH, meta_path=SEMANTIC_META_PATH)
        print(f"[Phase 5] Saved FAISS index to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
