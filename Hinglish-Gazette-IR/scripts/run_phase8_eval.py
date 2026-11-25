"""Phase 8 CLI runner: Generate 200 Hinglish evaluation queries and benchmark all models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHUNKS_JSONL_PATH,
    EVAL_CATEGORY_CSV_PATH,
    EVAL_COMPARISON_CSV_PATH,
    EVAL_QUERIES_PATH,
    EVAL_RESULTS_PATH,
)
from src.phase4_bm25 import BM25Retriever
from src.phase5_semantic import SemanticRetriever
from src.phase6_hybrid import HybridRetriever
from src.phase8_dataset import build_eval_dataset
from src.phase8_eval import evaluate_models, save_evaluation_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 8 evaluation across 200 Hinglish queries.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for P@K, R@K, MRR@K, nDCG@K.")
    parser.add_argument(
        "--granularity",
        choices=["chunk", "doc"],
        default="chunk",
        help="Evaluate retrieved chunk_id or document-level doc_id.",
    )
    args = parser.parse_args()

    print("[Phase 8] Building 200-query Hinglish evaluation benchmark...")
    queries = build_eval_dataset(chunks_path=CHUNKS_JSONL_PATH, output_path=EVAL_QUERIES_PATH)
    print(f"[Phase 8] Saved {len(queries)} queries to {EVAL_QUERIES_PATH}")

    print("[Phase 8] Loading BM25 + Multilingual-E5 FAISS retrievers...")
    bm25 = BM25Retriever.load()
    semantic = SemanticRetriever.load()
    # Warm up embedding model before timing queries
    _ = semantic.model
    hybrid = HybridRetriever(bm25_retriever=bm25, semantic_retriever=semantic)

    print(f"[Phase 8] Evaluating 4 systems @K={args.k} ({args.granularity}-level)...")
    report = evaluate_models(
        hybrid_retriever=hybrid,
        queries=queries,
        k=args.k,
        granularity=args.granularity,
    )
    save_evaluation_reports(
        report,
        json_path=EVAL_RESULTS_PATH,
        comparison_csv_path=EVAL_COMPARISON_CSV_PATH,
        category_csv_path=EVAL_CATEGORY_CSV_PATH,
    )

    print("\n=== OVERALL BENCHMARK COMPARISON (200 Hinglish Queries) ===")
    summary_df = pd.DataFrame(report["summary"]).drop(columns=["system_key"])
    print(summary_df.to_string(index=False))

    print("\n=== RELATIVE IMPROVEMENT OF HYBRID RRF (%) ===")
    for vs_key, gains in report["improvements_pct"].items():
        print(f"  {vs_key}: {gains}")

    print("\n=== PER-QUERY-TYPE BREAKDOWN ===")
    cat_df = pd.DataFrame(report["per_category"]).drop(columns=["system_key"])
    print(cat_df.to_string(index=False))

    print(f"\n[Phase 8] Saved reports to {EVAL_RESULTS_PATH} and {EVAL_COMPARISON_CSV_PATH}")


if __name__ == "__main__":
    main()
