"""Phase 8: IR evaluation metrics (P@K, R@K, MRR@K, nDCG@K) and ablation benchmarking."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    EVAL_CATEGORY_CSV_PATH,
    EVAL_COMPARISON_CSV_PATH,
    EVAL_QUERIES_PATH,
    EVAL_RESULTS_PATH,
)
from src.storage import read_jsonl, write_json


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in relevant_ids)
    return hits / float(k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of all relevant items captured in top-k."""
    if not relevant_ids or k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for cid in top if cid in relevant_ids)
    return hits / float(len(relevant_ids))


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant item within top-k."""
    for idx, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Binary-relevance Normalized Discounted Cumulative Gain at k."""
    if not relevant_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for idx, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def score_query_ranking(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute P@K, R@K, MRR@K, and nDCG@K for a single query."""
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k=k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k=k),
        f"mrr@{k}": mrr_at_k(retrieved_ids, relevant_ids, k=k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k=k),
    }


def _mean_metrics(rows: list[dict[str, float]], k: int = 10) -> dict[str, float]:
    if not rows:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}
    keys = [f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
    return {metric: round(sum(r[metric] for r in rows) / len(rows), 4) for metric in keys}


def evaluate_models(
    hybrid_retriever: Any,
    queries: list[dict[str, Any]] | None = None,
    queries_path: Path = EVAL_QUERIES_PATH,
    k: int = 10,
    granularity: str = "chunk",
) -> dict[str, Any]:
    """Benchmark Raw BM25, Normalized BM25, Semantic E5, and Hybrid RRF across all queries."""
    eval_queries = queries if queries is not None else read_jsonl(queries_path)
    id_field = "chunk_id" if granularity == "chunk" else "doc_id"
    qrel_field = "relevant_chunk_ids" if granularity == "chunk" else "relevant_doc_ids"

    systems = [
        ("bm25_raw", "BM25 (Raw Baseline)", "bm25", False),
        ("bm25_normalized", "BM25 + Phonetic/Dialect", "bm25", True),
        ("semantic_e5", "Multilingual-E5 Semantic", "semantic", True),
        ("hybrid_rrf", "Proposed Hybrid RRF", "hybrid", True),
    ]

    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    per_system_details: dict[str, Any] = {}

    for sys_key, sys_label, mode, use_norm in systems:
        all_scores: list[dict[str, float]] = []
        by_type: dict[str, list[dict[str, float]]] = {}
        latencies_ms: list[float] = []

        for qrec in eval_queries:
            resp = hybrid_retriever.search(
                query=qrec["query"],
                top_k=k,
                mode=mode,
                use_normalization=use_norm,
            )
            latencies_ms.append(float(resp.get("elapsed_ms", 0.0)))

            # Deduplicate doc_ids while preserving rank order if evaluating at doc level
            raw_ids = [r[id_field] for r in resp["results"]]
            if granularity == "doc":
                seen: set[str] = set()
                retrieved_ids = []
                for rid in raw_ids:
                    if rid not in seen:
                        seen.add(rid)
                        retrieved_ids.append(rid)
            else:
                retrieved_ids = raw_ids

            rel_set = set(qrec[qrel_field])
            metrics = score_query_ranking(retrieved_ids, rel_set, k=k)
            all_scores.append(metrics)
            by_type.setdefault(qrec["query_type"], []).append(metrics)

        overall = _mean_metrics(all_scores, k=k)
        avg_latency = round(sum(latencies_ms) / max(1, len(latencies_ms)), 2)

        summary_rows.append(
            {
                "system_key": sys_key,
                "system": sys_label,
                "num_queries": len(eval_queries),
                **overall,
                "avg_latency_ms": avg_latency,
            }
        )

        cat_breakdown: dict[str, dict[str, float]] = {}
        for qtype, qrows in sorted(by_type.items()):
            cat_metrics = _mean_metrics(qrows, k=k)
            cat_breakdown[qtype] = cat_metrics
            category_rows.append(
                {
                    "system_key": sys_key,
                    "system": sys_label,
                    "query_type": qtype,
                    "num_queries": len(qrows),
                    **cat_metrics,
                }
            )

        per_system_details[sys_key] = {
            "label": sys_label,
            "overall": overall,
            "avg_latency_ms": avg_latency,
            "by_query_type": cat_breakdown,
        }

    # Compute relative gains of Proposed Hybrid RRF over Raw BM25 and Semantic-only
    by_key = {row["system_key"]: row for row in summary_rows}
    improvements: dict[str, dict[str, float]] = {}
    if "hybrid_rrf" in by_key:
        hyb = by_key["hybrid_rrf"]
        for base_key in ("bm25_raw", "bm25_normalized", "semantic_e5"):
            if base_key in by_key:
                base = by_key[base_key]
                improvements[f"vs_{base_key}"] = {
                    metric: round(
                        ((hyb[metric] - base[metric]) / base[metric] * 100.0)
                        if base[metric] > 0
                        else 0.0,
                        2,
                    )
                    for metric in (f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}")
                }

    return {
        "k": k,
        "granularity": granularity,
        "num_queries": len(eval_queries),
        "summary": summary_rows,
        "per_category": category_rows,
        "improvements_pct": improvements,
        "details": per_system_details,
    }


def save_evaluation_reports(
    report: dict[str, Any],
    json_path: Path = EVAL_RESULTS_PATH,
    comparison_csv_path: Path = EVAL_COMPARISON_CSV_PATH,
    category_csv_path: Path = EVAL_CATEGORY_CSV_PATH,
) -> None:
    """Persist Phase 8 benchmark JSON and CSV tables."""
    write_json(json_path, report)
    comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report["summary"]).to_csv(comparison_csv_path, index=False, encoding="utf-8")
    pd.DataFrame(report["per_category"]).to_csv(category_csv_path, index=False, encoding="utf-8")
