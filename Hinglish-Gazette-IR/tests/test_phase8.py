"""Tests for Phase 8: IR evaluation metrics, 200-query benchmark dataset, and model comparison."""

from __future__ import annotations

import math
from pathlib import Path

from src.config import CHUNKS_JSONL_PATH
from src.phase4_bm25 import BM25Retriever
from src.phase5_semantic import SemanticRetriever
from src.phase6_hybrid import HybridRetriever
from src.phase8_dataset import TOPIC_SPECS, build_eval_dataset
from src.phase8_eval import (
    evaluate_models,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from tests.test_phases_2_to_7 import _DummyEmbeddingModel, _sample_chunks


def test_ir_metrics_exact_calculations() -> None:
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}

    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5
    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    assert mrr_at_k(retrieved, relevant, k=5) == 0.5

    expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(retrieved, relevant, k=5), expected_dcg / expected_idcg, rel_tol=1e-6)


def test_topic_specs_count_is_50_for_200_queries() -> None:
    assert len(TOPIC_SPECS) == 50
    total_queries = sum(len(t.queries) for t in TOPIC_SPECS)
    assert total_queries == 200


def test_build_eval_dataset_if_chunks_exist(tmp_path: Path) -> None:
    if not CHUNKS_JSONL_PATH.exists():
        return
    records = build_eval_dataset(chunks_path=CHUNKS_JSONL_PATH, output_path=tmp_path / "q200.jsonl")
    assert len(records) == 200
    assert all(len(r["relevant_chunk_ids"]) >= 1 for r in records)
    assert {r["query_type"] for r in records} == {
        "normal_hinglish",
        "phonetic_typo",
        "dialect_regional",
        "informal_question",
    }


def test_evaluate_models_pipeline() -> None:
    chunks = _sample_chunks()
    bm25 = BM25Retriever.build_from_chunks(chunks)
    semantic = SemanticRetriever.build_from_chunks(chunks, model=_DummyEmbeddingModel())
    hybrid = HybridRetriever(bm25_retriever=bm25, semantic_retriever=semantic)

    queries = [
        {
            "query_id": "Q1",
            "topic_id": "T1",
            "query_type": "normal_hinglish",
            "query": "Mahila maternity leave",
            "relevant_chunk_ids": ["d1__chk0001"],
            "relevant_doc_ids": ["d1"],
        },
        {
            "query_id": "Q2",
            "topic_id": "T2",
            "query_type": "dialect_regional",
            "query": "Jameen adhigrahan",
            "relevant_chunk_ids": ["d1__chk0002"],
            "relevant_doc_ids": ["d1"],
        },
    ]
    report = evaluate_models(hybrid, queries=queries, k=3, granularity="chunk")
    assert len(report["summary"]) == 4
    by_key = {row["system_key"]: row for row in report["summary"]}
    assert by_key["hybrid_rrf"]["mrr@3"] == 1.0
    assert by_key["hybrid_rrf"]["ndcg@3"] == 1.0
