"""Tests for Phases 2-7: Chunking, Phonetic/Dialect Normalization, BM25, FAISS, RRF, and FastAPI."""

from __future__ import annotations

import numpy as np

from src.api import CompareRequest, SearchRequest, create_app
from src.phase2_chunker import build_clause_chunks
from src.phase3_normalizer import HinglishNormalizer, hinglish_phonetic_key
from src.phase4_bm25 import BM25Retriever
from src.phase5_semantic import SemanticRetriever
from src.phase6_hybrid import HybridRetriever, reciprocal_rank_fusion


class _DummyEmbeddingModel:
    """Deterministic lightweight embedder for offline unit tests."""

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True):
        vecs = []
        for t in texts:
            low = t.lower()
            v = np.array(
                [
                    float("maternity" in low or "women" in low or "महिला" in low or "प्रसूति" in low),
                    float("land" in low or "acquisition" in low or "भूमि" in low or "अधिग्रहण" in low),
                    float("promotion" in low or "salary" in low or "पदोन्नति" in low or "वेतन" in low),
                    float("court" in low or "judge" in low or "न्यायालय" in low),
                    0.1,
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.vstack(vecs)


def _sample_chunks() -> list[dict]:
    docs = [
        {
            "doc_id": "d1",
            "title": "Gazette Issue 1",
            "category": "Rules and Regulations",
            "gazette_date": "2026-09-19",
            "ministry": "MAHILA KALYAN VIBHAG",
            "notification_no": "No. 101/2026",
            "language_mix": "bilingual",
        }
    ]
    clauses = [
        {
            "clause_id": "d1__c0001",
            "doc_id": "d1",
            "clause_index": 1,
            "clause_label": "अनुभाग-1",
            "heading": "Maternity Benefit Rules",
            "page_start": 1,
            "page_end": 1,
            "text": "Employees shall be entitled to maternity benefits and paid leave for women staff.",
        },
        {
            "clause_id": "d1__c0002",
            "doc_id": "d1",
            "clause_index": 2,
            "clause_label": "अनुभाग-2",
            "heading": "Land Acquisition",
            "page_start": 2,
            "page_end": 2,
            "text": "भूमि अधिग्रहण एवं राजस्व मुआवजा नियम लागू होंगे।",
        },
        {
            "clause_id": "d1__c0003",
            "doc_id": "d1",
            "clause_index": 3,
            "clause_label": "अनुभाग-3",
            "heading": "पदोन्नति आदेश",
            "page_start": 3,
            "page_end": 3,
            "text": "पे-मैट्रिक्स लेवल-11 के पद पर पदोन्नति एवं वेतन निर्धारण।",
        },
    ]
    return build_clause_chunks(docs, clauses, min_chars=20, max_chars=500)


def test_phase2_chunking_merges_stubs_and_splits_long_clause() -> None:
    docs = [{"doc_id": "d1", "title": "T", "category": "Ministry Order"}]
    long_sentence = "This is a formal government gazette clause on service conditions. " * 25
    clauses = [
        {
            "clause_id": "d1__c0001",
            "doc_id": "d1",
            "clause_index": 1,
            "clause_label": "अनुभाग-1",
            "heading": "अनुभाग-1",
            "page_start": 1,
            "page_end": 1,
            "text": "अनुभाग-1",
        },
        {
            "clause_id": "d1__c0002",
            "doc_id": "d1",
            "clause_index": 2,
            "clause_label": "No. 1",
            "heading": "Service Rules",
            "page_start": 1,
            "page_end": 2,
            "text": long_sentence.strip(),
        },
    ]
    chunks = build_clause_chunks(docs, clauses, min_chars=80, max_chars=400, overlap_chars=60)
    assert len(chunks) >= 3
    assert all(c["char_count"] <= 420 for c in chunks)
    assert "अनुभाग-1" in chunks[0]["text"]


def test_phase3_phonetic_and_dialect_normalization() -> None:
    assert hinglish_phonetic_key("benifit") == hinglish_phonetic_key("benefit")
    assert hinglish_phonetic_key("chhutti") == hinglish_phonetic_key("chutti")

    norm = HinglishNormalizer().normalize("Mahila ko maternity benifit aur chutti ke niyam")
    assert norm.phonetic_corrections.get("benifit") == "benefit" or "benefit" in norm.bm25_tokens
    assert "women" in norm.bm25_tokens
    assert "leave" in norm.bm25_tokens
    assert "महिला" in norm.bm25_tokens
    assert "नियम" in norm.bm25_tokens
    assert len(norm.explanation) >= 2


def test_phases_4_to_6_bm25_semantic_and_rrf() -> None:
    chunks = _sample_chunks()
    bm25 = BM25Retriever.build_from_chunks(chunks)
    semantic = SemanticRetriever.build_from_chunks(chunks, model=_DummyEmbeddingModel())
    hybrid = HybridRetriever(bm25_retriever=bm25, semantic_retriever=semantic)

    res = hybrid.search("Mahila ko maternity chhutti ke rules", top_k=3, mode="hybrid")
    assert res["results"][0]["heading"] == "Maternity Benefit Rules"
    assert res["results"][0]["rrf_score"] > 0

    jameen_res = hybrid.search("Jameen adhigrahan muavja", top_k=3, mode="hybrid")
    assert jameen_res["results"][0]["heading"] == "Land Acquisition"


def test_reciprocal_rank_fusion_combines_ranks() -> None:
    b = [
        {"chunk_id": "c1", "bm25_rank": 1, "bm25_score": 8.0, "text": "a"},
        {"chunk_id": "c2", "bm25_rank": 2, "bm25_score": 5.0, "text": "b"},
    ]
    s = [
        {"chunk_id": "c2", "semantic_rank": 1, "semantic_score": 0.9, "text": "b"},
        {"chunk_id": "c3", "semantic_rank": 2, "semantic_score": 0.8, "text": "c"},
    ]
    fused = reciprocal_rank_fusion(b, s, top_k=3, rrf_k=60)
    assert fused[0]["chunk_id"] == "c2"  # present in both lists at ranks 2 and 1


def test_phase7_fastapi_endpoints() -> None:
    chunks = _sample_chunks()
    bm25 = BM25Retriever.build_from_chunks(chunks)
    semantic = SemanticRetriever.build_from_chunks(chunks, model=_DummyEmbeddingModel())
    hybrid = HybridRetriever(bm25_retriever=bm25, semantic_retriever=semantic)

    application = create_app(retriever=hybrid)
    routes = {route.path: route.endpoint for route in application.routes if hasattr(route, "endpoint")}
    assert {"/health", "/documents", "/normalize", "/search", "/compare"}.issubset(routes.keys())

    health_body = routes["/health"]()
    assert health_body["status"] == "ok"
    assert health_body["chunks_count"] == len(chunks)

    search_body = routes["/search"](
        SearchRequest(query="Mahila maternity leave", top_k=2, mode="hybrid")
    )
    assert len(search_body["results"]) == 2
    assert search_body["results"][0]["heading"] == "Maternity Benefit Rules"

    cmp_body = routes["/compare"](CompareRequest(query="Jameen adhigrahan", top_k=2))
    assert "hybrid_rrf" in cmp_body
    assert cmp_body["hybrid_rrf"][0]["heading"] == "Land Acquisition"
