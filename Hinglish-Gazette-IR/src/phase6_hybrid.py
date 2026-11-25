"""Phase 6: Hybrid retrieval combining BM25 and Semantic search via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import time
from typing import Any

from src.config import DEFAULT_BM25_WEIGHT, DEFAULT_SEMANTIC_WEIGHT, RRF_K
from src.phase3_normalizer import HinglishNormalizer, NormalizedQuery


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = 10,
    rrf_k: int = RRF_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> list[dict[str, Any]]:
    """Fuse BM25 and Semantic ranked lists using weighted Reciprocal Rank Fusion."""
    fused: dict[str, dict[str, Any]] = {}

    for item in bm25_results:
        cid = item["chunk_id"]
        entry = fused.setdefault(cid, dict(item))
        rank = item["bm25_rank"]
        entry["bm25_rank"] = rank
        entry["bm25_score"] = item["bm25_score"]
        entry["rrf_score"] = entry.get("rrf_score", 0.0) + bm25_weight / (rrf_k + rank)

    for item in semantic_results:
        cid = item["chunk_id"]
        entry = fused.setdefault(cid, dict(item))
        rank = item["semantic_rank"]
        entry["semantic_rank"] = rank
        entry["semantic_score"] = item["semantic_score"]
        entry["rrf_score"] = entry.get("rrf_score", 0.0) + semantic_weight / (rrf_k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda x: (x["rrf_score"], x.get("semantic_score", 0.0), x.get("bm25_score", 0.0)),
        reverse=True,
    )

    output: list[dict[str, Any]] = []
    for idx, item in enumerate(ranked[:top_k], start=1):
        item["rank"] = idx
        item["rrf_score"] = round(float(item["rrf_score"]), 6)
        item.setdefault("bm25_rank", None)
        item.setdefault("bm25_score", None)
        item.setdefault("semantic_rank", None)
        item.setdefault("semantic_score", None)
        output.append(item)
    return output


class HybridRetriever:
    """Unified search facade supporting BM25, Semantic, Hybrid RRF, and comparison mode."""

    def __init__(
        self,
        bm25_retriever: Any,
        semantic_retriever: Any,
        rrf_k: int = RRF_K,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    ) -> None:
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.normalizer = HinglishNormalizer()

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        use_normalization: bool = True,
        category: str | None = None,
        candidate_k: int | None = None,
    ) -> dict[str, Any]:
        """Run retrieval in 'bm25', 'semantic', or 'hybrid' mode and return normalized trace + hits."""
        t0 = time.perf_counter()
        normalized: NormalizedQuery = self.normalizer.normalize(query)
        fetch_k = candidate_k or max(top_k * 3, 30)

        if mode == "bm25":
            hits = self.bm25.search(
                query, top_k=top_k, use_normalization=use_normalization, category=category
            )
            for idx, h in enumerate(hits, start=1):
                h["rank"] = idx
        elif mode == "semantic":
            hits = self.semantic.search(
                query, top_k=top_k, use_normalization=use_normalization, category=category
            )
            for idx, h in enumerate(hits, start=1):
                h["rank"] = idx
        elif mode == "hybrid":
            bm25_hits = self.bm25.search(
                query, top_k=fetch_k, use_normalization=use_normalization, category=category
            )
            sem_hits = self.semantic.search(
                query, top_k=fetch_k, use_normalization=use_normalization, category=category
            )
            hits = reciprocal_rank_fusion(
                bm25_results=bm25_hits,
                semantic_results=sem_hits,
                top_k=top_k,
                rrf_k=self.rrf_k,
                bm25_weight=self.bm25_weight,
                semantic_weight=self.semantic_weight,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'hybrid', 'bm25', or 'semantic'.")

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "query": query,
            "mode": mode,
            "use_normalization": use_normalization,
            "category_filter": category,
            "elapsed_ms": elapsed_ms,
            "normalization": normalized.to_dict(),
            "results": hits,
        }

    def compare(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Run baseline BM25, Semantic, and Normalized Hybrid RRF side by side."""
        return {
            "query": query,
            "normalization": self.normalizer.normalize(query).to_dict(),
            "bm25_raw": self.search(
                query, top_k=top_k, mode="bm25", use_normalization=False, category=category
            )["results"],
            "bm25_normalized": self.search(
                query, top_k=top_k, mode="bm25", use_normalization=True, category=category
            )["results"],
            "semantic": self.search(
                query, top_k=top_k, mode="semantic", use_normalization=True, category=category
            )["results"],
            "hybrid_rrf": self.search(
                query, top_k=top_k, mode="hybrid", use_normalization=True, category=category
            )["results"],
        }
