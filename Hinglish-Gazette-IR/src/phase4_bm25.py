"""Phase 4: BM25 keyword retrieval over clause-aware Gazette chunks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH, CHUNKS_JSONL_PATH
from src.phase3_normalizer import HinglishNormalizer, tokenize_document_for_bm25
from src.storage import read_jsonl


class BM25Retriever:
    """BM25Okapi keyword retriever with optional Phase 3 Hinglish normalization."""

    def __init__(self, chunks: list[dict[str, Any]], bm25: BM25Okapi) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.normalizer = HinglishNormalizer()

    @classmethod
    def build_from_chunks(cls, chunks: list[dict[str, Any]]) -> "BM25Retriever":
        tokenized_corpus = [
            tokenize_document_for_bm25(
                f"{c.get('title', '')} {c.get('ministry', '')} {c.get('retrieval_text', c['text'])}"
            )
            for c in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(chunks=chunks, bm25=bm25)

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, handle)

    @classmethod
    def load(
        cls,
        index_path: Path = BM25_INDEX_PATH,
        chunks_path: Path = CHUNKS_JSONL_PATH,
    ) -> "BM25Retriever":
        if index_path.exists():
            with index_path.open("rb") as handle:
                payload = pickle.load(handle)
            return cls(chunks=payload["chunks"], bm25=payload["bm25"])
        chunks = read_jsonl(chunks_path)
        retriever = cls.build_from_chunks(chunks)
        retriever.save(index_path)
        return retriever

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_normalization: bool = True,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.chunks or top_k <= 0:
            return []

        if use_normalization:
            query_tokens = self.normalizer.normalize(query).bm25_tokens
        else:
            query_tokens = tokenize_document_for_bm25(query)

        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0 and results:
                break
            chunk = self.chunks[int(idx)]
            if category and chunk.get("category") != category:
                continue
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "clause_id": chunk["clause_id"],
                    "heading": chunk.get("heading", ""),
                    "title": chunk.get("title", ""),
                    "category": chunk.get("category", ""),
                    "gazette_date": chunk.get("gazette_date", ""),
                    "ministry": chunk.get("ministry", ""),
                    "notification_no": chunk.get("notification_no", ""),
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "bm25_score": round(score, 6),
                    "bm25_rank": len(results) + 1,
                }
            )
            if len(results) >= top_k:
                break

        return results
