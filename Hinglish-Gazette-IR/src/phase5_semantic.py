"""Phase 5: Multilingual-E5 semantic retrieval backed by FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import (
    CHUNKS_JSONL_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    SEMANTIC_META_PATH,
)
from src.phase3_normalizer import HinglishNormalizer
from src.storage import read_jsonl


class SemanticRetriever:
    """Dense vector retriever using Multilingual-E5 embeddings and FAISS IndexFlatIP."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        index: faiss.Index,
        model_name: str = EMBEDDING_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.model_name = model_name
        self._model = model
        self.normalizer = HinglishNormalizer()

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @classmethod
    def build_from_chunks(
        cls,
        chunks: list[dict[str, Any]],
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        model: Any | None = None,
    ) -> "SemanticRetriever":
        if model is None:
            from sentence_transformers import SentenceTransformer

            embed_model = SentenceTransformer(model_name)
        else:
            embed_model = model
        passages = [
            f"passage: {c.get('category', '')} | {c.get('ministry', '')} | {c.get('retrieval_text', c['text'])}"
            for c in chunks
        ]
        embeddings = np.asarray(
            embed_model.encode(
                passages,
                batch_size=batch_size,
                show_progress_bar=len(passages) > 64,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(chunks=chunks, index=index, model_name=model_name, model=embed_model)

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = SEMANTIC_META_PATH,
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": self.model_name,
                    "num_chunks": len(self.chunks),
                    "dimension": self.index.d,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = SEMANTIC_META_PATH,
        chunks_path: Path = CHUNKS_JSONL_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> "SemanticRetriever":
        chunks = read_jsonl(chunks_path)
        if index_path.exists() and meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            resolved_model = meta.get("model_name", model_name)
            index = faiss.read_index(str(index_path))
            return cls(chunks=chunks, index=index, model_name=resolved_model)

        retriever = cls.build_from_chunks(chunks, model_name=model_name)
        retriever.save(index_path=index_path, meta_path=meta_path)
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

        effective_query = (
            self.normalizer.normalize(query).semantic_query if use_normalization else query.strip()
        )
        if not effective_query:
            return []

        query_vec = self._encode([f"query: {effective_query}"], batch_size=1)
        # Over-fetch if filtering by category
        fetch_k = min(len(self.chunks), top_k * 5 if category else top_k)
        scores, indices = self.index.search(query_vec, fetch_k)

        results: list[dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0]):
            if int(idx) < 0:
                continue
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
                    "semantic_score": round(float(score), 6),
                    "semantic_rank": len(results) + 1,
                }
            )
            if len(results) >= top_k:
                break

        return results
