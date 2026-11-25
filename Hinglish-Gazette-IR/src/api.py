"""Phase 7: FastAPI backend serving Hinglish Gazette search and ablation comparison."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import DOCUMENTS_JSONL_PATH, EMBEDDING_MODEL_NAME
from src.phase3_normalizer import HinglishNormalizer
from src.phase4_bm25 import BM25Retriever
from src.phase5_semantic import SemanticRetriever
from src.phase6_hybrid import HybridRetriever
from src.storage import read_jsonl


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query in Hinglish, Hindi, English, or dialect.")
    top_k: int = Field(default=10, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "semantic"] = "hybrid"
    use_normalization: bool = True
    category: str | None = None


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    bm25 = BM25Retriever.load()
    semantic = SemanticRetriever.load()
    return HybridRetriever(bm25_retriever=bm25, semantic_retriever=semantic)


@lru_cache(maxsize=1)
def get_documents_catalog() -> list[dict[str, Any]]:
    if not DOCUMENTS_JSONL_PATH.exists():
        return []
    docs = read_jsonl(DOCUMENTS_JSONL_PATH)
    return [
        {
            "doc_id": d["doc_id"],
            "filename": d["filename"],
            "title": d.get("title", ""),
            "category": d.get("category", ""),
            "gazette_date": d.get("gazette_date", ""),
            "ministry": d.get("ministry", ""),
            "notification_no": d.get("notification_no", ""),
            "language_mix": d.get("language_mix", ""),
            "page_count": d.get("page_count", 0),
            "char_count": d.get("char_count", 0),
        }
        for d in docs
    ]


def create_app(retriever: HybridRetriever | None = None) -> FastAPI:
    """Create the FastAPI application (supports injecting a mock retriever in tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if retriever is None:
            app.state.retriever = get_retriever()
        else:
            app.state.retriever = retriever
        yield

    application = FastAPI(
        title="Hinglish-Gazette-IR API",
        description="Phonetic & Dialect-Aware Hybrid Information Retrieval over Government Gazettes",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.retriever = retriever

    def _active_retriever() -> HybridRetriever:
        if application.state.retriever is None:
            application.state.retriever = get_retriever()
        return application.state.retriever

    @application.get("/health")
    def health() -> dict[str, Any]:
        ret = _active_retriever()
        docs = get_documents_catalog()
        return {
            "status": "ok",
            "documents_count": len(docs),
            "chunks_count": len(ret.bm25.chunks),
            "embedding_model": getattr(ret.semantic, "model_name", EMBEDDING_MODEL_NAME),
        }

    @application.get("/documents")
    def list_documents() -> dict[str, Any]:
        docs = get_documents_catalog()
        return {"count": len(docs), "documents": docs}

    @application.post("/normalize")
    def normalize_query(payload: SearchRequest) -> dict[str, Any]:
        normalizer = HinglishNormalizer()
        return normalizer.normalize(payload.query).to_dict()

    @application.post("/search")
    def search(payload: SearchRequest) -> dict[str, Any]:
        ret = _active_retriever()
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        return ret.search(
            query=payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
            use_normalization=payload.use_normalization,
            category=payload.category,
        )

    @application.post("/compare")
    def compare(payload: CompareRequest) -> dict[str, Any]:
        ret = _active_retriever()
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        return ret.compare(
            query=payload.query,
            top_k=payload.top_k,
            category=payload.category,
        )

    return application


app = create_app()
