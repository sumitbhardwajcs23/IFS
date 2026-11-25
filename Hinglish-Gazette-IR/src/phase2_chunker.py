"""Phase 2: Clause-aware chunking that respects Gazette clause boundaries and merges stub headers."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNKS_JSONL_PATH,
    CLAUSES_JSONL_PATH,
    DOCUMENTS_JSONL_PATH,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
)
from src.storage import read_jsonl, write_jsonl

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।\.!\?])\s+|\n+")


def _split_long_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split a long clause along sentence/line boundaries with character overlap."""
    if len(text) <= max_chars:
        return [text]

    segments = [seg.strip() for seg in _SENTENCE_SPLIT_RE.split(text) if seg.strip()]
    if not segments:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars - overlap_chars)]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for seg in segments:
        # Handle single oversized sentence/table row gracefully
        if len(seg) > max_chars:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_len = 0
            step = max(100, max_chars - overlap_chars)
            for start in range(0, len(seg), step):
                chunks.append(seg[start : start + max_chars].strip())
            continue

        projected = current_len + (1 if current_parts else 0) + len(seg)
        if projected > max_chars and current_parts:
            chunk_text = " ".join(current_parts).strip()
            chunks.append(chunk_text)
            # Carry trailing context into next chunk for boundary continuity
            overlap_parts: list[str] = []
            overlap_len = 0
            for prev_seg in reversed(current_parts):
                if overlap_len + len(prev_seg) + 1 > overlap_chars:
                    break
                overlap_parts.insert(0, prev_seg)
                overlap_len += len(prev_seg) + 1
            current_parts = overlap_parts + [seg]
            current_len = sum(len(p) for p in current_parts) + max(0, len(current_parts) - 1)
        else:
            current_parts.append(seg)
            current_len = projected

    if current_parts:
        tail = " ".join(current_parts).strip()
        if tail:
            chunks.append(tail)

    return chunks


def _merge_stub_clauses(
    doc_clauses: list[dict[str, Any]],
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict[str, Any]]:
    """Merge tiny heading-only clauses forward into the following clause within the same document."""
    if not doc_clauses:
        return []

    merged: list[dict[str, Any]] = []
    pending_stub: dict[str, Any] | None = None

    for clause in doc_clauses:
        item = dict(clause)
        if pending_stub is not None:
            combined_text = f"{pending_stub['text']}\n{item['text']}".strip()
            if len(combined_text) <= max_chars or len(pending_stub["text"]) < min_chars:
                item["text"] = combined_text
                item["char_count"] = len(combined_text)
                item["page_start"] = min(pending_stub["page_start"], item["page_start"])
                item["page_end"] = max(pending_stub["page_end"], item["page_end"])
                if not item.get("heading") or len(item["heading"]) < 8:
                    item["heading"] = pending_stub.get("heading", "")
                if pending_stub.get("clause_label") and not item.get("clause_label"):
                    item["clause_label"] = pending_stub["clause_label"]
            else:
                merged.append(pending_stub)
            pending_stub = None

        if len(item["text"]) < min_chars:
            pending_stub = item
        else:
            merged.append(item)

    if pending_stub is not None:
        if merged and len(merged[-1]["text"]) + len(pending_stub["text"]) + 1 <= max_chars:
            merged[-1]["text"] = f"{merged[-1]['text']}\n{pending_stub['text']}".strip()
            merged[-1]["char_count"] = len(merged[-1]["text"])
            merged[-1]["page_end"] = max(merged[-1]["page_end"], pending_stub["page_end"])
        else:
            merged.append(pending_stub)

    return merged


def build_clause_chunks(
    documents: list[dict[str, Any]],
    clauses: list[dict[str, Any]],
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    """Produce retrieval-ready, clause-aware chunks enriched with document metadata."""
    doc_lookup = {doc["doc_id"]: doc for doc in documents}
    clauses_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clause in clauses:
        clauses_by_doc[clause["doc_id"]].append(clause)

    all_chunks: list[dict[str, Any]] = []

    for doc_id in sorted(clauses_by_doc.keys()):
        doc_meta = doc_lookup.get(doc_id, {})
        ordered_clauses = sorted(clauses_by_doc[doc_id], key=lambda c: c["clause_index"])
        coalesced = _merge_stub_clauses(ordered_clauses, min_chars=min_chars, max_chars=max_chars)

        doc_chunk_idx = 0
        for clause in coalesced:
            sub_texts = _split_long_text(
                clause["text"],
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            total_parts = len(sub_texts)
            for part_idx, chunk_text in enumerate(sub_texts, start=1):
                doc_chunk_idx += 1
                chunk_id = f"{doc_id}__chk{doc_chunk_idx:04d}"
                heading = (clause.get("heading") or "").strip()
                header_prefix = f"{heading}\n" if heading and heading not in chunk_text[:180] else ""
                retrieval_text = f"{header_prefix}{chunk_text}".strip()

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "clause_id": clause["clause_id"],
                        "chunk_index": doc_chunk_idx,
                        "part_index": part_idx,
                        "total_parts": total_parts,
                        "clause_label": clause.get("clause_label", ""),
                        "heading": heading,
                        "title": doc_meta.get("title", ""),
                        "category": doc_meta.get("category", ""),
                        "gazette_date": doc_meta.get("gazette_date", ""),
                        "ministry": doc_meta.get("ministry", ""),
                        "notification_no": doc_meta.get("notification_no", ""),
                        "language_mix": doc_meta.get("language_mix", ""),
                        "page_start": clause["page_start"],
                        "page_end": clause["page_end"],
                        "text": chunk_text,
                        "retrieval_text": retrieval_text,
                        "char_count": len(chunk_text),
                    }
                )

    return all_chunks


def run_chunking(
    documents_path: Path = DOCUMENTS_JSONL_PATH,
    clauses_path: Path = CLAUSES_JSONL_PATH,
    output_path: Path = CHUNKS_JSONL_PATH,
) -> list[dict[str, Any]]:
    """Load cleaned Phase 1 outputs, chunk them, and write chunks.jsonl."""
    documents = read_jsonl(documents_path)
    clauses = read_jsonl(clauses_path)
    chunks = build_clause_chunks(documents=documents, clauses=clauses)
    write_jsonl(output_path, chunks)
    return chunks
