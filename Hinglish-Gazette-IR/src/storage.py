"""Persistence helpers for JSON, JSONL, and SQLite outputs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON object to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write an iterable of dictionaries as UTF-8 JSON Lines and return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON Lines file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def save_to_sqlite(
    db_path: Path,
    documents: list[dict[str, Any]],
    clauses: list[dict[str, Any]],
    pages: list[dict[str, Any]],
) -> None:
    """Persist cleaned documents, clauses, and pages into a fresh SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                category TEXT,
                gazette_date TEXT,
                ministry TEXT,
                department TEXT,
                notification_no TEXT,
                language_mix TEXT,
                page_count INTEGER NOT NULL,
                native_text_pages INTEGER NOT NULL,
                ocr_pages INTEGER NOT NULL,
                empty_pages INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                clean_text TEXT NOT NULL
            );

            CREATE TABLE clauses (
                clause_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                clause_index INTEGER NOT NULL,
                clause_label TEXT,
                heading TEXT,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            );

            CREATE TABLE pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                extraction_method TEXT NOT NULL,
                raw_char_count INTEGER NOT NULL,
                clean_char_count INTEGER NOT NULL,
                has_devanagari INTEGER NOT NULL,
                clean_text TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            );

            CREATE INDEX idx_clauses_doc_id ON clauses (doc_id);
            CREATE INDEX idx_pages_doc_id ON pages (doc_id);
            """
        )

        cursor.executemany(
            """
            INSERT INTO documents (
                doc_id, filename, title, category, gazette_date, ministry,
                department, notification_no, language_mix, page_count,
                native_text_pages, ocr_pages, empty_pages, char_count, clean_text
            ) VALUES (
                :doc_id, :filename, :title, :category, :gazette_date, :ministry,
                :department, :notification_no, :language_mix, :page_count,
                :native_text_pages, :ocr_pages, :empty_pages, :char_count, :clean_text
            )
            """,
            documents,
        )

        cursor.executemany(
            """
            INSERT INTO clauses (
                clause_id, doc_id, clause_index, clause_label, heading,
                page_start, page_end, text, char_count
            ) VALUES (
                :clause_id, :doc_id, :clause_index, :clause_label, :heading,
                :page_start, :page_end, :text, :char_count
            )
            """,
            clauses,
        )

        cursor.executemany(
            """
            INSERT INTO pages (
                doc_id, page_number, extraction_method, raw_char_count,
                clean_char_count, has_devanagari, clean_text
            ) VALUES (
                :doc_id, :page_number, :extraction_method, :raw_char_count,
                :clean_char_count, :has_devanagari, :clean_text
            )
            """,
            pages,
        )

        conn.commit()
    finally:
        conn.close()
