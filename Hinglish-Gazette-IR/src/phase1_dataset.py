"""Phase 1: Dataset discovery, manifest management, and 100-PDF target tracking."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CATEGORY_TARGETS, MANIFEST_PATH, RAW_PDF_DIR

MANIFEST_COLUMNS = [
    "doc_id",
    "filename",
    "relative_path",
    "file_size_bytes",
    "sha256_prefix",
    "inferred_date",
    "category",
    "title",
    "ministry",
    "notification_no",
    "page_count",
    "extraction_status",
    "source_url",
    "notes",
]

DATE_FROM_FILENAME_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")


def compute_sha256_prefix(file_path: Path, length: int = 12) -> str:
    """Return the first `length` hex characters of a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def infer_date_from_filename(filename: str) -> str:
    """Infer ISO date (YYYY-MM-DD) from filenames like C_202609141739155022.pdf."""
    match = DATE_FROM_FILENAME_RE.search(filename)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def sync_manifest(
    raw_dir: Path = RAW_PDF_DIR,
    manifest_path: Path = MANIFEST_PATH,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Scan `raw_dir` for PDFs and synchronize `manifest.csv`, preserving manual labels."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or raw_dir.parent.parent

    existing_by_filename: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path, dtype=str).fillna("")
        for row in existing_df.to_dict(orient="records"):
            existing_by_filename[row["filename"]] = row

    pdf_files = sorted(raw_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    rows: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        prev = existing_by_filename.get(pdf_path.name, {})
        doc_id = prev.get("doc_id") or pdf_path.stem
        try:
            rel_path = str(pdf_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel_path = str(pdf_path).replace("\\", "/")

        rows.append(
            {
                "doc_id": doc_id,
                "filename": pdf_path.name,
                "relative_path": rel_path,
                "file_size_bytes": str(pdf_path.stat().st_size),
                "sha256_prefix": compute_sha256_prefix(pdf_path),
                "inferred_date": prev.get("inferred_date") or infer_date_from_filename(pdf_path.name),
                "category": prev.get("category") or "Unclassified",
                "title": prev.get("title") or "",
                "ministry": prev.get("ministry") or "",
                "notification_no": prev.get("notification_no") or "",
                "page_count": prev.get("page_count") or "",
                "extraction_status": prev.get("extraction_status") or "pending",
                "source_url": prev.get("source_url") or "",
                "notes": prev.get("notes") or "",
            }
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    return df


def update_manifest_with_cleaned_docs(
    cleaned_docs: list[dict[str, Any]],
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    """Update manifest metadata (category, title, ministry, page_count, status) from cleaned docs."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path, dtype=str).fillna("")
    doc_map = {doc["doc_id"]: doc for doc in cleaned_docs}

    for idx, row in df.iterrows():
        doc = doc_map.get(row["doc_id"])
        if not doc:
            continue
        if not row["category"] or row["category"] == "Unclassified":
            df.at[idx, "category"] = doc.get("category") or "Unclassified"
        if doc.get("title"):
            df.at[idx, "title"] = doc["title"]
        if doc.get("ministry"):
            df.at[idx, "ministry"] = doc["ministry"]
        if doc.get("notification_no"):
            df.at[idx, "notification_no"] = doc["notification_no"]
        df.at[idx, "page_count"] = str(doc.get("page_count", ""))
        df.at[idx, "extraction_status"] = "cleaned"

    df.to_csv(manifest_path, index=False, encoding="utf-8")
    return df


def dataset_target_progress(manifest_df: pd.DataFrame) -> dict[str, Any]:
    """Compare current collected PDFs against the 100-PDF Phase 1 target breakdown."""
    counts = manifest_df["category"].value_counts().to_dict() if not manifest_df.empty else {}
    breakdown: list[dict[str, Any]] = []
    for category, target in CATEGORY_TARGETS.items():
        current = int(counts.get(category, 0))
        breakdown.append(
            {
                "category": category,
                "current": current,
                "target": target,
                "remaining": max(0, target - current),
            }
        )
    unclassified = int(counts.get("Unclassified", 0))
    total_current = int(len(manifest_df))
    total_target = sum(CATEGORY_TARGETS.values())
    return {
        "total_current": total_current,
        "total_target": total_target,
        "unclassified": unclassified,
        "breakdown": breakdown,
    }
