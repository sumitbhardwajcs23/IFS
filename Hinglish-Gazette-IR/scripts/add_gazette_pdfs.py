"""Helper script to register newly added Gazette PDFs or import from a URL/CSV list."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MANIFEST_PATH, RAW_PDF_DIR
from src.phase1_dataset import dataset_target_progress, sync_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize data/raw/*.pdf with data/manifest.csv and show 100-PDF target progress."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    df = sync_manifest(raw_dir=args.raw_dir, manifest_path=args.manifest, project_root=PROJECT_ROOT)
    progress = dataset_target_progress(df)
    print(f"Manifest synced: {args.manifest}")
    print(f"Collected PDFs: {progress['total_current']} / {progress['total_target']}")
    if progress["unclassified"]:
        print(f"Unclassified PDFs: {progress['unclassified']}")
    for item in progress["breakdown"]:
        print(
            f"  - {item['category']}: {item['current']} / {item['target']} "
            f"(remaining: {item['remaining']})"
        )


if __name__ == "__main__":
    main()
