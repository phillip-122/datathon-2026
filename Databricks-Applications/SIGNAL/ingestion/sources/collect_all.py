"""
SIGNAL Unified Ingestion Runner
Orchestrates collection across arXiv, Hugging Face, and GitHub sources.
Generates raw payloads into SIGNAL/data/raw/ and publishes an ingestion manifest.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

from .arxiv_collector import ArxivCollector
from .huggingface_collector import HuggingFaceCollector
from .github_collector import GitHubCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SIGNAL.collect_all")


def run_all_collectors(
    mode: str = "mock",
    base_raw_dir: Path | str | None = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Executes all three SIGNAL collectors in sequence and records an ingestion manifest.
    """
    base_path = (
        Path(base_raw_dir)
        if base_raw_dir
        else Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    )
    base_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    manifest: Dict[str, Any] = {
        "manifest_version": "1.0.0",
        "started_at": start_time.isoformat(),
        "mode": mode,
        "sources": {},
        "summary": {
            "total_records": 0,
            "successful_sources": 0,
            "failed_sources": 0,
        },
    }

    # 1. arXiv Collector
    try:
        logger.info(f"--- Running ArxivCollector (mode={mode}) ---")
        arxiv_dir = base_path / "arxiv"
        arxiv_col = ArxivCollector(output_dir=arxiv_dir, mode=mode)
        arxiv_res = arxiv_col.collect(max_results=10, save=save)
        count = arxiv_res["metadata"]["record_count"]
        manifest["sources"]["arxiv"] = {
            "status": "SUCCESS",
            "record_count": count,
            "output_dir": str(arxiv_dir),
            "timestamp": arxiv_res["metadata"]["collection_timestamp"],
        }
        manifest["summary"]["total_records"] += count
        manifest["summary"]["successful_sources"] += 1
    except Exception as e:
        logger.error(f"ArxivCollector failed: {e}", exc_info=True)
        manifest["sources"]["arxiv"] = {"status": "ERROR", "error": str(e)}
        manifest["summary"]["failed_sources"] += 1

    # 2. Hugging Face Collector
    try:
        logger.info(f"--- Running HuggingFaceCollector (mode={mode}) ---")
        hf_dir = base_path / "huggingface"
        hf_col = HuggingFaceCollector(output_dir=hf_dir, mode=mode)
        hf_res = hf_col.collect(limit=10, save=save)
        count = hf_res["metadata"]["record_count"]
        manifest["sources"]["huggingface"] = {
            "status": "SUCCESS",
            "record_count": count,
            "output_dir": str(hf_dir),
            "timestamp": hf_res["metadata"]["collection_timestamp"],
        }
        manifest["summary"]["total_records"] += count
        manifest["summary"]["successful_sources"] += 1
    except Exception as e:
        logger.error(f"HuggingFaceCollector failed: {e}", exc_info=True)
        manifest["sources"]["huggingface"] = {"status": "ERROR", "error": str(e)}
        manifest["summary"]["failed_sources"] += 1

    # 3. GitHub Collector
    try:
        logger.info(f"--- Running GitHubCollector (mode={mode}) ---")
        gh_dir = base_path / "github"
        gh_col = GitHubCollector(output_dir=gh_dir, mode=mode)
        gh_res = gh_col.collect(limit=9, save=save)
        count = gh_res["metadata"]["record_count"]
        manifest["sources"]["github"] = {
            "status": "SUCCESS",
            "record_count": count,
            "output_dir": str(gh_dir),
            "timestamp": gh_res["metadata"]["collection_timestamp"],
        }
        manifest["summary"]["total_records"] += count
        manifest["summary"]["successful_sources"] += 1
    except Exception as e:
        logger.error(f"GitHubCollector failed: {e}", exc_info=True)
        manifest["sources"]["github"] = {"status": "ERROR", "error": str(e)}
        manifest["summary"]["failed_sources"] += 1

    end_time = datetime.now(timezone.utc)
    manifest["finished_at"] = end_time.isoformat()
    manifest["duration_seconds"] = (end_time - start_time).total_seconds()

    if save:
        manifest_path = base_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved ingestion manifest to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Run all SIGNAL ingestion collectors.")
    parser.add_argument("--mode", choices=["live", "mock"], default="mock", help="Collector execution mode")
    parser.add_argument("--raw-dir", type=str, default=None, help="Base directory for raw data")
    args = parser.parse_args()

    manifest = run_all_collectors(mode=args.mode, base_raw_dir=args.raw_dir)
    print("\n=== INGESTION RUN COMPLETE ===")
    print(f"Mode: {manifest['mode']}")
    print(f"Total Records: {manifest['summary']['total_records']}")
    print(f"Successful Sources: {manifest['summary']['successful_sources']}")
    print(f"Failed Sources: {manifest['summary']['failed_sources']}")


if __name__ == "__main__":
    main()
