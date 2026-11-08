"""
SIGNAL Pipeline Orchestration Runner
Coordinates end-to-end execution:
1. Ingestion of raw feeds into Bronze (with content hashing and deduplication)
2. Normalization, entity extraction, reference validation into Silver
3. Generation and materialization of Gold analytical views
4. Exporting to local DuckDB, Parquet files, or Databricks Unity Catalog SQL DDL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb

from SIGNAL.pipelines.bronze_pipeline import BronzePipeline
from SIGNAL.pipelines.silver_pipeline import SilverPipeline
from SIGNAL.pipelines.gold_pipeline import GoldPipeline
from SIGNAL.pipelines.seed_data import get_curated_seed_records

logger = logging.getLogger("signal.pipeline.runner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "signal.duckdb"
DEFAULT_GOLD_EXPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "gold"


class SignalPipelineRunner:
    """
    Unified end-to-end pipeline runner for SIGNAL.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None, connection: Optional[duckdb.DuckDBPyConnection] = None):
        if connection is not None:
            self.con = connection
        elif db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(str(db_path))
        else:
            self.con = duckdb.connect(":memory:")

        self.bronze = BronzePipeline(connection=self.con)
        self.silver = SilverPipeline(connection=self.con)
        self.gold = GoldPipeline(connection=self.con)

    def run_full_pipeline(
        self,
        raw_records: Optional[List[Dict[str, Any]]] = None,
        raw_files_dir: Optional[Union[str, Path]] = None,
        export_parquet_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline:
        1. Bronze ingestion (from raw_records or raw_files_dir)
        2. Silver normalization
        3. Gold views & materialization
        """
        logger.info("--- Starting SIGNAL End-to-End Pipeline Run ---")

        # 1. Bronze ingestion
        bronze_stats = {"total": 0, "inserted": 0, "skipped_duplicates": 0}
        if raw_files_dir:
            r_dir = Path(raw_files_dir)
            if r_dir.exists():
                for p in r_dir.glob("*.json*"):
                    res = self.bronze.ingest_file(p)
                    bronze_stats["total"] += res["total"]
                    bronze_stats["inserted"] += res["inserted"]
                    bronze_stats["skipped_duplicates"] += res["skipped_duplicates"]

        if raw_records:
            res = self.bronze.ingest_batch(raw_records)
            bronze_stats["total"] += res["total"]
            bronze_stats["inserted"] += res["inserted"]
            bronze_stats["skipped_duplicates"] += res["skipped_duplicates"]

        # 2. Silver processing
        silver_stats = self.silver.process_bronze_records()

        # 3. Gold views & materialization
        self.gold.init_views()
        self.gold.materialize_gold_tables()

        parquet_paths = {}
        if export_parquet_dir:
            parquet_paths = self.gold.export_gold_parquet(export_parquet_dir)

        summary_metrics = self.gold.get_metrics()

        logger.info("--- Completed SIGNAL End-to-End Pipeline Run Successfully ---")
        return {
            "status": "SUCCESS",
            "bronze": bronze_stats,
            "silver": silver_stats,
            "metrics": summary_metrics,
            "parquet_exports": parquet_paths,
        }

    def seed_initial_data(self) -> Dict[str, Any]:
        """Hydrates the pipeline with realistic curated AI landscape data."""
        logger.info("Hydrating pipeline with curated AI landscape seed records...")
        seed_records = get_curated_seed_records()
        return self.run_full_pipeline(raw_records=seed_records, export_parquet_dir=DEFAULT_GOLD_EXPORT_DIR)


def main():
    parser = argparse.ArgumentParser(description="SIGNAL AI Landscape Data Pipeline Runner")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="DuckDB database file path")
    parser.add_argument("--seed", action="store_true", help="Seed with curated realistic AI landscape data")
    parser.add_argument("--raw-dir", default=None, help="Directory containing raw JSON/JSONL feed payloads")
    parser.add_argument("--export-parquet", default=str(DEFAULT_GOLD_EXPORT_DIR), help="Export Gold views to Parquet")
    args = parser.parse_args()

    runner = SignalPipelineRunner(db_path=args.db_path)
    if args.seed:
        result = runner.seed_initial_data()
        print(f"Pipeline seeded successfully: {result['metrics']}")
    elif args.raw_dir:
        result = runner.run_full_pipeline(raw_files_dir=args.raw_dir, export_parquet_dir=args.export_parquet)
        print(f"Pipeline finished: {result['metrics']}")
    else:
        # Default behavior: run silver and gold refresh on existing DB
        result = runner.run_full_pipeline(export_parquet_dir=args.export_parquet)
        print(f"Pipeline refreshed: {result['metrics']}")


if __name__ == "__main__":
    main()
