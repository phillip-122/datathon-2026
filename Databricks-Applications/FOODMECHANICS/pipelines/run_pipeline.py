"""End-to-End Pipeline Runner for FOODMECHANICS.

Executes Bronze Ingestion -> Silver Normalization (100g basis) -> Gold Analytical Views.
Persists the final database to data/processed/foodmechanics.duckdb.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

from FOODMECHANICS.pipelines.bronze_pipeline import ingest_fixture_files
from FOODMECHANICS.pipelines.silver_pipeline import run_silver_pipeline
from FOODMECHANICS.pipelines.gold_pipeline import (
    create_gold_views,
    get_ingredient_comparisons,
    get_formulation_tradeoffs,
    get_foam_stability_factors,
)

logger = logging.getLogger("foodmechanics.runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_pipeline(db_path: Path, fixtures_dir: Path, rebuild: bool = False) -> None:
    """Run full pipeline."""
    if rebuild and db_path.exists() and db_path.is_file():
        logger.info("Rebuilding database: removing %s", db_path)
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Opening database at %s ...", db_path)
    con = duckdb.connect(str(db_path))

    try:
        # Step 1: Bronze Ingestion
        logger.info("=== STEP 1: BRONZE INGESTION ===")
        b_stats = ingest_fixture_files(con, fixtures_dir=fixtures_dir)
        total_bronze = con.execute("SELECT COUNT(*) FROM bronze_food_raw").fetchone()[0]
        logger.info("Bronze total records: %d (%s)", total_bronze, b_stats)

        # Step 2: Silver Normalization
        logger.info("=== STEP 2: SILVER NORMALIZATION (STRICT 100g BASIS) ===")
        s_stats = run_silver_pipeline(con)
        for tbl, cnt in s_stats.items():
            logger.info("  %s: %d rows", tbl, cnt)

        # Step 3: Gold Analytical Views
        logger.info("=== STEP 3: GOLD ANALYTICAL VIEWS ===")
        create_gold_views(con)

        # Step 4: Verification and Analytics Snapshot
        logger.info("=== STEP 4: ANALYTICS VALIDATION ===")
        ing_comps = get_ingredient_comparisons(con)
        form_trades = get_formulation_tradeoffs(con)
        foam_factors = get_foam_stability_factors(con)

        logger.info("Verified %d ingredient comparisons on 100g basis.", len(ing_comps))
        logger.info("Verified %d formulation tradeoff simulations.", len(form_trades))
        logger.info("Verified %d foam stability factor profiles.", len(foam_factors))

        print("\n" + "=" * 70)
        print(" FOODMECHANICS PIPELINE SUMMARY REPORT")
        print("=" * 70)
        print(f"Database: {db_path.resolve()}")
        print(f"Bronze Raw Records: {total_bronze}")
        print(f"Silver Ingredients: {s_stats['silver_ingredients']}")
        print(f"Silver Nutritional Profiles (100g basis): {s_stats['silver_nutritional_profiles_100g']}")
        print(f"Silver Functional Properties: {s_stats['silver_functional_properties']}")
        print(f"Silver Chemical Compounds: {s_stats['silver_chemical_compounds']}")
        print(f"Silver Food Mechanisms: {s_stats['silver_food_mechanisms']}")
        print(f"Silver Mechanism Relationships: {s_stats['silver_mechanism_relationships']}")
        print(f"Silver Formulations: {s_stats['silver_formulations']}")
        print(f"Silver Scientific Evidence: {s_stats['silver_scientific_evidence']}")
        print("-" * 70)
        print("SAMPLE FORMULATION TRADEOFF ANALYSIS:")
        for ft in form_trades[:4]:
            print(f"  • {ft['recipe_name']}: {ft['norm_protein_g_per_100g']}g protein, {ft['norm_fat_g_per_100g']}g fat | Stability Score: {ft['estimated_foam_stability_index']} ({ft['stability_classification']})")
        print("=" * 70 + "\n")

    finally:
        con.close()
        logger.info("Database closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FOODMECHANICS Medallion Pipeline Runner")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "foodmechanics.duckdb",
        help="Path to DuckDB database file",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "fixtures",
        help="Path to raw JSON fixtures directory",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe and recreate database from scratch",
    )
    args = parser.parse_args()
    run_pipeline(db_path=args.db_path, fixtures_dir=args.fixtures_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
