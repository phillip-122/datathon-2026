"""FOODMECHANICS Pipelines Package."""

from FOODMECHANICS.pipelines.bronze_pipeline import (
    compute_content_hash,
    init_bronze_tables,
    ingest_raw_record,
    ingest_fixture_files,
    get_bronze_records,
)
from FOODMECHANICS.pipelines.silver_pipeline import (
    init_silver_tables,
    normalize_nutrition_100g,
    run_silver_pipeline,
)
from FOODMECHANICS.pipelines.gold_pipeline import (
    create_gold_views,
    get_ingredient_comparisons,
    get_mechanism_evidence,
    get_formulation_tradeoffs,
    get_foam_stability_factors,
    simulate_substitution,
)

__all__ = [
    "compute_content_hash",
    "init_bronze_tables",
    "ingest_raw_record",
    "ingest_fixture_files",
    "get_bronze_records",
    "init_silver_tables",
    "normalize_nutrition_100g",
    "run_silver_pipeline",
    "create_gold_views",
    "get_ingredient_comparisons",
    "get_mechanism_evidence",
    "get_formulation_tradeoffs",
    "get_foam_stability_factors",
    "simulate_substitution",
]
