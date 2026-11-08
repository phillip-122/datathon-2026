"""
SIGNAL Lakeflow Spark Declarative Pipeline (DLT)
Automated Medallion Ingestion & Benchmark Quality Enforcement for AI Landscape Observatory
"""

import dlt
from pyspark.sql.functions import col, current_timestamp

# =========================================================================
# 1. BRONZE LAYER: Raw AI Announcements & Benchmark Landing
# =========================================================================
@dlt.table(
    name="bronze_ai_events_raw",
    comment="Raw landed AI model announcements, preprints, and benchmark JSON runs.",
    table_properties={"quality": "bronze"}
)
def bronze_ai_events_raw():
    return spark.readStream.table("workspace.signal_bronze.raw_events_landing")


# =========================================================================
# 2. SILVER LAYER: Verified Models, Normalized Pricing, and Evaluation Quality
# =========================================================================
@dlt.table(
    name="silver_models",
    comment="Conformed frontier & open-weights models with verified context windows and pricing.",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_fail("valid_model_id", "model_id IS NOT NULL AND length(model_id) > 1")
@dlt.expect_or_drop("valid_context", "context_window >= 1024")
@dlt.expect_or_drop("valid_pricing", "pricing_output_per_m >= 0.0 AND pricing_input_per_m >= 0.0")
def silver_models():
    return (
        dlt.read_stream("bronze_ai_events_raw")
        .select(
            col("model_id"),
            col("name"),
            col("provider"),
            col("release_date"),
            col("architecture"),
            col("parameter_count_b").cast("double"),
            col("context_window").cast("int"),
            col("license"),
            col("is_open_weights").cast("boolean"),
            col("pricing_input_per_m").cast("double"),
            col("pricing_output_per_m").cast("double"),
            current_timestamp().alias("ingestion_timestamp")
        )
    )


@dlt.table(
    name="silver_evaluations",
    comment="Verified benchmark evaluations with methodology checks.",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("score_bounds", "score >= 0.0 AND score <= 2000.0")
@dlt.expect_or_fail("model_reference", "model_id IS NOT NULL")
@dlt.expect_or_fail("benchmark_reference", "benchmark_id IS NOT NULL")
def silver_evaluations():
    return (
        dlt.read("bronze_ai_events_raw")
        .select(
            col("eval_id"),
            col("model_id"),
            col("benchmark_id"),
            col("score").cast("double"),
            col("eval_date"),
            col("methodology")
        )
    )


# =========================================================================
# 3. GOLD LAYER: Pareto Frontier & Model Comparison Analytical Views
# =========================================================================
@dlt.table(
    name="gold_pareto_frontier",
    comment="Materialized Pareto efficient frontier across reasoning, coding, and blended price.",
    table_properties={"quality": "gold"}
)
def gold_pareto_frontier():
    return (
        spark.read.table("workspace.signal_gold.v_model_comparisons")
    )
