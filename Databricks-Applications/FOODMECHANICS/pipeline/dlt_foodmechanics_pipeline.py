"""
FOODMECHANICS Lakeflow Spark Declarative Pipeline (DLT)
Automated Medallion Ingestion & Quality Control for USDA 100g Foundation Data & Food Rheology
"""

import dlt
from pyspark.sql.functions import col, current_timestamp, round as spark_round

# =========================================================================
# 1. BRONZE LAYER: Raw Data Ingestion with Append-Flow Preservation
# =========================================================================
@dlt.table(
    name="bronze_usda_raw",
    comment="Raw landed USDA FoodData Central and laboratory colloid records.",
    table_properties={"quality": "bronze"}
)
def bronze_usda_raw():
    return spark.readStream.table("workspace.foodmechanics_bronze.raw_usda_landing")


# =========================================================================
# 2. SILVER LAYER: 100g Mass Normalization & Data Quality Contracts
# =========================================================================
@dlt.table(
    name="silver_ingredients",
    comment="Conformed food ingredients with physical property metadata.",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_fail("valid_ingredient_id", "ingredient_id IS NOT NULL AND length(ingredient_id) > 2")
@dlt.expect_or_drop("valid_density", "density_g_ml > 0.1 AND density_g_ml < 3.0")
def silver_ingredients():
    return (
        dlt.read_stream("bronze_usda_raw")
        .select(
            col("ingredient_id"),
            col("name"),
            col("common_name"),
            col("category"),
            col("default_serving_g").cast("double"),
            col("density_g_ml").cast("double"),
            col("description"),
            col("source_id"),
            current_timestamp().alias("conformed_timestamp")
        )
    )


@dlt.table(
    name="silver_nutritional_profiles",
    comment="Strictly normalized 100g standard nutritional basis profiles with conservation law checks.",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("valid_basis", "basis_amount_g = 100.0")
@dlt.expect_or_drop("valid_protein", "protein_g >= 0.0 AND protein_g <= 100.0")
@dlt.expect_or_drop("valid_fat", "fat_total_g >= 0.0 AND fat_total_g <= 100.0")
@dlt.expect_or_drop("valid_carbs", "carb_g >= 0.0 AND carb_g <= 100.0")
@dlt.expect_or_drop("macro_conservation", "(protein_g + fat_total_g + carb_g + water_g) <= 105.0")
def silver_nutritional_profiles():
    return (
        dlt.read("bronze_usda_raw")
        .select(
            col("profile_id"),
            col("ingredient_id"),
            col("basis_amount_g").cast("double"),
            col("energy_kcal").cast("double"),
            col("protein_g").cast("double"),
            col("fat_total_g").cast("double"),
            col("fat_saturated_g").cast("double"),
            col("carb_g").cast("double"),
            col("sugars_g").cast("double"),
            col("fiber_g").cast("double"),
            col("sodium_mg").cast("double"),
            col("water_g").cast("double"),
            col("source_id")
        )
    )


# =========================================================================
# 3. GOLD LAYER: Analytical Colloid Simulations & Materialized Views
# =========================================================================
@dlt.table(
    name="gold_formulation_simulations",
    comment="Pre-calculated thermodynamic formulation simulations for sub-second UI sweeps.",
    table_properties={"quality": "gold"}
)
@dlt.expect_or_drop("positive_viscosity", "calculated_continuous_viscosity_pa_s > 0.0")
@dlt.expect_or_drop("valid_drainage_half_life", "predicted_drainage_half_life_min >= 0.0")
def gold_formulation_simulations():
    return (
        spark.read.table("workspace.foodmechanics_gold.gold_formulation_simulations")
    )
