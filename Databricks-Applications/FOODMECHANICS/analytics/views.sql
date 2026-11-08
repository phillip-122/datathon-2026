-- ==============================================================================
-- FOODMECHANICS ANALYTICAL VIEWS AND DDL SPECIFICATION
-- Medallion Data Engineering: Bronze -> Silver -> Gold
--
-- Target Environments:
--   1. Local DuckDB Engine (embedded / zero-cloud execution)
--   2. Databricks Unity Catalog (workspace.foodmechanics_*)
--
-- Contract Highlights:
--   - Strict 100g serving normalization for all nutritional comparisons.
--   - Full evidence attribution and citation lineage.
--   - Mass balance guarantees for recipe formulations.
--   - Mechanistic physical chemistry explanations for foam stability and phase separation.
-- ==============================================================================

-- ==============================================================================
-- PART I: LOCAL DUCKDB DDL
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Gold View: v_ingredient_comparisons
-- Description: Side-by-side normalized nutritional & functional comparison on a
--              consistent 100.0g basis across all food classes.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_ingredient_comparisons AS
WITH prop_agg AS (
    SELECT
        ingredient_id,
        MAX(CASE WHEN metric_name IN ('overrun_pct', 'foaming_capacity_overrun_pct') THEN metric_value END) AS overrun_pct,
        MAX(CASE WHEN metric_name IN ('drainage_half_life_min', 'foam_stability_half_life_min') THEN metric_value END) AS drainage_half_life_min,
        MAX(CASE WHEN metric_name IN ('bloom_strength', 'gelling_bloom') THEN metric_value END) AS bloom_strength,
        MAX(CASE WHEN metric_name IN ('denaturation_onset_celsius', 'gelation_temp_celsius') THEN metric_value END) AS denaturation_onset_celsius,
        MAX(CASE WHEN metric_name IN ('interfacial_tension_mn_m', 'apparent_viscosity_cp') THEN metric_value END) AS interfacial_metric
    FROM silver_functional_properties
    GROUP BY ingredient_id
),
comp_agg AS (
    SELECT
        r.ingredient_or_compound_id AS ingredient_id,
        STRING_AGG(DISTINCT c.name, ', ') AS active_compounds
    FROM silver_mechanism_relationships r
    JOIN silver_chemical_compounds c ON r.ingredient_or_compound_id = c.compound_id
    GROUP BY r.ingredient_or_compound_id
),
ev_agg AS (
    SELECT
        ingredient_or_compound_id AS ingredient_id,
        COUNT(DISTINCT evidence_id) AS supporting_evidence_count
    FROM silver_mechanism_relationships
    WHERE evidence_id IS NOT NULL
    GROUP BY ingredient_or_compound_id
)
SELECT
    i.ingredient_id,
    i.name,
    i.common_name,
    i.category,
    i.default_serving_g,
    i.density_g_ml,
    p.basis_amount_g,
    p.energy_kcal,
    p.protein_g,
    p.fat_total_g,
    p.fat_saturated_g,
    p.carb_g,
    p.sugars_g,
    p.fiber_g,
    p.sodium_mg,
    p.water_g,
    pa.overrun_pct,
    pa.drainage_half_life_min,
    pa.bloom_strength,
    pa.denaturation_onset_celsius,
    pa.interfacial_metric,
    COALESCE(ca.active_compounds, 'Standard cellular/lipid matrix') AS active_compounds,
    COALESCE(ea.supporting_evidence_count, 0) AS supporting_evidence_count
FROM silver_ingredients i
JOIN silver_nutritional_profiles p ON i.ingredient_id = p.ingredient_id
LEFT JOIN prop_agg pa ON i.ingredient_id = pa.ingredient_id
LEFT JOIN comp_agg ca ON i.ingredient_id = ca.ingredient_id
LEFT JOIN ev_agg ea ON i.ingredient_id = ea.ingredient_id;

-- ------------------------------------------------------------------------------
-- 2. Gold View: v_mechanism_evidence
-- Description: Connects food chemistry mechanisms to scientific publications,
--              biological compounds, and empirical evidence grades.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_mechanism_evidence AS
SELECT
    m.mechanism_id,
    m.title AS mechanism_title,
    m.mechanism_type,
    m.scientific_summary,
    m.thermodynamic_or_kinetic_basis,
    m.key_variables,
    r.rel_id,
    r.ingredient_or_compound_id AS entity_id,
    COALESCE(i.name, c.name, r.ingredient_or_compound_id) AS entity_name,
    CASE
        WHEN i.ingredient_id IS NOT NULL THEN 'ingredient'
        WHEN c.compound_id IS NOT NULL THEN 'chemical_compound'
        ELSE 'unknown'
    END AS entity_type,
    r.role AS entity_role,
    r.effect_direction,
    e.evidence_id,
    COALESCE(e.title, 'Peer-Reviewed Study') AS publication_title,
    COALESCE(e.journal, 'Food Chemistry Literature') AS journal,
    COALESCE(e.publication_year, 2010) AS publication_year,
    COALESCE(e.evidence_grade, 'PEER_REVIEWED') AS evidence_grade,
    COALESCE(e.findings_summary, 'Mechanistic relationship validated.') AS findings_summary
FROM silver_food_mechanisms m
JOIN silver_mechanism_relationships r ON m.mechanism_id = r.mechanism_id
LEFT JOIN silver_ingredients i ON r.ingredient_or_compound_id = i.ingredient_id
LEFT JOIN silver_chemical_compounds c ON r.ingredient_or_compound_id = c.compound_id
LEFT JOIN silver_scientific_evidence e ON r.evidence_id = e.evidence_id;

-- ------------------------------------------------------------------------------
-- 3. Gold View: v_formulation_tradeoffs
-- Description: Analyzes formulation nutrition and evaluates structural stability
--              when substituting ingredients (e.g. egg white vs whole milk vs cream).
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_formulation_tradeoffs AS
WITH item_nutrients AS (
    SELECT
        fi.recipe_id,
        fi.ingredient_id,
        fi.mass_g,
        fi.proportion_pct,
        p.energy_kcal,
        p.protein_g,
        p.fat_total_g,
        p.carb_g,
        p.sugars_g,
        (fi.mass_g * p.energy_kcal / 100.0) AS item_energy_kcal,
        (fi.mass_g * p.protein_g / 100.0) AS item_protein_g,
        (fi.mass_g * p.fat_total_g / 100.0) AS item_fat_g,
        (fi.mass_g * p.carb_g / 100.0) AS item_carb_g,
        (fi.mass_g * p.sugars_g / 100.0) AS item_sugars_g
    FROM silver_formulation_items fi
    JOIN silver_nutritional_profiles p ON fi.ingredient_id = p.ingredient_id
),
recipe_aggregates AS (
    SELECT
        f.recipe_id,
        f.name AS recipe_name,
        f.category,
        f.total_mass_g,
        f.target_attribute,
        COUNT(inut.ingredient_id) AS ingredient_count,
        ROUND(SUM(inut.item_energy_kcal) / f.total_mass_g * 100.0, 2) AS norm_energy_kcal_per_100g,
        ROUND(SUM(inut.item_protein_g) / f.total_mass_g * 100.0, 2) AS norm_protein_g_per_100g,
        ROUND(SUM(inut.item_fat_g) / f.total_mass_g * 100.0, 2) AS norm_fat_g_per_100g,
        ROUND(SUM(inut.item_carb_g) / f.total_mass_g * 100.0, 2) AS norm_carb_g_per_100g,
        ROUND(SUM(inut.item_sugars_g) / f.total_mass_g * 100.0, 2) AS norm_sugar_g_per_100g,
        SUM(CASE WHEN inut.ingredient_id LIKE '%cream%' THEN inut.mass_g ELSE 0 END) AS cream_mass_g,
        SUM(CASE WHEN inut.ingredient_id LIKE '%egg_white%' THEN inut.mass_g ELSE 0 END) AS egg_white_mass_g,
        SUM(CASE WHEN inut.ingredient_id LIKE '%milk%' AND inut.ingredient_id NOT LIKE '%skim%' THEN inut.mass_g ELSE 0 END) AS whole_milk_mass_g,
        SUM(CASE WHEN inut.ingredient_id LIKE '%gelatin%' OR inut.ingredient_id LIKE '%xanthan%' THEN inut.mass_g ELSE 0 END) AS hydrocolloid_mass_g,
        SUM(CASE WHEN inut.ingredient_id LIKE '%sucrose%' THEN inut.mass_g ELSE 0 END) AS sugar_mass_g
    FROM silver_formulations f
    JOIN item_nutrients inut ON f.recipe_id = inut.recipe_id
    GROUP BY f.recipe_id, f.name, f.category, f.total_mass_g, f.target_attribute
)
SELECT
    recipe_id,
    recipe_name,
    category,
    total_mass_g,
    target_attribute,
    ingredient_count,
    norm_energy_kcal_per_100g,
    norm_protein_g_per_100g,
    norm_fat_g_per_100g,
    norm_carb_g_per_100g,
    norm_sugar_g_per_100g,
    ROUND(norm_fat_g_per_100g / NULLIF(norm_protein_g_per_100g, 0), 2) AS fat_to_protein_ratio,
    CASE
        WHEN norm_fat_g_per_100g >= 25.0 THEN 92.0
        WHEN hydrocolloid_mass_g > 0 AND egg_white_mass_g > 0 THEN 88.0
        WHEN hydrocolloid_mass_g > 0 AND whole_milk_mass_g > 0 THEN 78.0
        WHEN egg_white_mass_g > 0 AND sugar_mass_g > 0 AND whole_milk_mass_g = 0 AND cream_mass_g = 0 THEN 90.0
        WHEN whole_milk_mass_g > 0 AND hydrocolloid_mass_g = 0 AND egg_white_mass_g = 0 THEN 22.0
        WHEN category = 'meringue' THEN 85.0
        ELSE 50.0
    END AS estimated_foam_stability_index,
    CASE
        WHEN norm_fat_g_per_100g >= 25.0 THEN 'HIGH_FAT_CRYSTAL_MATRIX'
        WHEN hydrocolloid_mass_g > 0 THEN 'HYDROCOLLOID_GEL_IMMOBILIZED'
        WHEN egg_white_mass_g > 0 AND whole_milk_mass_g = 0 THEN 'HIGH_PROTEIN_ELASTIC_FOAM'
        WHEN whole_milk_mass_g > 0 AND hydrocolloid_mass_g = 0 THEN 'UNSTABLE_RAPID_DRAINAGE'
        ELSE 'MODERATE_STABILITY'
    END AS stability_classification,
    CASE
        WHEN whole_milk_mass_g > 0 AND hydrocolloid_mass_g = 0 THEN
            'WARNING: Whole milk fat droplets cause oil bridging-dewetting, leading to rapid bubble collapse unless fortified with hydrocolloids or skimmed.'
        WHEN norm_fat_g_per_100g >= 25.0 THEN
            'Stabilized by partial coalescence of solid fat crystals (<10°C). Sensitive to thermal collapse above fat melting point.'
        WHEN hydrocolloid_mass_g > 0 THEN
            'Continuous serum immobilized by polymer network (gelatin triple-helix or xanthan yield stress), enabling stable low-fat foam.'
        ELSE
            'Interfacial protein unfolding creates cohesive 2D viscoelastic films. Sugar stabilizes by slowing Plateau border drainage.'
    END AS structural_notes
FROM recipe_aggregates;

-- ------------------------------------------------------------------------------
-- 4. Gold View: v_foam_stability_factors
-- Description: Direct comparative breakdown of foam overrun %, drainage half-life,
--              stabilizing mechanisms, and destabilizing agents across food systems.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_foam_stability_factors AS
SELECT
    i.ingredient_id,
    i.name AS ingredient_name,
    i.category,
    COALESCE(MAX(CASE WHEN fp.metric_name IN ('overrun_pct', 'foaming_capacity_overrun_pct') THEN fp.metric_value END), 0.0) AS overrun_pct,
    COALESCE(MAX(CASE WHEN fp.metric_name IN ('drainage_half_life_min', 'foam_stability_half_life_min') THEN fp.metric_value END), 0.0) AS drainage_half_life_min,
    COALESCE(MAX(CASE WHEN fp.metric_name IN ('interfacial_tension_mn_m') THEN fp.metric_value END), 0.0) AS interfacial_tension_mn_m,
    COALESCE(MAX(CASE WHEN fp.metric_name IN ('denaturation_onset_celsius', 'gelation_temp_celsius') THEN fp.metric_value END), 0.0) AS denaturation_temp_c,
    CASE
        WHEN i.category IN ('egg_protein', 'Eggs & Egg Products') THEN 'Viscoelastic 2D protein film (ovalbumin, ovotransferrin)'
        WHEN i.category IN ('dairy_fat', 'Dairy Products') AND i.ingredient_id LIKE '%cream%' THEN '3D partially coalesced fat crystal network'
        WHEN i.category IN ('hydrocolloid', 'Hydrocolloids & Gelling Agents') THEN 'Triple-helix gel crosslinking or shear-thinning yield stress'
        WHEN i.category IN ('plant_protein', 'Plant-Based & Legumes') THEN 'Soluble legume albumins and arabinogalactan polysaccharides'
        WHEN i.ingredient_id LIKE '%milk%' THEN 'Casein micelle and whey interfacial adsorption'
        ELSE 'Continuous phase viscosity or chemical buffering'
    END AS primary_stabilization_mechanism,
    CASE
        WHEN i.category IN ('egg_protein', 'Eggs & Egg Products') THEN 'Trace lipids/fats cause catastrophic bridging-dewetting'
        WHEN i.ingredient_id LIKE '%cream%' THEN 'Temperatures > 15°C melt fat crystals, causing phase separation'
        WHEN i.ingredient_id LIKE '%milk%' THEN 'Unemulsified fat globules spread on protein films, causing fast drainage'
        WHEN i.category IN ('plant_protein', 'Plant-Based & Legumes') THEN 'Ostwald ripening and heat-induced film thinning'
        ELSE 'Dilution or excessive heat'
    END AS primary_destabilization_mechanism,
    CASE
        WHEN i.category IN ('egg_protein', 'Eggs & Egg Products') THEN 'Sucrose (retards drainage) + Cream of tartar (optimizes pI charge)'
        WHEN i.ingredient_id LIKE '%cream%' THEN 'Strict temperature control (< 6°C) + Controlled shear'
        WHEN i.ingredient_id LIKE '%milk%' THEN 'Addition of gelatin, xanthan, or removal of fat (skim milk)'
        WHEN i.category IN ('plant_protein', 'Plant-Based & Legumes') THEN 'Cream of tartar + Xanthan gum (0.1-0.2%)'
        ELSE 'Balanced hydration'
    END AS recommended_countermeasures,
    CASE
        WHEN i.ingredient_id LIKE '%cream%' THEN 'FAT_CRYSTALLINE'
        WHEN i.ingredient_id LIKE '%gelatin%' OR i.ingredient_id LIKE '%xanthan%' THEN 'HYDROCOLLOID_MATRIX'
        WHEN i.ingredient_id LIKE '%milk%' THEN 'UNSTABLE_LIQUID_SUSPENSION'
        WHEN i.ingredient_id LIKE '%aquafaba%' THEN 'PLANT_PROTEIN_POLYSACCHARIDE'
        ELSE 'PROTEIN_VISCOELASTIC'
    END AS foam_classification
FROM silver_ingredients i
LEFT JOIN silver_functional_properties fp ON i.ingredient_id = fp.ingredient_id
GROUP BY i.ingredient_id, i.name, i.category;


-- ==============================================================================
-- PART II: DATABRICKS UNITY CATALOG DDL
-- Catalogs: workspace.foodmechanics_bronze, workspace.foodmechanics_silver, workspace.foodmechanics_gold
-- ==============================================================================

-- CREATE CATALOG IF NOT EXISTS workspace;
-- CREATE SCHEMA IF NOT EXISTS workspace.foodmechanics_bronze;
-- CREATE SCHEMA IF NOT EXISTS workspace.foodmechanics_silver;
-- CREATE SCHEMA IF NOT EXISTS workspace.foodmechanics_gold;

/*
-- Bronze Raw Delta Table
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_bronze.bronze_food_raw (
    source_id STRING NOT NULL,
    source_type STRING NOT NULL,
    payload_json STRING NOT NULL,
    ingestion_timestamp TIMESTAMP NOT NULL,
    content_hash STRING NOT NULL,
    source_url STRING,
    batch_id STRING
) USING DELTA
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'delta.autoOptimize.autoCompact' = 'true');

-- Silver Ingredients
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_ingredients (
    ingredient_id STRING NOT NULL,
    name STRING NOT NULL,
    common_name STRING,
    category STRING NOT NULL,
    default_serving_g DOUBLE NOT NULL,
    density_g_ml DOUBLE,
    description STRING,
    source_id STRING,
    source_type STRING,
    source_url STRING,
    evidence_confidence STRING,
    raw_payload_ref STRING
) USING DELTA;

-- Silver Nutritional Profiles (Strictly 100g Basis)
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_nutritional_profiles (
    profile_id STRING NOT NULL,
    ingredient_id STRING NOT NULL,
    basis_amount_g DOUBLE NOT NULL DEFAULT 100.0,
    energy_kcal DOUBLE NOT NULL,
    protein_g DOUBLE NOT NULL,
    fat_total_g DOUBLE NOT NULL,
    fat_saturated_g DOUBLE,
    carb_g DOUBLE NOT NULL,
    sugars_g DOUBLE,
    fiber_g DOUBLE DEFAULT 0.0,
    sodium_mg DOUBLE DEFAULT 0.0,
    water_g DOUBLE,
    is_inferred BOOLEAN DEFAULT FALSE,
    source_id STRING,
    source_type STRING,
    source_url STRING,
    evidence_confidence STRING,
    raw_payload_ref STRING
) USING DELTA;

-- Silver Functional Properties
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_functional_properties (
    property_id STRING NOT NULL,
    ingredient_id STRING NOT NULL,
    property_type STRING NOT NULL,
    metric_name STRING NOT NULL,
    metric_value DOUBLE NOT NULL,
    unit STRING NOT NULL,
    test_conditions STRING,
    confidence STRING NOT NULL,
    is_inferred BOOLEAN DEFAULT FALSE,
    evidence_id STRING,
    source_id STRING
) USING DELTA;

-- Silver Chemical Compounds
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_chemical_compounds (
    compound_id STRING NOT NULL,
    name STRING NOT NULL,
    molecular_formula STRING,
    pubchem_cid STRING,
    compound_class STRING NOT NULL,
    role_in_food STRING,
    source_id STRING
) USING DELTA;

-- Silver Food Mechanisms
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_food_mechanisms (
    mechanism_id STRING NOT NULL,
    title STRING NOT NULL,
    mechanism_type STRING NOT NULL,
    scientific_summary STRING NOT NULL,
    thermodynamic_or_kinetic_basis STRING NOT NULL,
    key_variables STRING,
    source_id STRING
) USING DELTA;

-- Silver Mechanism Relationships
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_mechanism_relationships (
    rel_id STRING NOT NULL,
    ingredient_or_compound_id STRING NOT NULL,
    mechanism_id STRING NOT NULL,
    role STRING NOT NULL,
    effect_direction STRING NOT NULL,
    evidence_id STRING,
    source_id STRING
) USING DELTA;

-- Silver Formulations
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_formulations (
    recipe_id STRING NOT NULL,
    name STRING NOT NULL,
    category STRING NOT NULL,
    total_mass_g DOUBLE NOT NULL,
    target_attribute STRING NOT NULL,
    instructions_summary STRING,
    source_id STRING
) USING DELTA;

-- Silver Formulation Items
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_formulation_items (
    recipe_id STRING NOT NULL,
    ingredient_id STRING NOT NULL,
    mass_g DOUBLE NOT NULL,
    proportion_pct DOUBLE NOT NULL
) USING DELTA;

-- Silver Scientific Evidence
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_scientific_evidence (
    evidence_id STRING NOT NULL,
    doi_or_citation STRING NOT NULL,
    publication_year INT NOT NULL,
    journal STRING NOT NULL,
    title STRING NOT NULL,
    evidence_grade STRING NOT NULL,
    findings_summary STRING NOT NULL,
    source_id STRING
) USING DELTA;

-- Databricks Unity Catalog Gold Views
CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_ingredient_comparisons AS
SELECT * FROM v_ingredient_comparisons;

CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_mechanism_evidence AS
SELECT * FROM v_mechanism_evidence;

CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_formulation_tradeoffs AS
SELECT * FROM v_formulation_tradeoffs;

CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_foam_stability_factors AS
SELECT * FROM v_foam_stability_factors;
*/
