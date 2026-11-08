-- ============================================================================
-- Databricks Genie Agent: FOODMECHANICS Intelligence
-- Verified Example SQL Assets for Unity Catalog
-- Target Catalogs & Schemas: workspace.foodmechanics_gold / workspace.foodmechanics_silver
-- ============================================================================

-- ============================================================================
-- QUERY 1: Standardized 100g Nutritional Comparison
-- Purpose: Compare nutritional composition of egg whites, whole milk, and heavy cream
--          strictly on a standardized 100g basis to eliminate serving volume bias.
-- ============================================================================
SELECT 
    i.ingredient_id,
    i.common_name,
    i.category,
    np.serving_basis,
    np.serving_size_g,
    np.calories_kcal,
    ROUND(np.water_g, 2) AS water_g,
    ROUND(np.protein_g, 2) AS protein_g,
    ROUND(np.fat_total_g, 2) AS fat_total_g,
    ROUND(np.fat_saturated_g, 2) AS fat_saturated_g,
    ROUND(np.carbohydrates_g, 2) AS carbs_g,
    ROUND(np.sugars_g, 2) AS sugars_g,
    ROUND(np.calcium_mg, 1) AS calcium_mg,
    ROUND(np.sodium_mg, 1) AS sodium_mg,
    ROUND((np.protein_g / NULLIF(np.protein_g + np.fat_total_g + np.carbohydrates_g, 0)) * 100, 1) AS protein_pct_of_macros,
    np.measurement_method,
    np.is_100g_normalized
FROM workspace.foodmechanics_silver.ingredients i
JOIN workspace.foodmechanics_silver.nutritional_profiles np 
    ON i.ingredient_id = np.ingredient_id
WHERE np.is_100g_normalized = TRUE
  AND i.ingredient_id IN ('ING_EGG_WHITE', 'ING_WHOLE_MILK', 'ING_HEAVY_CREAM')
ORDER BY np.protein_g DESC;


-- ============================================================================
-- QUERY 2: Molecular Drivers of Egg White Foaming Capacity
-- Purpose: Identify the specific proteins and physical properties in egg whites
--          that make them uniquely suited for foam expansion (overrun) and stability.
-- ============================================================================
SELECT 
    i.common_name AS ingredient_name,
    c.compound_name,
    c.compound_class,
    c.molecular_weight_kda,
    c.isoelectric_point_pi,
    c.denaturation_temp_c,
    c.hydrophobicity_index,
    c.functional_role AS molecular_function,
    fp.property_name,
    fp.quantitative_value,
    fp.unit_of_measure,
    fp.test_conditions
FROM workspace.foodmechanics_silver.ingredients i
JOIN workspace.foodmechanics_silver.functional_properties fp 
    ON i.ingredient_id = fp.ingredient_id
JOIN workspace.foodmechanics_silver.chemical_compounds c 
    ON fp.compound_id = c.compound_id
WHERE i.ingredient_id = 'ING_EGG_WHITE'
ORDER BY c.molecular_weight_kda ASC;


-- ============================================================================
-- QUERY 3: Fat-Stabilized (Cream) vs. Protein-Stabilized (Meringue) Mechanisms
-- Purpose: Contrast the colloidal physics of whipped cream vs egg white meringue,
--          explaining why fat stabilizes cream but catastrophically destabilizes egg white.
-- ============================================================================
SELECT 
    i.common_name AS ingredient,
    m.mechanism_name,
    m.mechanism_class,
    mr.relationship_type,
    c.compound_name AS active_compound,
    m.primary_stabilizing_agent,
    m.temperature_sensitivity,
    mr.impact_summary,
    m.evidence_confidence
FROM workspace.foodmechanics_silver.mechanism_relationships mr
JOIN workspace.foodmechanics_silver.ingredients i 
    ON mr.ingredient_id = i.ingredient_id
JOIN workspace.foodmechanics_silver.food_mechanisms m 
    ON mr.mechanism_id = m.mechanism_id
LEFT JOIN workspace.foodmechanics_silver.chemical_compounds c 
    ON mr.compound_id = c.compound_id
WHERE i.ingredient_id IN ('ING_EGG_WHITE', 'ING_HEAVY_CREAM')
ORDER BY i.ingredient_id, mr.relationship_type DESC;


-- ============================================================================
-- QUERY 4: Mechanisms and Stabilizers Retarding Foam Drainage (Syneresis)
-- Purpose: Identify food mechanisms and hydrocolloid stabilizers that prevent
--          drainage of liquid from foam lamellae, ranking empirical half-lives.
-- ============================================================================
SELECT 
    i.common_name AS stabilizer_name,
    i.category AS stabilizer_category,
    m.mechanism_name,
    m.structural_outcome,
    fp.property_name AS measured_property,
    fp.quantitative_value AS drainage_half_life_min,
    fp.unit_of_measure,
    fp.test_conditions,
    m.evidence_confidence
FROM workspace.foodmechanics_silver.ingredients i
JOIN workspace.foodmechanics_silver.mechanism_relationships mr 
    ON i.ingredient_id = mr.ingredient_id
JOIN workspace.foodmechanics_silver.food_mechanisms m 
    ON mr.mechanism_id = m.mechanism_id
JOIN workspace.foodmechanics_silver.functional_properties fp 
    ON i.ingredient_id = fp.ingredient_id
WHERE m.mechanism_id IN ('MECH_DRAINAGE_RETARDATION_VISCOSITY', 'MECH_STABILIZER_NETWORK_GELATION')
  AND fp.property_category = 'Foam Stability'
ORDER BY fp.quantitative_value DESC;


-- ============================================================================
-- QUERY 5: Substitution Trade-off: Egg White vs. Milk + Hydrocolloid (Xanthan Gum)
-- Purpose: Quantify the nutritional shift and functional gap when replacing egg whites
--          with a skim milk + xanthan gum combination in an aerated dessert.
-- ============================================================================
WITH ingredient_baseline AS (
    SELECT 
        i.ingredient_id,
        i.common_name,
        np.protein_g,
        np.fat_total_g,
        np.carbohydrates_g,
        np.calories_kcal,
        COALESCE(MAX(CASE WHEN fp.property_category = 'Foam Formation' THEN fp.quantitative_value END), 0) AS max_overrun_pct,
        COALESCE(MAX(CASE WHEN fp.property_category = 'Foam Stability' THEN fp.quantitative_value END), 0) AS drainage_half_life_min
    FROM workspace.foodmechanics_silver.ingredients i
    JOIN workspace.foodmechanics_silver.nutritional_profiles np 
        ON i.ingredient_id = np.ingredient_id AND np.is_100g_normalized = TRUE
    LEFT JOIN workspace.foodmechanics_silver.functional_properties fp 
        ON i.ingredient_id = fp.ingredient_id
    WHERE i.ingredient_id IN ('ING_EGG_WHITE', 'ING_SKIM_MILK', 'ING_XANTHAN_GUM')
    GROUP BY i.ingredient_id, i.common_name, np.protein_g, np.fat_total_g, np.carbohydrates_g, np.calories_kcal
)
SELECT 
    'Original (100g Egg White)' AS formulation_system,
    protein_g,
    fat_total_g,
    carbohydrates_g,
    calories_kcal,
    max_overrun_pct AS expected_overrun_pct,
    drainage_half_life_min AS stability_t_half_min,
    'Elastic protein viscoelastic film - heat coagulable at 65-84°C' AS structural_nature
FROM ingredient_baseline 
WHERE ingredient_id = 'ING_EGG_WHITE'

UNION ALL

SELECT 
    'Substitute (99.5g Skim Milk + 0.5g Xanthan Gum)' AS formulation_system,
    ROUND((m.protein_g * 0.995) + (x.protein_g * 0.005), 2) AS protein_g,
    ROUND((m.fat_total_g * 0.995) + (x.fat_total_g * 0.005), 2) AS fat_total_g,
    ROUND((m.carbohydrates_g * 0.995) + (x.carbohydrates_g * 0.005), 2) AS carbohydrates_g,
    ROUND((m.calories_kcal * 0.995) + (x.calories_kcal * 0.005), 1) AS calories_kcal,
    ROUND(m.max_overrun_pct * 0.65, 1) AS expected_overrun_pct,
    ROUND(x.drainage_half_life_min * 1.15, 1) AS stability_t_half_min,
    'Viscous steric drainage retardation - fragile interface, non-heat-setting' AS structural_nature
FROM (SELECT * FROM ingredient_baseline WHERE ingredient_id = 'ING_SKIM_MILK') m
CROSS JOIN (SELECT * FROM ingredient_baseline WHERE ingredient_id = 'ING_XANTHAN_GUM') x;


-- ============================================================================
-- QUERY 6: Protein Screening for High Overrun (> 400% Foaming Capacity)
-- Purpose: Screen ingredients that contain high-expansion proteins for food foaming.
-- ============================================================================
SELECT 
    i.ingredient_id,
    i.common_name,
    i.category,
    c.compound_name AS active_protein,
    c.compound_class,
    c.denaturation_temp_c,
    fp.property_name,
    fp.quantitative_value AS overrun_percent,
    fp.test_conditions,
    fp.ph_condition,
    fp.temperature_condition_c
FROM workspace.foodmechanics_silver.ingredients i
JOIN workspace.foodmechanics_silver.functional_properties fp 
    ON i.ingredient_id = fp.ingredient_id
JOIN workspace.foodmechanics_silver.chemical_compounds c 
    ON fp.compound_id = c.compound_id
WHERE fp.property_category = 'Foam Formation'
  AND fp.quantitative_value >= 400.0
ORDER BY fp.quantitative_value DESC;


-- ============================================================================
-- QUERY 7: Scientific Provenance Trace: Mechanisms to Literature and DOIs
-- Purpose: Ground physical food mechanisms into peer-reviewed research papers,
--          author citations, publication years, methodologies, and DOIs.
-- ============================================================================
SELECT 
    m.mechanism_name,
    m.mechanism_class,
    m.structural_outcome,
    se.title AS paper_title,
    se.authors,
    se.journal_or_book AS publication_venue,
    se.publication_year,
    se.doi_or_url,
    se.evidence_type,
    se.study_methodology,
    se.key_findings,
    se.evidence_confidence
FROM workspace.foodmechanics_silver.food_mechanisms m
JOIN workspace.foodmechanics_silver.scientific_evidence se 
    ON m.mechanism_id = se.mechanism_id
WHERE m.mechanism_id IN (
    'MECH_PROTEIN_DENATURATION_INTERFACE',
    'MECH_FAT_PARTIAL_COALESCENCE',
    'MECH_LIPID_INTERFERENCE_PROTEIN_FOAM'
)
ORDER BY m.mechanism_name, se.publication_year DESC;


-- ============================================================================
-- QUERY 8: Thermal Denaturation Thresholds & Heat-Induced Structural Transitions
-- Purpose: Compare thermal stability of major food biopolymers to determine
--          which proteins heat-set during baking versus which degrade or melt.
-- ============================================================================
SELECT 
    c.compound_name,
    c.compound_class,
    c.molecular_weight_kda,
    c.isoelectric_point_pi,
    c.denaturation_temp_c,
    CASE 
        WHEN c.denaturation_temp_c IS NULL THEN 'Non-protein / Non-denaturing'
        WHEN c.denaturation_temp_c <= 65.0 THEN 'Low thermal threshold (readily heat-coagulates)'
        WHEN c.denaturation_temp_c <= 85.0 THEN 'Moderate thermal threshold (baking gelation)'
        ELSE 'High thermal stability'
    END AS thermal_behavior_category,
    c.functional_role
FROM workspace.foodmechanics_silver.chemical_compounds c
WHERE c.compound_class LIKE '%Protein%'
ORDER BY c.denaturation_temp_c ASC;


-- ============================================================================
-- QUERY 9: Recipe-Level Formulation Nutritional Comparison (per 100g)
-- Purpose: Compute aggregate macronutrient density and calorie contribution
--          across classic meringue, traditional whipped cream, and low-fat foam recipes.
-- ============================================================================
SELECT 
    r.recipe_id,
    r.recipe_name,
    r.category,
    r.target_structure,
    ROUND(SUM(ri.quantity_g), 1) AS total_recipe_weight_g,
    ROUND(SUM(ri.quantity_g * (np.calories_kcal / 100.0)), 1) AS total_recipe_calories,
    ROUND(SUM(ri.quantity_g * (np.protein_g / 100.0)), 2) AS total_protein_g,
    ROUND(SUM(ri.quantity_g * (np.fat_total_g / 100.0)), 2) AS total_fat_g,
    ROUND(SUM(ri.quantity_g * (np.carbohydrates_g / 100.0)), 2) AS total_carbs_g,
    -- Standardized per 100g finished recipe:
    ROUND((SUM(ri.quantity_g * (np.calories_kcal / 100.0)) / SUM(ri.quantity_g)) * 100, 1) AS calories_per_100g,
    ROUND((SUM(ri.quantity_g * (np.protein_g / 100.0)) / SUM(ri.quantity_g)) * 100, 2) AS protein_per_100g,
    ROUND((SUM(ri.quantity_g * (np.fat_total_g / 100.0)) / SUM(ri.quantity_g)) * 100, 2) AS fat_per_100g,
    ROUND((SUM(ri.quantity_g * (np.carbohydrates_g / 100.0)) / SUM(ri.quantity_g)) * 100, 2) AS carbs_per_100g
FROM workspace.foodmechanics_silver.recipes r
JOIN workspace.foodmechanics_silver.recipe_ingredients ri 
    ON r.recipe_id = ri.recipe_id
JOIN workspace.foodmechanics_silver.nutritional_profiles np 
    ON ri.ingredient_id = np.ingredient_id AND np.is_100g_normalized = TRUE
GROUP BY r.recipe_id, r.recipe_name, r.category, r.target_structure
ORDER BY calories_per_100g DESC;


-- ============================================================================
-- QUERY 10: Evidence Certainty & Research Gap Identification
-- Purpose: Identify food mechanisms or relationships that lack peer-reviewed citations
--          or are tagged as unverified hypotheses (Data Rule 5 & 8 compliance).
-- ============================================================================
SELECT 
    m.mechanism_id,
    m.mechanism_name,
    m.mechanism_class,
    m.evidence_confidence AS mechanism_confidence,
    COUNT(se.evidence_id) AS peer_reviewed_citations_count,
    COUNT(CASE WHEN mr.is_hypothesized = TRUE THEN 1 END) AS hypothesized_interactions_count,
    CASE 
        WHEN COUNT(se.evidence_id) = 0 THEN 'RESEARCH GAP: No peer-reviewed publications linked'
        WHEN COUNT(se.evidence_id) < 2 THEN 'LIMITED EVIDENCE: Single publication cited'
        ELSE 'ROBUST: Supported by multiple peer-reviewed sources'
    END AS scientific_grounding_status
FROM workspace.foodmechanics_silver.food_mechanisms m
LEFT JOIN workspace.foodmechanics_silver.scientific_evidence se 
    ON m.mechanism_id = se.mechanism_id
LEFT JOIN workspace.foodmechanics_silver.mechanism_relationships mr 
    ON m.mechanism_id = mr.mechanism_id
GROUP BY m.mechanism_id, m.mechanism_name, m.mechanism_class, m.evidence_confidence
ORDER BY peer_reviewed_citations_count ASC, hypothesized_interactions_count DESC;


-- ============================================================================
-- QUERY 11: Experimental Processing Conditions vs. Overrun & Stability
-- Purpose: Analyze how shear rate, whipping temperature, and duration impact
--          mean bubble size and drainage half-life in food experiments.
-- ============================================================================
SELECT 
    e.experiment_id,
    e.experiment_name,
    i.common_name AS ingredient,
    pm.process_name,
    pm.temperature_range_c,
    pm.shear_rate,
    e.overrun_percent,
    e.drainage_half_life_min,
    e.bubble_size_mean_um,
    e.sensory_texture_score,
    e.is_successful,
    e.conclusion
FROM workspace.foodmechanics_silver.food_experiments e
JOIN workspace.foodmechanics_silver.ingredients i 
    ON e.ingredient_id = i.ingredient_id
JOIN workspace.foodmechanics_silver.processing_methods pm 
    ON e.process_id = pm.process_id
ORDER BY e.drainage_half_life_min DESC;


-- ============================================================================
-- QUERY 12: Analytical Gold View Exploration: v_ingredient_comparisons
-- Purpose: Direct query on the pre-computed comparison view for quick user exploration.
-- ============================================================================
SELECT 
    ingredient_name,
    category,
    calories_per_100g,
    protein_g_per_100g,
    fat_total_g_per_100g,
    carbs_g_per_100g,
    primary_functional_protein,
    foaming_overrun_pct,
    drainage_half_life_min,
    primary_stabilization_mechanism
FROM workspace.foodmechanics_gold.v_ingredient_comparisons
ORDER BY protein_g_per_100g DESC;
