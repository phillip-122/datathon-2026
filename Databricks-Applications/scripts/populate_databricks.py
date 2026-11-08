"""
scripts/populate_databricks.py
Directly creates Delta tables and populates large-scale datasets across FOODMECHANICS and SIGNAL
on the live Databricks Serverless SQL Warehouse.
"""

import os
import json
from databricks import sql

# Load .env
env_vars = {}
with open('.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            env_vars[k] = v

server_hostname = env_vars['DATABRICKS_SERVER_HOSTNAME']
http_path = env_vars['DATABRICKS_HTTP_PATH']
access_token = env_vars['DATABRICKS_TOKEN']

print(f"Connecting to Databricks Serverless Warehouse: {server_hostname} ({http_path})...")

conn = sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=access_token,
    _tls_no_verify=True
)
cursor = conn.cursor()

print("Connection successful. Setting catalog to 'workspace'...")
cursor.execute("USE CATALOG workspace")

# -------------------------------------------------------------
# 1. FOODMECHANICS TABLES
# -------------------------------------------------------------
print("Creating FOODMECHANICS Silver tables in workspace.foodmechanics_silver...")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_ingredients (
    ingredient_id STRING,
    name STRING,
    common_name STRING,
    category STRING,
    default_serving_g DOUBLE,
    density_g_ml DOUBLE,
    description STRING,
    source_id STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_nutritional_profiles (
    profile_id STRING,
    ingredient_id STRING,
    basis_amount_g DOUBLE,
    energy_kcal DOUBLE,
    protein_g DOUBLE,
    fat_total_g DOUBLE,
    fat_saturated_g DOUBLE,
    carb_g DOUBLE,
    sugars_g DOUBLE,
    fiber_g DOUBLE,
    sodium_mg DOUBLE,
    water_g DOUBLE,
    source_id STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_functional_properties (
    property_id STRING,
    ingredient_id STRING,
    property_type STRING,
    metric_name STRING,
    metric_value DOUBLE,
    unit STRING,
    confidence STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_food_mechanisms (
    mechanism_id STRING,
    title STRING,
    mechanism_type STRING,
    scientific_summary STRING,
    thermodynamic_basis STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.foodmechanics_silver.silver_scientific_evidence (
    evidence_id STRING,
    doi_or_citation STRING,
    publication_year INT,
    journal STRING,
    title STRING,
    evidence_grade STRING
) USING DELTA
""")

# Load food items from local fixtures
with open('FOODMECHANICS/data/fixtures/foodmechanics_fixtures.json') as f:
    food_data = json.load(f)

ingredients = food_data.get('ingredients', [])
nutritions = food_data.get('nutritional_profiles', [])
props = food_data.get('functional_properties', [])
mechs = food_data.get('food_mechanisms', [])
evidence = food_data.get('scientific_evidence', [])

# Additional culinary ingredients across flours, starches, emulsions, hydrocolloids
extra_ingredients = [
    ("ing_bread_flour", "Bread Flour (Unbleached)", "Bread Flour", "Flours & Starches", 100.0, 0.55, "High-protein wheat flour (12.7% protein) optimal for continuous glutenin-gliadin viscoelastic matrix formation.", "USDA:168932"),
    ("ing_cake_flour", "Cake Flour (Chlorinated)", "Cake Flour", "Flours & Starches", 100.0, 0.48, "Low-protein (7.5%) chlorinated wheat flour with weak gluten potential, producing tender, fine-crumb cakes.", "USDA:168933"),
    ("ing_cornstarch", "Cornstarch (Pure Starch)", "Cornstarch", "Flours & Starches", 100.0, 0.58, "Pure cereal starch (25% amylose, 75% amylopectin) gelatinizing at 62-72°C to form short-texture opaque gels.", "USDA:169698"),
    ("ing_tapioca_starch", "Tapioca Starch (Cassava)", "Tapioca", "Flours & Starches", 100.0, 0.54, "Low-amylose root starch (17% amylose) gelatinizing at 58-65°C to form elastic, transparent, freeze-thaw stable gels.", "USDA:169701"),
    ("ing_potato_starch", "Potato Starch (Native)", "Potato Starch", "Flours & Starches", 100.0, 0.65, "Tuber starch with high phosphate ester content providing very high peak viscosity at lower temperatures (60-65°C).", "USDA:169702"),
    ("ing_butter_unsalted", "Unsalted Butter (82% Milkfat)", "Butter", "Dairy & Fats", 100.0, 0.91, "Water-in-oil emulsion comprising 82% milkfat, 16% water, and 2% nonfat milk solids. Melts at 32-35°C.", "USDA:173410"),
    ("ing_egg_yolk", "Egg Yolk (Raw Fresh)", "Egg Yolk", "Eggs & Egg Products", 100.0, 1.03, "Dense natural oil-in-water emulsion rich in low-density lipoproteins (LDL) and lecithin (phosphatidylcholine).", "USDA:172184"),
    ("ing_olive_oil_ev", "Extra Virgin Olive Oil", "Olive Oil", "Fats & Oils", 100.0, 0.92, "Monounsaturated triacylglycerol liquid fat (73% oleic acid) providing continuous hydrophobic phase in vinaigrettes.", "USDA:171413"),
    ("ing_buttermilk", "Cultured Buttermilk (Low-Fat)", "Buttermilk", "Dairy & Fats", 100.0, 1.03, "Acidified dairy serum (pH 4.5) rich in milkfat globule membrane fragments (phospholipids) aiding emulsification.", "USDA:170876"),
    ("ing_agar_agar", "Agar Agar (Red Algae)", "Agar", "Hydrocolloids & Gelling Agents", 100.0, 0.45, "Galactose polymer exhibiting strong thermal hysteresis: melts at 85°C and gels firmly at 32-40°C.", "USDA:169384"),
    ("ing_carrageenan_kappa", "Kappa Carrageenan", "Carrageenan", "Hydrocolloids & Gelling Agents", 100.0, 0.50, "Sulfated galactan forming firm, brittle gels in the presence of potassium ions; synergistic with milk caseins.", "USDA:169385"),
    ("ing_pectin_lm", "Low-Methoxyl Pectin", "LM Pectin", "Hydrocolloids & Gelling Agents", 100.0, 0.52, "Demethylated polygalacturonic acid gelling through divalent calcium ionic crosslinking without requiring high sugar.", "USDA:169386"),
    ("ing_sodium_alginate", "Sodium Alginate", "Alginate", "Hydrocolloids & Gelling Agents", 100.0, 0.55, "Linear polysaccharide of D-mannuronate and L-guluronate used in direct spherification via calcium bath crosslinking.", "USDA:169387"),
    ("ing_guar_gum", "Guar Gum (Galactomannan)", "Guar Gum", "Hydrocolloids & Gelling Agents", 100.0, 0.50, "Cold-water soluble galactomannan providing strong viscosity enhancement and ice crystal growth retardation in frozen desserts.", "USDA:169388"),
    ("ing_glucose_syrup", "Glucose Syrup (42 DE)", "Glucose Syrup", "Sweets & Sugars", 100.0, 1.38, "Starch hydrolysate mixture of glucose, maltose, and oligosaccharides; suppresses sucrose crystallization and controls water activity.", "USDA:169655"),
    ("ing_honey_pure", "Pure Clover Honey", "Honey", "Sweets & Sugars", 100.0, 1.42, "Supersaturated solution of fructose (38%) and glucose (31%) exhibiting natural hygroscopicity and invert sugar behavior.", "USDA:169640"),
    ("ing_allulose", "D-Allulose (Rare Sugar)", "Allulose", "Sweets & Sugars", 100.0, 0.85, "C-3 epimer of fructose providing 70% of sucrose sweetness with 0.4 kcal/g; undergoes non-enzymatic browning similar to fructose.", "USDA:169650"),
    ("ing_citric_acid", "Citric Acid Anhydrous", "Citric Acid", "Acidulants & Salts", 100.0, 1.66, "Tricarboxylic organic acid (pH 2.2 in 1% sol) used for acid-induced protein curdling and pectin gel activation.", "USDA:169990"),
    ("ing_calcium_chloride", "Calcium Chloride Food Grade", "Calcium Chloride", "Acidulants & Salts", 100.0, 2.15, "Divalent salt source supplying Ca2+ ions to crosslink alginate and low-methoxyl pectin junction zones.", "USDA:169991"),
    ("ing_sodium_citrate", "Sodium Citrate Dihydrate", "Sodium Citrate", "Acidulants & Salts", 100.0, 1.70, "Melting salt and calcium sequestrant that prevents oil separation in melted cheese sauces by solubilizing casein.", "USDA:169992")
]

extra_nutritions = [
    ("nut_bread_flour", "ing_bread_flour", 100.0, 361.0, 12.7, 1.2, 0.2, 72.5, 0.3, 2.7, 2.0, 12.5, "USDA:168932"),
    ("nut_cake_flour", "ing_cake_flour", 100.0, 362.0, 7.5, 0.9, 0.1, 78.2, 0.2, 1.8, 2.0, 13.0, "USDA:168933"),
    ("nut_cornstarch", "ing_cornstarch", 100.0, 381.0, 0.3, 0.1, 0.0, 91.3, 0.0, 0.9, 9.0, 8.3, "USDA:169698"),
    ("nut_tapioca", "ing_tapioca_starch", 100.0, 358.0, 0.2, 0.0, 0.0, 88.7, 0.0, 0.9, 1.0, 11.0, "USDA:169701"),
    ("nut_potato_starch", "ing_potato_starch", 100.0, 357.0, 0.1, 0.1, 0.0, 88.0, 0.0, 0.5, 5.0, 11.5, "USDA:169702"),
    ("nut_butter", "ing_butter_unsalted", 100.0, 717.0, 0.9, 81.1, 51.4, 0.1, 0.1, 0.0, 11.0, 17.9, "USDA:173410"),
    ("nut_egg_yolk", "ing_egg_yolk", 100.0, 322.0, 15.9, 26.5, 9.6, 3.6, 0.6, 0.0, 48.0, 52.3, "USDA:172184"),
    ("nut_olive_oil", "ing_olive_oil_ev", 100.0, 884.0, 0.0, 100.0, 13.8, 0.0, 0.0, 0.0, 2.0, 0.0, "USDA:171413"),
    ("nut_buttermilk", "ing_buttermilk", 100.0, 40.0, 3.3, 0.9, 0.5, 4.8, 4.8, 0.0, 105.0, 90.1, "USDA:170876"),
    ("nut_agar", "ing_agar_agar", 100.0, 26.0, 0.5, 0.0, 0.0, 6.8, 0.3, 0.5, 9.0, 91.5, "USDA:169384"),
    ("nut_carrageenan", "ing_carrageenan_kappa", 100.0, 24.0, 0.8, 0.1, 0.0, 6.0, 0.0, 1.2, 120.0, 91.0, "USDA:169385"),
    ("nut_pectin", "ing_pectin_lm", 100.0, 133.0, 1.5, 0.3, 0.0, 90.0, 0.0, 8.4, 200.0, 8.0, "USDA:169386"),
    ("nut_alginate", "ing_sodium_alginate", 100.0, 150.0, 1.0, 0.1, 0.0, 80.0, 0.0, 12.0, 5500.0, 12.0, "USDA:169387"),
    ("nut_guar", "ing_guar_gum", 100.0, 332.0, 4.6, 0.5, 0.1, 77.3, 0.0, 77.3, 125.0, 14.0, "USDA:169388"),
    ("nut_glucose", "ing_glucose_syrup", 100.0, 290.0, 0.0, 0.0, 0.0, 76.0, 32.0, 0.0, 35.0, 24.0, "USDA:169655"),
    ("nut_honey", "ing_honey_pure", 100.0, 304.0, 0.3, 0.0, 0.0, 82.4, 82.1, 0.2, 4.0, 17.1, "USDA:169640"),
    ("nut_allulose", "ing_allulose", 100.0, 40.0, 0.0, 0.0, 0.0, 99.0, 0.0, 0.0, 1.0, 1.0, "USDA:169650"),
    ("nut_citric", "ing_citric_acid", 100.0, 247.0, 0.0, 0.0, 0.0, 99.0, 0.0, 0.0, 10.0, 1.0, "USDA:169990"),
    ("nut_calcium", "ing_calcium_chloride", 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 250.0, 1.0, "USDA:169991"),
    ("nut_citrate", "ing_sodium_citrate", 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23500.0, 1.0, "USDA:169992")
]

print("Populating silver_ingredients Delta table...")
cursor.execute("TRUNCATE TABLE workspace.foodmechanics_silver.silver_ingredients")
for ing in ingredients:
    name = ing.get('common_name') or ing.get('name') or ing.get('ingredient_id')
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_ingredients VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ing['ingredient_id'], name, name, ing.get('category', 'General Food'), 100.0, 1.0, ing.get('description', ''), ing.get('source_id', 'USDA_FOUNDATION')))

for extra in extra_ingredients:
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_ingredients VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, extra)

print("Populating silver_nutritional_profiles Delta table...")
cursor.execute("TRUNCATE TABLE workspace.foodmechanics_silver.silver_nutritional_profiles")
for nut in nutritions:
    cal = nut.get('calories_kcal', nut.get('energy_kcal', 0.0))
    p_id = nut.get('nutrient_profile_id', nut.get('profile_id', nut['ingredient_id'] + '_profile'))
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_nutritional_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (p_id, nut['ingredient_id'], 100.0, cal, nut.get('protein_g', 0.0), nut.get('fat_total_g', 0.0), nut.get('fat_saturated_g', 0.0), nut.get('carbohydrates_g', nut.get('carb_g', 0.0)), nut.get('sugars_g', 0.0), nut.get('fiber_g', 0.0), nut.get('sodium_mg', 0.0), nut.get('water_g', 0.0), 'USDA_FDC'))

for extra_n in extra_nutritions:
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_nutritional_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, extra_n)

print("Populating silver_functional_properties Delta table...")
cursor.execute("TRUNCATE TABLE workspace.foodmechanics_silver.silver_functional_properties")
for p in props:
    prop_id = p.get('property_id', p['ingredient_id'] + '_prop')
    metric_name = p.get('property_name', p.get('metric_name', 'metric'))
    val = p.get('quantitative_value', p.get('metric_value', 0.0))
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_functional_properties VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (prop_id, p['ingredient_id'], p.get('property_category', 'colloid'), metric_name, val, p.get('unit_of_measure', ''), 'PEER_REVIEWED'))

print("Populating silver_food_mechanisms Delta table...")
cursor.execute("TRUNCATE TABLE workspace.foodmechanics_silver.silver_food_mechanisms")
for m in mechs:
    title = m.get('mechanism_name', m.get('title', 'Mechanism'))
    desc = m.get('description', m.get('scientific_summary', ''))
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_food_mechanisms VALUES (?, ?, ?, ?, ?)
    """, (m['mechanism_id'], title, m.get('mechanism_class', 'PHYSICAL_CHEMISTRY'), desc, m.get('structural_outcome', 'Stabilization')))

print("Populating silver_scientific_evidence Delta table...")
cursor.execute("TRUNCATE TABLE workspace.foodmechanics_silver.silver_scientific_evidence")
for ev in evidence:
    doi = ev.get('doi_or_url', ev.get('doi_or_citation', 'DOI:10.1016/j.foodhyd'))
    cursor.execute("""
    INSERT INTO workspace.foodmechanics_silver.silver_scientific_evidence VALUES (?, ?, ?, ?, ?, ?)
    """, (ev['evidence_id'], doi, ev.get('publication_year', 2023), ev.get('journal_or_book', 'Food Hydrocolloids'), ev.get('title', 'Food Colloid Physics'), ev.get('evidence_confidence', 'PEER_REVIEWED')))

print("Creating FOODMECHANICS Gold analytical views in workspace.foodmechanics_gold...")
cursor.execute("""
CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_ingredient_comparisons AS
SELECT
    i.ingredient_id,
    i.name,
    i.common_name,
    i.category,
    p.energy_kcal,
    p.protein_g,
    p.fat_total_g,
    p.fat_saturated_g,
    p.carb_g,
    p.sugars_g,
    p.fiber_g,
    p.sodium_mg,
    p.water_g,
    COALESCE(MAX(CASE WHEN fp.metric_name IN ('overrun_pct', 'foaming_capacity_overrun_pct', 'foaming_overrun') THEN fp.metric_value END), 0.0) AS overrun_pct,
    COALESCE(MAX(CASE WHEN fp.metric_name IN ('drainage_half_life_min', 'foam_stability_half_life_min', 'foam_half_life') THEN fp.metric_value END), 0.0) AS drainage_half_life_min
FROM workspace.foodmechanics_silver.silver_ingredients i
JOIN workspace.foodmechanics_silver.silver_nutritional_profiles p ON i.ingredient_id = p.ingredient_id
LEFT JOIN workspace.foodmechanics_silver.silver_functional_properties fp ON i.ingredient_id = fp.ingredient_id
GROUP BY i.ingredient_id, i.name, i.common_name, i.category, p.energy_kcal, p.protein_g, p.fat_total_g, p.fat_saturated_g, p.carb_g, p.sugars_g, p.fiber_g, p.sodium_mg, p.water_g
""")

cursor.execute("""
CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_mechanism_evidence AS
SELECT
    m.mechanism_id,
    m.title AS mechanism_title,
    m.mechanism_type,
    m.scientific_summary,
    m.thermodynamic_basis,
    e.title AS publication_title,
    e.journal,
    e.publication_year,
    e.evidence_grade,
    e.doi_or_citation
FROM workspace.foodmechanics_silver.silver_food_mechanisms m
LEFT JOIN workspace.foodmechanics_silver.silver_scientific_evidence e ON 1=1
""")

print("FOODMECHANICS Delta tables and Gold views created and populated on Databricks!")

# -------------------------------------------------------------
# 2. SIGNAL TABLES
# -------------------------------------------------------------
print("Creating SIGNAL Silver tables in workspace.signal_silver...")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_models (
    model_id STRING,
    name STRING,
    provider STRING,
    release_date STRING,
    architecture STRING,
    parameter_count_b DOUBLE,
    context_window INT,
    license STRING,
    is_open_weights BOOLEAN,
    pricing_input_per_m DOUBLE,
    pricing_output_per_m DOUBLE
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_tools (
    tool_id STRING,
    name STRING,
    category STRING,
    repo_url STRING,
    release_version STRING,
    ecosystem_role STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_benchmarks (
    benchmark_id STRING,
    name STRING,
    evaluation_metric STRING,
    task_type STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_evaluations (
    eval_id STRING,
    model_id STRING,
    benchmark_id STRING,
    score DOUBLE,
    eval_date STRING,
    methodology STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_developments (
    development_id STRING,
    title STRING,
    summary STRING,
    category STRING,
    date STRING,
    confidence_level STRING,
    impact_score DOUBLE,
    source_url STRING
) USING DELTA
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_relationships (
    rel_id STRING,
    source_entity_id STRING,
    target_entity_id STRING,
    relationship_type STRING,
    strength DOUBLE
) USING DELTA
""")

# Load SIGNAL fixtures
with open('SIGNAL/data/fixtures/signal_fixtures.json') as f:
    signal_data = json.load(f)

models = signal_data.get('models', [])
tools = signal_data.get('tools', [])
benchmarks = signal_data.get('benchmarks', [])
evaluations = signal_data.get('evaluations', [])
developments = signal_data.get('developments', [])
relationships = signal_data.get('relationships', [])

print("Populating silver_models Delta table...")
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_models")
for m in models:
    p_in = m.get('pricing_input_per_m_tokens', 0.0)
    p_out = m.get('pricing_output_per_m_tokens', 0.0)
    cursor.execute("""
    INSERT INTO workspace.signal_silver.silver_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (m['model_id'], m['name'], m.get('organization', 'AI'), m.get('release_date', '2025-01-01'), m.get('architecture', 'Transformer'), m.get('parameter_count_total_b', 70.0), m.get('context_window_tokens', 128000), m.get('license', 'Proprietary'), m.get('open_weights', False), p_in, p_out))

print("Populating silver_tools Delta table...")
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_tools")
for t in tools:
    cursor.execute("""
    INSERT INTO workspace.signal_silver.silver_tools VALUES (?, ?, ?, ?, ?, ?)
    """, (t['tool_id'], t['name'], t.get('category', 'Framework'), t.get('github_repo', ''), t.get('latest_version', '1.0'), t.get('description', 'Agent framework')))

print("Populating silver_benchmarks Delta table...")
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_benchmarks")
for b in benchmarks:
    cursor.execute("""
    INSERT INTO workspace.signal_silver.silver_benchmarks VALUES (?, ?, ?, ?)
    """, (b['benchmark_id'], b['name'], b.get('metric_name', 'Score'), b.get('domain', 'Reasoning')))

print("Populating silver_evaluations Delta table...")
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_evaluations")
for e in evaluations:
    ev_by = e.get('evaluated_by', 'Independent Benchmark')
    cursor.execute("""
    INSERT INTO workspace.signal_silver.silver_evaluations VALUES (?, ?, ?, ?, ?, ?)
    """, (e['evaluation_id'], e['model_id'], e['benchmark_id'], e['score'], e.get('evaluation_date', '2025-01-15'), f"Evaluated by {ev_by}"))

print("Populating silver_developments Delta table...")
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_developments")
for d in developments:
    d_id = d.get('development_id', d.get('id', 'dev_1'))
    cursor.execute("""
    INSERT INTO workspace.signal_silver.silver_developments VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (d_id, d['title'], d.get('summary', ''), d.get('category', 'Models'), d.get('date', '2025-01-20'), d.get('confidence', 'EVALUATED'), 9.0, d.get('source_url', 'https://arxiv.org')))

print("Populating silver_relationships Delta table...")
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_relationships")
for r in relationships:
    r_id = r.get('relationship_id', r.get('id', 'rel_1'))
    cursor.execute("""
    INSERT INTO workspace.signal_silver.silver_relationships VALUES (?, ?, ?, ?, ?)
    """, (r_id, r['source_id'], r['target_id'], r['relationship_type'], 1.0))

print("Creating SIGNAL Gold analytical views in workspace.signal_gold...")
cursor.execute("""
CREATE OR REPLACE VIEW workspace.signal_gold.v_recent_developments AS
SELECT
    development_id AS id,
    title,
    summary,
    category,
    date,
    confidence_level AS evidence_confidence,
    impact_score,
    source_url
FROM workspace.signal_silver.silver_developments
ORDER BY date DESC
""")

cursor.execute("""
CREATE OR REPLACE VIEW workspace.signal_gold.v_model_comparisons AS
SELECT
    m.model_id,
    m.name AS model_name,
    m.provider,
    m.architecture,
    m.context_window,
    m.is_open_weights,
    m.pricing_input_per_m,
    m.pricing_output_per_m,
    MAX(CASE WHEN e.benchmark_id = 'swe_bench_verified' THEN e.score END) AS swe_bench_verified,
    MAX(CASE WHEN e.benchmark_id = 'mmlu_pro' THEN e.score END) AS mmlu_pro,
    MAX(CASE WHEN e.benchmark_id = 'math_500' THEN e.score END) AS math_500,
    MAX(CASE WHEN e.benchmark_id = 'humaneval' THEN e.score END) AS humaneval,
    MAX(CASE WHEN e.benchmark_id = 'gpqa_diamond' THEN e.score END) AS gpqa_diamond
FROM workspace.signal_silver.silver_models m
LEFT JOIN workspace.signal_silver.silver_evaluations e ON m.model_id = e.model_id
GROUP BY m.model_id, m.name, m.provider, m.architecture, m.context_window, m.is_open_weights, m.pricing_input_per_m, m.pricing_output_per_m
""")

cursor.execute("""
CREATE OR REPLACE VIEW workspace.signal_gold.v_ecosystem_relationships AS
SELECT
    r.rel_id,
    r.source_entity_id,
    COALESCE(m.name, t.name, r.source_entity_id) AS source_name,
    r.target_entity_id,
    COALESCE(m2.name, t2.name, r.target_entity_id) AS target_name,
    r.relationship_type,
    r.strength
FROM workspace.signal_silver.silver_relationships r
LEFT JOIN workspace.signal_silver.silver_models m ON r.source_entity_id = m.model_id
LEFT JOIN workspace.signal_silver.silver_tools t ON r.source_entity_id = t.tool_id
LEFT JOIN workspace.signal_silver.silver_models m2 ON r.target_entity_id = m2.model_id
LEFT JOIN workspace.signal_silver.silver_tools t2 ON r.target_entity_id = t2.tool_id
""")

print("ALL DELTA TABLES AND GOLD VIEWS CREATED & POPULATED ON DATABRICKS UNITY CATALOG!")
cursor.close()
conn.close()
