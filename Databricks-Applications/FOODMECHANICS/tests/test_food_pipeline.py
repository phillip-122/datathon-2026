"""Comprehensive automated tests for FOODMECHANICS Data Engineering Pipeline.

Verifies:
1. Unit normalization strictly to 100g basis.
2. Non-null guarantees across all Bronze and Silver tables.
3. Formulation mass balance and percentage proportions.
4. Bronze hash deduplication and idempotent ingestion.
5. Mechanism relationship graph referential integrity.
6. Gold analytical views accuracy and stability indices.
7. Ingredient substitution simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from FOODMECHANICS.pipelines.bronze_pipeline import (
    compute_content_hash,
    init_bronze_tables,
    ingest_fixture_files,
    ingest_raw_record,
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


@pytest.fixture
def db_conn():
    """In-memory DuckDB connection populated through full Bronze, Silver, Gold pipelines."""
    con = duckdb.connect(":memory:")
    ingest_fixture_files(con)
    run_silver_pipeline(con)
    create_gold_views(con)
    yield con
    con.close()


# ==============================================================================
# TEST SUITE 1: 100g NUTRITIONAL NORMALIZATION CONTRACT
# ==============================================================================


def test_normalization_arbitrary_serving_to_100g():
    """Verify exact linear scaling from arbitrary serving size to strictly 100g basis."""
    # 33g egg white serving with 3.6g protein, 0.056g fat, 17.16 kcal
    raw_payload = {
        "energy_kcal": 17.16,
        "protein_g": 3.60,
        "fat_total_g": 0.056,
        "fat_saturated_g": 0.0,
        "carb_g": 0.241,
        "sugars_g": 0.234,
        "sodium_mg": 54.78,
    }
    norm = normalize_nutrition_100g(raw_payload, serving_size_g=33.0, precision=4)

    expected_scale = 100.0 / 33.0
    assert norm["protein_g"] == round(3.60 * expected_scale, 4)
    assert norm["energy_kcal"] == round(17.16 * expected_scale, 4)
    assert norm["fat_total_g"] == round(0.056 * expected_scale, 4)
    assert pytest.approx(norm["protein_g"], 0.01) == 10.91


def test_normalization_identity_at_100g():
    """Verify that a serving size of 100g produces identical values without distortion."""
    raw_payload = {
        "energy_kcal": 52.0,
        "protein_g": 10.9,
        "fat_total_g": 0.17,
        "carb_g": 0.73,
    }
    norm = normalize_nutrition_100g(raw_payload, serving_size_g=100.0)
    assert norm["energy_kcal"] == 52.0
    assert norm["protein_g"] == 10.9
    assert norm["fat_total_g"] == 0.17
    assert norm["carb_g"] == 0.73


def test_normalization_invalid_serving_size():
    """Verify that zero or negative serving size raises a ValueError."""
    with pytest.raises(ValueError):
        normalize_nutrition_100g({"protein_g": 10.0}, serving_size_g=0.0)

    with pytest.raises(ValueError):
        normalize_nutrition_100g({"protein_g": 10.0}, serving_size_g=-50.0)


def test_all_silver_nutritional_profiles_enforce_100g_basis(db_conn):
    """Verify that 100% of rows in silver_nutritional_profiles have basis_amount_g == 100.0."""
    rows = db_conn.execute("SELECT ingredient_id, basis_amount_g FROM silver_nutritional_profiles").fetchall()
    assert len(rows) > 0, "No nutritional profiles found in Silver."
    for ing_id, basis_g in rows:
        assert basis_g == 100.0, f"Ingredient {ing_id} violated 100g contract: {basis_g}g"


# ==============================================================================
# TEST SUITE 2: NON-NULL GUARANTEES
# ==============================================================================


def test_silver_ingredients_non_null(db_conn):
    """Verify non-null constraints on silver_ingredients."""
    bad_rows = db_conn.execute(
        """
        SELECT COUNT(*) FROM silver_ingredients
        WHERE ingredient_id IS NULL OR name IS NULL OR category IS NULL OR default_serving_g IS NULL
        """
    ).fetchone()[0]
    assert bad_rows == 0, f"Found {bad_rows} silver_ingredients with NULL in mandatory columns."


def test_silver_nutritional_profiles_non_null(db_conn):
    """Verify non-null constraints on silver_nutritional_profiles."""
    bad_rows = db_conn.execute(
        """
        SELECT COUNT(*) FROM silver_nutritional_profiles
        WHERE profile_id IS NULL OR ingredient_id IS NULL OR basis_amount_g IS NULL
           OR energy_kcal IS NULL OR protein_g IS NULL OR fat_total_g IS NULL OR carb_g IS NULL
        """
    ).fetchone()[0]
    assert bad_rows == 0, f"Found {bad_rows} silver_nutritional_profiles with NULL in mandatory columns."


def test_silver_functional_properties_non_null(db_conn):
    """Verify non-null constraints on silver_functional_properties."""
    bad_rows = db_conn.execute(
        """
        SELECT COUNT(*) FROM silver_functional_properties
        WHERE property_id IS NULL OR ingredient_id IS NULL OR property_type IS NULL
           OR metric_name IS NULL OR metric_value IS NULL OR unit IS NULL OR confidence IS NULL
        """
    ).fetchone()[0]
    assert bad_rows == 0, f"Found {bad_rows} silver_functional_properties with NULL in mandatory columns."


def test_silver_chemical_compounds_non_null(db_conn):
    """Verify non-null constraints on silver_chemical_compounds."""
    bad_rows = db_conn.execute(
        """
        SELECT COUNT(*) FROM silver_chemical_compounds
        WHERE compound_id IS NULL OR name IS NULL OR compound_class IS NULL
        """
    ).fetchone()[0]
    assert bad_rows == 0, f"Found {bad_rows} silver_chemical_compounds with NULL in mandatory columns."


def test_silver_food_mechanisms_non_null(db_conn):
    """Verify non-null constraints on silver_food_mechanisms."""
    bad_rows = db_conn.execute(
        """
        SELECT COUNT(*) FROM silver_food_mechanisms
        WHERE mechanism_id IS NULL OR title IS NULL OR mechanism_type IS NULL
           OR scientific_summary IS NULL OR thermodynamic_or_kinetic_basis IS NULL
        """
    ).fetchone()[0]
    assert bad_rows == 0, f"Found {bad_rows} silver_food_mechanisms with NULL in mandatory columns."


def test_silver_scientific_evidence_non_null(db_conn):
    """Verify non-null constraints on silver_scientific_evidence."""
    bad_rows = db_conn.execute(
        """
        SELECT COUNT(*) FROM silver_scientific_evidence
        WHERE evidence_id IS NULL OR doi_or_citation IS NULL OR publication_year IS NULL
           OR journal IS NULL OR title IS NULL OR evidence_grade IS NULL OR findings_summary IS NULL
        """
    ).fetchone()[0]
    assert bad_rows == 0, f"Found {bad_rows} silver_scientific_evidence with NULL in mandatory columns."


# ==============================================================================
# TEST SUITE 3: FORMULATION MASS BALANCE & PROPORTIONS
# ==============================================================================


def test_formulation_mass_balance_and_proportions(db_conn):
    """
    Verify that for every formulation:
    1. Sum of item masses matches total_mass_g within tolerance.
    2. Sum of proportion_pct equals 100.0% within tolerance.
    3. Individual proportion_pct matches mass_g / total_mass_g * 100.0.
    """
    recipes = db_conn.execute(
        "SELECT recipe_id, name, total_mass_g FROM silver_formulations"
    ).fetchall()

    assert len(recipes) > 0, "No formulations found."

    for recipe_id, name, total_mass_g in recipes:
        items = db_conn.execute(
            """
            SELECT ingredient_id, mass_g, proportion_pct
            FROM silver_formulation_items
            WHERE recipe_id = ?
            """,
            [recipe_id],
        ).fetchall()

        assert len(items) > 0, f"Recipe {recipe_id} ({name}) has no items."

        sum_mass = sum(it[1] for it in items)
        sum_pct = sum(it[2] for it in items)

        # 1. Mass balance check (within 0.05g)
        assert abs(sum_mass - total_mass_g) < 0.05, (
            f"Mass imbalance in {recipe_id}: items sum to {sum_mass}g, expected {total_mass_g}g"
        )

        # 2. Percentage proportion check (within 0.05%)
        assert abs(sum_pct - 100.0) < 0.05, (
            f"Proportion percentage does not sum to 100% in {recipe_id}: sum is {sum_pct}%"
        )

        # 3. Item-level math check
        for ing_id, mass_g, prop_pct in items:
            expected_pct = round((mass_g / total_mass_g) * 100.0, 4)
            assert abs(prop_pct - expected_pct) < 0.01, (
                f"Item {ing_id} in {recipe_id} proportion math mismatch: got {prop_pct}%, expected {expected_pct}%"
            )


# ==============================================================================
# TEST SUITE 4: BRONZE HASH DEDUPLICATION & IDEMPOTENCY
# ==============================================================================


def test_bronze_hash_deduplication():
    """Verify that re-ingesting identical payloads is skipped, while changes trigger update."""
    con = duckdb.connect(":memory:")
    init_bronze_tables(con)

    source_id = "test:egg_001"
    payload_v1 = {"name": "Egg White", "protein_g": 10.9}

    # First ingestion -> INSERTED
    s_id, status1 = ingest_raw_record(con, source_id, "test_source", payload_v1)
    assert status1 == "INSERTED"

    # Second identical ingestion -> SKIPPED
    _, status2 = ingest_raw_record(con, source_id, "test_source", payload_v1)
    assert status2 == "SKIPPED"

    # Verify only 1 record exists
    cnt = con.execute("SELECT COUNT(*) FROM bronze_food_raw WHERE source_id = ?", [source_id]).fetchone()[0]
    assert cnt == 1

    # Updated payload -> UPDATED
    payload_v2 = {"name": "Egg White", "protein_g": 11.2}
    _, status3 = ingest_raw_record(con, source_id, "test_source", payload_v2)
    assert status3 == "UPDATED"

    # Row count remains 1
    cnt2 = con.execute("SELECT COUNT(*) FROM bronze_food_raw WHERE source_id = ?", [source_id]).fetchone()[0]
    assert cnt2 == 1

    con.close()


# ==============================================================================
# TEST SUITE 5: MECHANISM RELATIONSHIP GRAPH INTEGRITY
# ==============================================================================


def test_mechanism_graph_integrity(db_conn):
    """Verify referential integrity of mechanism relationships and evidence."""
    # Check that all mechanisms in relationships exist
    orphan_mechs = db_conn.execute(
        """
        SELECT r.rel_id, r.mechanism_id
        FROM silver_mechanism_relationships r
        LEFT JOIN silver_food_mechanisms m ON r.mechanism_id = m.mechanism_id
        WHERE m.mechanism_id IS NULL
        """
    ).fetchall()
    assert len(orphan_mechs) == 0, f"Found orphan mechanism relationships: {orphan_mechs}"

    # Check that when evidence_id is present, it exists in scientific evidence
    orphan_evidence = db_conn.execute(
        """
        SELECT r.rel_id, r.evidence_id
        FROM silver_mechanism_relationships r
        LEFT JOIN silver_scientific_evidence e ON r.evidence_id = e.evidence_id
        WHERE r.evidence_id IS NOT NULL AND e.evidence_id IS NULL
        """
    ).fetchall()
    assert len(orphan_evidence) == 0, f"Found orphan evidence references: {orphan_evidence}"


# ==============================================================================
# TEST SUITE 6: GOLD ANALYTICAL VIEWS
# ==============================================================================


def test_v_ingredient_comparisons_view(db_conn):
    """Verify v_ingredient_comparisons returns complete normalized profiles."""
    comps = get_ingredient_comparisons(db_conn)
    assert len(comps) >= 8

    # Check egg white profile
    ew = next((c for c in comps if "egg_white" in c["ingredient_id"]), None)
    assert ew is not None, "Egg white comparison record not found."
    assert ew["protein_g"] > 10.0, f"Egg white protein should be >10g per 100g, got {ew['protein_g']}"
    assert ew["basis_amount_g"] == 100.0

    # Check heavy cream profile
    cream = next((c for c in comps if "heavy_cream" in c["ingredient_id"]), None)
    assert cream is not None, "Heavy cream record not found."
    assert cream["fat_total_g"] > 30.0, f"Heavy cream fat should be >30g per 100g, got {cream['fat_total_g']}"


def test_v_mechanism_evidence_view(db_conn):
    """Verify v_mechanism_evidence provides end-to-end evidence attribution."""
    evidence_rows = get_mechanism_evidence(db_conn)
    assert len(evidence_rows) > 0

    # Verify presence of peer-reviewed evidence grades
    grades = {r["evidence_grade"] for r in evidence_rows}
    assert "PEER_REVIEWED" in grades


def test_v_formulation_tradeoffs_view(db_conn):
    """Verify v_formulation_tradeoffs calculates accurate recipe nutrition and stability."""
    trades = get_formulation_tradeoffs(db_conn)
    assert len(trades) >= 5

    # Look for low fat mousse and whipped cream
    whipped_cream = next((t for t in trades if "whipped" in t["recipe_id"] or "cream" in t["recipe_id"]), None)
    mousse = next((t for t in trades if "mousse" in t["recipe_id"]), None)
    milk_foam = next((t for t in trades if "milk_foam" in t["recipe_id"]), None)

    assert whipped_cream is not None
    assert float(whipped_cream["norm_fat_g_per_100g"]) >= 25.0
    assert float(whipped_cream["estimated_foam_stability_index"]) >= 90.0

    if mousse:
        # Low fat mousse achieves high stability (>75) despite low fat (<5%)
        assert float(mousse["norm_fat_g_per_100g"]) < 5.0
        assert float(mousse["estimated_foam_stability_index"]) >= 75.0

    if milk_foam:
        # Unstabilized whole milk foam has low stability index (<30)
        assert float(milk_foam["estimated_foam_stability_index"]) < 30.0


def test_v_foam_stability_factors_view(db_conn):
    """Verify v_foam_stability_factors returns overrun and drainage metrics."""
    factors = get_foam_stability_factors(db_conn)
    assert len(factors) >= 8

    ew = next((f for f in factors if "egg_white" in f["ingredient_id"]), None)
    assert ew is not None
    assert ew["overrun_pct"] >= 500.0, f"Expected egg white overrun >= 500%, got {ew['overrun_pct']}"


# ==============================================================================
# TEST SUITE 7: INGREDIENT SUBSTITUTION SIMULATION
# ==============================================================================


def test_substitution_simulation(db_conn):
    """
    Test substituting egg white with whole milk in classic French meringue.
    Should demonstrate dramatic increase in fat, drop in protein, and negative stability impact.
    """
    result = simulate_substitution(
        db_conn,
        base_recipe_id="rec_classic_meringue",
        target_ingredient_id="ing_egg_white",
        substitute_ingredient_id="ing_whole_milk",
    )

    assert result["recipe_id"] == "rec_classic_meringue"
    assert result["target_ingredient"] == "ing_egg_white"
    assert result["substitute_ingredient"] == "ing_whole_milk"

    # Deltas
    deltas = result["deltas_100g"]
    assert deltas["delta_fat_g"] > 0, "Substituting with milk should increase fat"
    assert deltas["delta_protein_g"] < 0, "Substituting egg white with milk should reduce protein"

    # Stability impact
    assert result["stability_impact"]["estimated_stability_delta"] < 0
    assert "lipid" in result["stability_impact"]["mechanistic_explanation"].lower() or "fat" in result["stability_impact"]["mechanistic_explanation"].lower()


# ==============================================================================
# TEST SUITE 8: INVERSE FORMULATION OPTIMIZER & SUBSTITUTION APIS
# ==============================================================================

def test_api_inverse_optimizer_solve():
    """Verify POST /api/optimizer/solve generates valid recipe and Stokes kinetics."""
    from fastapi.testclient import TestClient
    from FOODMECHANICS.app.main import app
    client = TestClient(app)

    res = client.post("/api/optimizer/solve", json={
        "target_system": "whipped_foam",
        "max_fat_g": 0.5,
        "min_protein_g": 3.0,
        "target_overrun_pct": 380.0,
        "min_stability_hours": 24.0,
        "dietary_flags": ["vegan"]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["solved_recipe"]) >= 4
    assert len(data["stokes_kinetics_24h"]) == 11
    assert "Stokes" in data["thermodynamic_proof"] or "yield stress" in data["thermodynamic_proof"].lower()
    assert data["databricks_telemetry"]["engine"]


def test_api_substitution_analyze():
    """Verify POST /api/substitution/analyze calculates deltas and scientific risk."""
    from fastapi.testclient import TestClient
    from FOODMECHANICS.app.main import app
    client = TestClient(app)

    res = client.post("/api/substitution/analyze", json={
        "recipe_id": "rec_whipped_cream",
        "target_ingredient_id": "ing_heavy_cream_36",
        "substitute_ingredient_id": "ing_aquafaba",
        "substitute_mass_g": 85.0,
        "compensator_ingredient_id": "ing_xanthan_gum",
        "compensator_mass_g": 0.35
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["nutritional_and_rheological_deltas"]["fat_total_g"] < 0
    assert len(data["scientific_risk_assessment"]) > 10
    assert data["literature_citation"]["doi"]


def test_api_recipes_catalog():
    """Verify GET /api/recipes returns baseline culinary formulas."""
    from fastapi.testclient import TestClient
    from FOODMECHANICS.app.main import app
    client = TestClient(app)

    res = client.get("/api/recipes")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 5
    assert any(r["recipe_id"] == "rec_whipped_cream" for r in data["recipes"])
