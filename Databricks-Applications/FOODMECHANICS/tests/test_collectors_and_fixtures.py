"""
FOODMECHANICS Test Suite: Ingestion Collectors, Raw Payloads, and Fixture Integrity
"""

import os
import json
import pytest
from FOODMECHANICS.ingestion.sources.usda_collector import USDAFoodCollector
from FOODMECHANICS.ingestion.sources.openfoodfacts_collector import OpenFoodFactsCollector
from FOODMECHANICS.ingestion.sources.pubchem_food_collector import PubChemFoodCollector
from FOODMECHANICS.ingestion.sources.scientific_lit_collector import ScientificLiteratureCollector

PERMITTED_CONFIDENCES = {"PEER_REVIEWED", "EMPIRICAL", "HYPOTHESIS", "ANNOUNCEMENT", "EVALUATED"}


def test_usda_collector():
    collector = USDAFoodCollector()
    results = collector.collect(dry_run=True)
    assert len(results) >= 10, "Expected at least 10 foundation USDA ingredients"

    for item in results:
        assert "ingredient_key" in item
        assert "fdc_id" in item
        assert item["serving_basis"] == "100g"
        nutrients = item["nutritional_profile_per_100g"]
        assert "energy_kcal" in nutrients
        assert "protein_g" in nutrients
        assert "water_g" in nutrients
        assert "total_fat_g" in nutrients

        # Verify evidence record
        evidence = item["evidence"]
        assert evidence["source_id"].startswith("usda:")
        assert evidence["source_type"] == "usda_fdc"
        assert evidence["evidence_confidence"] in PERMITTED_CONFIDENCES
        assert os.path.exists(evidence["raw_payload_ref"]), f"Raw file must exist: {evidence['raw_payload_ref']}"


def test_openfoodfacts_collector():
    collector = OpenFoodFactsCollector()
    results = collector.collect(dry_run=True)
    assert len(results) >= 5, "Expected at least 5 commercial OFF products"

    for item in results:
        assert "barcode" in item
        assert "product_name" in item
        assert item["serving_basis"] == "100g"
        nutrients = item["nutritional_profile_per_100g"]
        assert "energy_kcal" in nutrients
        assert "protein_g" in nutrients

        evidence = item["evidence"]
        assert evidence["source_id"].startswith("off:")
        assert evidence["source_type"] == "openfoodfacts"
        assert evidence["evidence_confidence"] in PERMITTED_CONFIDENCES
        assert os.path.exists(evidence["raw_payload_ref"]), f"Raw file must exist: {evidence['raw_payload_ref']}"


def test_pubchem_food_collector():
    collector = PubChemFoodCollector()
    results = collector.collect(dry_run=True)
    assert len(results) >= 10, "Expected at least 10 food chemistry compounds"

    for item in results:
        assert "compound_key" in item
        assert "name" in item
        assert "molecular_weight_da" in item
        assert "classification" in item

        evidence = item["evidence"]
        assert evidence["source_type"] == "pubchem"
        assert evidence["evidence_confidence"] in PERMITTED_CONFIDENCES
        assert os.path.exists(evidence["raw_payload_ref"]), f"Raw file must exist: {evidence['raw_payload_ref']}"


def test_scientific_lit_collector():
    collector = ScientificLiteratureCollector()
    results = collector.collect(dry_run=True)
    assert len(results) >= 5, "Expected at least 5 peer-reviewed papers"

    for item in results:
        assert "doi" in item
        assert "title" in item
        assert "authors" in item
        assert "key_findings" in item

        evidence = item["evidence"]
        assert evidence["source_id"].startswith("doi:")
        assert evidence["source_type"] == "scientific_lit"
        assert evidence["evidence_confidence"] == "PEER_REVIEWED"
        assert os.path.exists(evidence["raw_payload_ref"]), f"Raw file must exist: {evidence['raw_payload_ref']}"


def test_food_fixtures_json_structure():
    fixtures_path = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures", "food_fixtures.json")
    assert os.path.exists(fixtures_path), "food_fixtures.json must exist"

    with open(fixtures_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate top-level keys
    assert "metadata" in data
    assert "ingredients" in data
    assert "compounds" in data
    assert "mechanisms" in data
    assert "formulation_comparisons" in data

    # 1. Validate Ingredients
    ingredients = data["ingredients"]
    assert len(ingredients) >= 10
    required_ids = {
        "ing_egg_white_raw", "ing_egg_white_dried", "ing_whole_egg_raw",
        "ing_heavy_cream", "ing_whole_milk", "ing_skim_milk",
        "ing_aquafaba_liquid", "ing_sucrose_granulated", "ing_xanthan_gum",
        "ing_gelatin_powder", "ing_cream_of_tartar"
    }
    present_ids = {ing["id"] for ing in ingredients}
    assert required_ids.issubset(present_ids), f"Missing required ingredients: {required_ids - present_ids}"

    for ing in ingredients:
        assert "density_g_per_ml" in ing
        assert "nutritional_profile_per_100g" in ing
        assert "functional_properties" in ing
        assert "foam_expansion_overrun_pct" in ing["functional_properties"]
        assert "evidence" in ing
        assert ing["evidence"]["evidence_confidence"] in PERMITTED_CONFIDENCES

    # 2. Validate Compounds
    compounds = data["compounds"]
    assert len(compounds) >= 8
    compound_ids = {c["compound_id"] for c in compounds}
    assert "comp_ovalbumin" in compound_ids
    assert "comp_ovotransferrin" in compound_ids
    assert "comp_ovomucin" in compound_ids
    assert "comp_casein_micelle" in compound_ids
    assert "comp_beta_lactoglobulin" in compound_ids
    assert "comp_triglycerides" in compound_ids
    assert "comp_phospholipids" in compound_ids

    # 3. Validate Mechanisms
    mechanisms = data["mechanisms"]
    assert len(mechanisms) >= 4
    mechanism_ids = {m["mechanism_id"] for m in mechanisms}
    assert "mech_protein_mechanical_denaturation" in mechanism_ids
    assert "mech_fat_interference_foam_disruption" in mechanism_ids
    assert "mech_acid_electrostatic_stabilization" in mechanism_ids
    assert "mech_sugar_drainage_retardation" in mechanism_ids

    # 4. Validate Formulations
    formulations = data["formulation_comparisons"]
    assert len(formulations) >= 3
    for form in formulations:
        assert "formulation_id" in form
        assert "ingredients_formula" in form
        assert "target_attributes" in form
        attrs = form["target_attributes"]
        assert "foam_expansion_overrun_pct" in attrs
        assert "foam_stability_half_life_min" in attrs
        assert "drainage_rate_ml_per_10min" in attrs
        assert "evidence" in form
