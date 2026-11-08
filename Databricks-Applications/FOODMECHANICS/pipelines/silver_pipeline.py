"""Silver Pipeline for FOODMECHANICS.

Normalizes raw Bronze records into canonical entities, enforces the strict 100g nutritional
normalization contract, calculates mass percentages, verifies foreign key integrity, and constructs
the food chemistry mechanism relationship graph with full evidence attribution.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import duckdb

logger = logging.getLogger("foodmechanics.silver")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def init_silver_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Initialize Silver layer schema in DuckDB."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS silver_ingredients (
            ingredient_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            common_name VARCHAR,
            category VARCHAR NOT NULL,
            default_serving_g DOUBLE NOT NULL,
            density_g_ml DOUBLE,
            description TEXT,
            source_id VARCHAR,
            source_type VARCHAR,
            source_url VARCHAR,
            evidence_confidence VARCHAR,
            raw_payload_ref VARCHAR
        );

        CREATE TABLE IF NOT EXISTS silver_nutritional_profiles (
            profile_id VARCHAR PRIMARY KEY,
            ingredient_id VARCHAR NOT NULL,
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
            source_id VARCHAR,
            source_type VARCHAR,
            source_url VARCHAR,
            evidence_confidence VARCHAR,
            raw_payload_ref VARCHAR,
            FOREIGN KEY (ingredient_id) REFERENCES silver_ingredients(ingredient_id)
        );

        CREATE TABLE IF NOT EXISTS silver_scientific_evidence (
            evidence_id VARCHAR PRIMARY KEY,
            doi_or_citation VARCHAR NOT NULL,
            publication_year INTEGER NOT NULL,
            journal VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            evidence_grade VARCHAR NOT NULL,
            findings_summary TEXT NOT NULL,
            source_id VARCHAR
        );

        CREATE TABLE IF NOT EXISTS silver_chemical_compounds (
            compound_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            molecular_formula VARCHAR,
            pubchem_cid VARCHAR,
            compound_class VARCHAR NOT NULL,
            role_in_food TEXT,
            source_id VARCHAR
        );

        CREATE TABLE IF NOT EXISTS silver_food_mechanisms (
            mechanism_id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            mechanism_type VARCHAR NOT NULL,
            scientific_summary TEXT NOT NULL,
            thermodynamic_or_kinetic_basis TEXT NOT NULL,
            key_variables TEXT,
            source_id VARCHAR
        );

        CREATE TABLE IF NOT EXISTS silver_functional_properties (
            property_id VARCHAR PRIMARY KEY,
            ingredient_id VARCHAR NOT NULL,
            property_type VARCHAR NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value DOUBLE NOT NULL,
            unit VARCHAR NOT NULL,
            test_conditions TEXT,
            confidence VARCHAR NOT NULL,
            is_inferred BOOLEAN DEFAULT FALSE,
            evidence_id VARCHAR,
            source_id VARCHAR,
            FOREIGN KEY (ingredient_id) REFERENCES silver_ingredients(ingredient_id)
        );

        CREATE TABLE IF NOT EXISTS silver_mechanism_relationships (
            rel_id VARCHAR PRIMARY KEY,
            ingredient_or_compound_id VARCHAR NOT NULL,
            mechanism_id VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            effect_direction VARCHAR NOT NULL,
            evidence_id VARCHAR,
            source_id VARCHAR,
            FOREIGN KEY (mechanism_id) REFERENCES silver_food_mechanisms(mechanism_id)
        );

        CREATE TABLE IF NOT EXISTS silver_formulations (
            recipe_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            total_mass_g DOUBLE NOT NULL,
            target_attribute TEXT NOT NULL,
            instructions_summary TEXT,
            source_id VARCHAR
        );

        CREATE TABLE IF NOT EXISTS silver_formulation_items (
            recipe_id VARCHAR NOT NULL,
            ingredient_id VARCHAR NOT NULL,
            mass_g DOUBLE NOT NULL,
            proportion_pct DOUBLE NOT NULL,
            PRIMARY KEY (recipe_id, ingredient_id),
            FOREIGN KEY (recipe_id) REFERENCES silver_formulations(recipe_id),
            FOREIGN KEY (ingredient_id) REFERENCES silver_ingredients(ingredient_id)
        );
        """
    )
    logger.info("Silver tables initialized successfully.")


def normalize_nutrition_100g(
    nutrients_raw: Dict[str, Any],
    serving_size_g: float,
    precision: int = 4,
) -> Dict[str, float]:
    """
    Mandatory contract: Normalize any input serving nutritional payload strictly to a 100g basis.

    Formula:
        val_100g = round((val_raw / serving_size_g) * 100.0, precision)
    """
    if serving_size_g <= 0:
        raise ValueError(f"Invalid serving size for normalization: {serving_size_g}g")

    scale = 100.0 / float(serving_size_g)

    def _get(keys: List[str], default: float = 0.0) -> float:
        for k in keys:
            if k in nutrients_raw and nutrients_raw[k] is not None:
                try:
                    return float(nutrients_raw[k])
                except (ValueError, TypeError):
                    pass
        return default

    energy = _get(["energy_kcal", "calories_kcal", "energy", "calories"])
    protein = _get(["protein_g", "protein"])
    fat_total = _get(["fat_total_g", "total_fat_g", "fat_g", "fat"])
    fat_sat = _get(["fat_saturated_g", "saturated_fat_g", "saturated_fat"])
    carb = _get(["carb_g", "carbohydrates_g", "carbs_g", "carbohydrate_g", "carbs"])
    sugars = _get(["sugars_g", "sugar_g", "sugars", "sugar"])
    fiber = _get(["fiber_g", "dietary_fiber_g", "fiber"])
    sodium = _get(["sodium_mg", "sodium"])
    water = _get(["water_g", "moisture_g", "water"])

    # If serving size is already 100g, no scaling needed
    if abs(serving_size_g - 100.0) < 0.001:
        return {
            "energy_kcal": round(energy, precision),
            "protein_g": round(protein, precision),
            "fat_total_g": round(fat_total, precision),
            "fat_saturated_g": round(fat_sat, precision),
            "carb_g": round(carb, precision),
            "sugars_g": round(sugars, precision),
            "fiber_g": round(fiber, precision),
            "sodium_mg": round(sodium, precision),
            "water_g": round(water, precision),
        }

    return {
        "energy_kcal": round(energy * scale, precision),
        "protein_g": round(protein * scale, precision),
        "fat_total_g": round(fat_total * scale, precision),
        "fat_saturated_g": round(fat_sat * scale, precision),
        "carb_g": round(carb * scale, precision),
        "sugars_g": round(sugars * scale, precision),
        "fiber_g": round(fiber * scale, precision),
        "sodium_mg": round(sodium * scale, precision),
        "water_g": round(water * scale, precision),
    }


def process_scientific_evidence(con: duckdb.DuckDBPyConnection) -> int:
    """Extract and load scientific evidence records."""
    rows = con.execute(
        """
        SELECT source_id, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    count = 0
    seen_ids = set()

    for source_id, payload_json in rows:
        payload = json.loads(payload_json)

        # Check explicit evidence records
        if "findings_summary" in payload or "doi_or_citation" in payload or payload.get("evidence_id"):
            ev_id = payload.get("evidence_id") or source_id
            doi = payload.get("doi_or_citation") or payload.get("doi") or ev_id
            pub_year = int(payload.get("publication_year", 2010))
            journal = payload.get("journal", "Food Science Research")
            title = payload.get("title", f"Scientific Investigation {ev_id}")
            grade = payload.get("evidence_grade", "PEER_REVIEWED")
            findings = payload.get("findings_summary") or payload.get("summary") or "Empirical measurement and analysis."

            if ev_id not in seen_ids:
                con.execute(
                    """
                    INSERT OR REPLACE INTO silver_scientific_evidence (
                        evidence_id, doi_or_citation, publication_year, journal,
                        title, evidence_grade, findings_summary, source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [ev_id, doi, pub_year, journal, title, grade, findings, source_id],
                )
                seen_ids.add(ev_id)
                count += 1

        # Check embedded evidence blocks
        embedded = payload.get("evidence")
        if isinstance(embedded, dict) and "evidence_id" in embedded:
            ev_id = embedded["evidence_id"]
            if ev_id not in seen_ids:
                doi = embedded.get("doi", embedded.get("citation", ev_id))
                pub_year = int(embedded.get("year", 2010))
                journal = embedded.get("journal", "Food Science Journal")
                title = embedded.get("title", f"Study {ev_id}")
                grade = embedded.get("evidence_grade", "PEER_REVIEWED")
                findings = embedded.get("key_finding", "Documented evidence.")
                con.execute(
                    """
                    INSERT OR REPLACE INTO silver_scientific_evidence (
                        evidence_id, doi_or_citation, publication_year, journal,
                        title, evidence_grade, findings_summary, source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [ev_id, doi, pub_year, journal, title, grade, findings, source_id],
                )
                seen_ids.add(ev_id)
                count += 1

    logger.info("Processed %d scientific evidence publications.", count)
    return count


def process_chemical_compounds(con: duckdb.DuckDBPyConnection) -> int:
    """Extract and load chemical compounds."""
    rows = con.execute(
        """
        SELECT source_id, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    count = 0
    seen_ids = set()

    for source_id, payload_json in rows:
        payload = json.loads(payload_json)
        compound_id = payload.get("compound_id")
        if compound_id and compound_id not in seen_ids:
            name = payload.get("name") or payload.get("common_name") or compound_id
            formula = payload.get("molecular_formula") or payload.get("formula")
            cid = str(payload.get("pubchem_cid") or payload.get("uniprot_id") or "")
            c_class = payload.get("compound_class") or payload.get("functional_classification") or payload.get("structure_type") or "biochemical"
            role = payload.get("role_in_food") or payload.get("chemical_rationale") or payload.get("biological_source", "")

            con.execute(
                """
                INSERT OR REPLACE INTO silver_chemical_compounds (
                    compound_id, name, molecular_formula, pubchem_cid,
                    compound_class, role_in_food, source_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [compound_id, name, formula, cid, c_class, role, source_id],
            )
            seen_ids.add(compound_id)
            count += 1

    logger.info("Processed %d chemical compounds.", count)
    return count


def process_food_mechanisms(con: duckdb.DuckDBPyConnection) -> int:
    """Extract and load food mechanisms."""
    rows = con.execute(
        """
        SELECT source_id, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    count = 0
    seen_ids = set()

    for source_id, payload_json in rows:
        payload = json.loads(payload_json)
        mech_id = payload.get("mechanism_id") or payload.get("id")
        if mech_id and mech_id.startswith("mech_") and mech_id not in seen_ids:
            title = payload.get("title") or payload.get("name") or mech_id
            m_type = payload.get("mechanism_type") or payload.get("category") or "foam_dynamics"
            summary = payload.get("scientific_summary") or payload.get("summary") or ""
            basis = payload.get("thermodynamic_or_kinetic_basis") or payload.get("molecular_physics_rationale") or payload.get("kinetics_and_thermodynamics") or "Thermodynamic and kinetic principles."
            variables = payload.get("key_variables") or (json.dumps(payload.get("critical_factors")) if payload.get("critical_factors") else None)

            con.execute(
                """
                INSERT OR REPLACE INTO silver_food_mechanisms (
                    mechanism_id, title, mechanism_type, scientific_summary,
                    thermodynamic_or_kinetic_basis, key_variables, source_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [mech_id, title, m_type, summary, basis, variables, source_id],
            )
            seen_ids.add(mech_id)
            count += 1

    logger.info("Processed %d food mechanisms.", count)
    return count


def process_ingredients_and_nutrition(con: duckdb.DuckDBPyConnection) -> Tuple[int, int]:
    """Extract and normalize ingredients and strictly 100g-basis nutrition profiles."""
    rows = con.execute(
        """
        SELECT source_id, source_type, source_url, content_hash, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    ing_count = 0
    nut_count = 0
    seen_ing_ids = set()

    for source_id, source_type, source_url, content_hash, payload_json in rows:
        payload = json.loads(payload_json)
        # Ignore functional property experiment records
        if "metric_value" in payload or "property_id" in payload:
            continue

        ing_id = payload.get("ingredient_id") or payload.get("id")

        if not ing_id or not ing_id.startswith("ing_"):
            continue

        if ing_id in seen_ing_ids:
            continue

        name = payload.get("name") or payload.get("common_name") or ing_id
        common_name = payload.get("common_name") or payload.get("name")
        category = payload.get("category") or "unclassified"
        serving_size_g = float(payload.get("serving_size_g") or payload.get("serving_basis_g") or 100.0)
        density_g_ml = payload.get("density_g_ml") or payload.get("density_g_per_ml")
        description = payload.get("description") or ""
        evidence_confidence = payload.get("evidence_confidence") or "CANONICAL"

        # 1. Upsert ingredient
        con.execute(
            """
            INSERT OR REPLACE INTO silver_ingredients (
                ingredient_id, name, common_name, category, default_serving_g,
                density_g_ml, description, source_id, source_type, source_url,
                evidence_confidence, raw_payload_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ing_id,
                name,
                common_name,
                category,
                serving_size_g,
                density_g_ml,
                description,
                source_id,
                source_type,
                source_url,
                evidence_confidence,
                content_hash,
            ],
        )
        seen_ing_ids.add(ing_id)
        ing_count += 1

        # 2. Extract nutrition payload (supports nutrients_raw, nutritional_profile_per_100g, or nutrients_per_100g)
        raw_nutrients = payload.get("nutrients_raw")
        if raw_nutrients:
            norm = normalize_nutrition_100g(raw_nutrients, serving_size_g=serving_size_g)
        else:
            profile_dict = payload.get("nutritional_profile_per_100g") or payload.get("nutrients_per_100g") or {}
            norm = normalize_nutrition_100g(profile_dict, serving_size_g=100.0)

        profile_id = f"nut_{ing_id.removeprefix('ing_')}"
        is_inferred = payload.get("is_inferred", False)

        con.execute(
            """
            INSERT OR REPLACE INTO silver_nutritional_profiles (
                profile_id, ingredient_id, basis_amount_g, energy_kcal, protein_g,
                fat_total_g, fat_saturated_g, carb_g, sugars_g, fiber_g, sodium_mg,
                water_g, is_inferred, source_id, source_type, source_url,
                evidence_confidence, raw_payload_ref
            ) VALUES (?, ?, 100.0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                profile_id,
                ing_id,
                norm["energy_kcal"],
                norm["protein_g"],
                norm["fat_total_g"],
                norm["fat_saturated_g"],
                norm["carb_g"],
                norm["sugars_g"],
                norm["fiber_g"],
                norm["sodium_mg"],
                norm["water_g"],
                is_inferred,
                source_id,
                source_type,
                source_url,
                evidence_confidence,
                content_hash,
            ],
        )
        nut_count += 1

    logger.info("Processed %d ingredients and %d normalized 100g nutritional profiles.", ing_count, nut_count)
    return ing_count, nut_count


def process_functional_properties(con: duckdb.DuckDBPyConnection) -> int:
    """Extract and load functional properties, linking to ingredients."""
    rows = con.execute(
        """
        SELECT source_id, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    count = 0
    seen_ids = set()

    # Get valid ingredients to ensure foreign key safety
    valid_ings = {r[0] for r in con.execute("SELECT ingredient_id FROM silver_ingredients").fetchall()}

    for source_id, payload_json in rows:
        payload = json.loads(payload_json)

        # 1. Direct functional property records
        if "metric_value" in payload and payload.get("property_id"):
            prop_id = payload.get("property_id")
            ing_id = payload.get("ingredient_id")
            if ing_id in valid_ings and prop_id not in seen_ids:
                prop_type = payload.get("property_type", "functional_metric")
                metric_name = payload.get("metric_name")
                metric_val = float(payload.get("metric_value", 0.0))
                unit = payload.get("unit", "")
                conditions = payload.get("test_conditions", "")
                confidence = payload.get("confidence", "HIGH")
                is_inferred = payload.get("is_inferred", False)
                evidence_id = payload.get("evidence_id")

                con.execute(
                    """
                    INSERT OR REPLACE INTO silver_functional_properties (
                        property_id, ingredient_id, property_type, metric_name,
                        metric_value, unit, test_conditions, confidence,
                        is_inferred, evidence_id, source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        prop_id,
                        ing_id,
                        prop_type,
                        metric_name,
                        metric_val,
                        unit,
                        conditions,
                        confidence,
                        is_inferred,
                        evidence_id,
                        source_id,
                    ],
                )
                seen_ids.add(prop_id)
                count += 1

        # 2. Embedded functional properties inside ingredient records
        ing_id = payload.get("ingredient_id") or payload.get("id")
        if ing_id in valid_ings and isinstance(payload.get("functional_properties"), dict):
            funcs = payload["functional_properties"]
            for fname, fval in funcs.items():
                prop_id = f"prop_{ing_id}_{fname}"
                if prop_id not in seen_ids:
                    val_num = 0.0
                    unit = ""
                    if isinstance(fval, (int, float)):
                        val_num = float(fval)
                        unit = "%" if "overrun" in fname or "pct" in fname else ("min" if "min" in fname else "")
                    elif isinstance(fval, dict):
                        val_num = float(fval.get("value", 0.0))
                        unit = str(fval.get("unit", ""))

                    con.execute(
                        """
                        INSERT OR REPLACE INTO silver_functional_properties (
                            property_id, ingredient_id, property_type, metric_name,
                            metric_value, unit, test_conditions, confidence,
                            is_inferred, evidence_id, source_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            prop_id,
                            ing_id,
                            "foaming_or_rheology",
                            fname,
                            val_num,
                            unit,
                            f"Standard assay for {fname}",
                            "HIGH",
                            False,
                            None,
                            source_id,
                        ],
                    )
                    seen_ids.add(prop_id)
                    count += 1

    logger.info("Processed %d functional properties.", count)
    return count


def process_mechanism_relationships(con: duckdb.DuckDBPyConnection) -> int:
    """Extract and load mechanism relationships with integrity verification."""
    rows = con.execute(
        """
        SELECT source_id, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    count = 0
    seen_ids = set()

    # Get valid mechanisms for foreign key integrity
    valid_mechs = {r[0] for r in con.execute("SELECT mechanism_id FROM silver_food_mechanisms").fetchall()}

    for source_id, payload_json in rows:
        payload = json.loads(payload_json)

        # 1. Explicit relationship records
        if "effect_direction" in payload and payload.get("rel_id"):
            rel_id = payload.get("rel_id")
            entity_id = payload.get("ingredient_or_compound_id")
            mech_id = payload.get("mechanism_id")
            role = payload.get("role", "modifier")
            direction = payload.get("effect_direction", "NEUTRAL")
            evidence_id = payload.get("evidence_id")

            if mech_id in valid_mechs and rel_id not in seen_ids:
                con.execute(
                    """
                    INSERT OR REPLACE INTO silver_mechanism_relationships (
                        rel_id, ingredient_or_compound_id, mechanism_id, role,
                        effect_direction, evidence_id, source_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [rel_id, entity_id, mech_id, role, direction, evidence_id, source_id],
                )
                seen_ids.add(rel_id)
                count += 1

        # 2. Embedded dominant mechanisms in ingredients
        ing_id = payload.get("ingredient_id") or payload.get("id")
        if ing_id and isinstance(payload.get("dominant_mechanisms"), list):
            for idx, dm in enumerate(payload["dominant_mechanisms"]):
                mech_id = dm.get("mechanism_id") if isinstance(dm, dict) else dm
                role = dm.get("role", "participant") if isinstance(dm, dict) else "participant"
                direction = dm.get("effect", "PROMOTES_OVERRUN") if isinstance(dm, dict) else "PROMOTES_OVERRUN"

                if mech_id in valid_mechs:
                    rel_id = f"rel_{ing_id}_{mech_id}_{idx}"
                    if rel_id not in seen_ids:
                        con.execute(
                            """
                            INSERT OR REPLACE INTO silver_mechanism_relationships (
                                rel_id, ingredient_or_compound_id, mechanism_id, role,
                                effect_direction, evidence_id, source_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            [rel_id, ing_id, mech_id, role, direction, None, source_id],
                        )
                        seen_ids.add(rel_id)
                        count += 1

    logger.info("Processed %d mechanism relationships.", count)
    return count


def process_formulations_and_items(con: duckdb.DuckDBPyConnection) -> Tuple[int, int]:
    """Extract formulations and calculate exact mass balance item proportions."""
    rows = con.execute(
        """
        SELECT source_id, payload_json
        FROM bronze_food_raw
        """
    ).fetchall()

    form_count = 0
    item_count = 0
    seen_recipes = set()

    valid_ings = {r[0] for r in con.execute("SELECT ingredient_id FROM silver_ingredients").fetchall()}

    for source_id, payload_json in rows:
        payload = json.loads(payload_json)
        recipe_id = payload.get("recipe_id") or payload.get("formulation_id")

        if not recipe_id or recipe_id in seen_recipes:
            continue

        raw_items = payload.get("items") or payload.get("ingredients_formula")
        if not raw_items or not isinstance(raw_items, list):
            continue

        name = payload.get("name") or payload.get("title") or recipe_id
        category = payload.get("category", "formulation")
        target_attribute = payload.get("target_attribute") or str(payload.get("target_attributes", ""))
        instructions = payload.get("instructions_summary") or payload.get("description", "")

        # Calculate exact total mass
        calculated_total = sum(float(it.get("mass_g", 0.0)) for it in raw_items)
        declared_total = float(payload.get("total_mass_g", calculated_total))
        total_mass_g = declared_total if abs(declared_total - calculated_total) < 0.05 and declared_total > 0 else calculated_total

        if total_mass_g <= 0:
            total_mass_g = 100.0

        con.execute(
            """
            INSERT OR REPLACE INTO silver_formulations (
                recipe_id, name, category, total_mass_g,
                target_attribute, instructions_summary, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [recipe_id, name, category, total_mass_g, target_attribute, instructions, source_id],
        )
        seen_recipes.add(recipe_id)
        form_count += 1

        for it in raw_items:
            ing_id = it.get("ingredient_id")
            if not ing_id:
                continue

            # Map raw ingredient alias if needed
            if ing_id not in valid_ings:
                # Try prefixing or finding match
                candidate = ing_id if ing_id.startswith("ing_") else f"ing_{ing_id}"
                if candidate in valid_ings:
                    ing_id = candidate
                else:
                    # Look for base name match (e.g. ing_egg_white vs ing_egg_white_raw)
                    matching = [v for v in valid_ings if ing_id.replace("_raw", "") in v or v in ing_id]
                    if matching:
                        ing_id = matching[0]
                    else:
                        continue

            mass_g = float(it.get("mass_g", 0.0))
            prop_pct = round((mass_g / total_mass_g) * 100.0, 4)

            con.execute(
                """
                INSERT OR REPLACE INTO silver_formulation_items (
                    recipe_id, ingredient_id, mass_g, proportion_pct
                ) VALUES (?, ?, ?, ?)
                """,
                [recipe_id, ing_id, mass_g, prop_pct],
            )
            item_count += 1

    logger.info("Processed %d formulations and %d formulation items.", form_count, item_count)
    return form_count, item_count


def run_silver_pipeline(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """
    Run complete Silver transformation pipeline against populated Bronze layer.
    Returns entity counts dict.
    """
    init_silver_tables(con)

    evidence_count = process_scientific_evidence(con)
    compounds_count = process_chemical_compounds(con)
    mechanisms_count = process_food_mechanisms(con)
    ing_count, nut_count = process_ingredients_and_nutrition(con)
    props_count = process_functional_properties(con)
    rels_count = process_mechanism_relationships(con)
    forms_count, items_count = process_formulations_and_items(con)

    summary = {
        "silver_ingredients": ing_count,
        "silver_nutritional_profiles_100g": nut_count,
        "silver_scientific_evidence": evidence_count,
        "silver_chemical_compounds": compounds_count,
        "silver_food_mechanisms": mechanisms_count,
        "silver_functional_properties": props_count,
        "silver_mechanism_relationships": rels_count,
        "silver_formulations": forms_count,
        "silver_formulation_items": items_count,
    }
    logger.info("Silver pipeline completed successfully: %s", summary)
    return summary
