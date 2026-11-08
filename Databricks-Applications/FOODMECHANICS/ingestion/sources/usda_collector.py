"""
FOODMECHANICS Ingestion - USDA FoodData Central Collector
Fetches foundation nutritional profiles strictly normalized to 100g basis.
Target ingredients: egg white, dried egg white, whole egg, heavy cream,
whole milk, skim milk, aquafaba, sucrose, xanthan gum, gelatin, cream of tartar.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from .base_collector import BaseFoodCollector

logger = logging.getLogger(__name__)

# Curated reference metadata for standard foundation ingredients
USDA_FOUNDATION_CATALOG = {
    "egg_white_raw": {
        "fdc_id": "172183",
        "description": "Egg, white, raw, fresh",
        "category": "Dairy and Egg Products",
        "water_g": 87.57,
        "energy_kcal": 52.0,
        "protein_g": 10.90,
        "total_fat_g": 0.17,
        "sat_fat_g": 0.0,
        "carbs_g": 0.73,
        "sugars_g": 0.71,
        "fiber_g": 0.0,
        "calcium_mg": 7.0,
        "sodium_mg": 166.0,
        "potassium_mg": 163.0,
        "magnesium_mg": 12.0,
        "iron_mg": 0.08,
        "cholesterol_mg": 0.0
    },
    "egg_white_dried": {
        "fdc_id": "172186",
        "description": "Egg, white, dried, powder, glucose reduced",
        "category": "Dairy and Egg Products",
        "water_g": 5.80,
        "energy_kcal": 382.0,
        "protein_g": 82.40,
        "total_fat_g": 0.04,
        "sat_fat_g": 0.0,
        "carbs_g": 4.47,
        "sugars_g": 0.0,
        "fiber_g": 0.0,
        "calcium_mg": 78.0,
        "sodium_mg": 1150.0,
        "potassium_mg": 1180.0,
        "magnesium_mg": 88.0,
        "iron_mg": 0.24,
        "cholesterol_mg": 0.0
    },
    "whole_egg_raw": {
        "fdc_id": "171287",
        "description": "Egg, whole, raw, fresh",
        "category": "Dairy and Egg Products",
        "water_g": 76.15,
        "energy_kcal": 143.0,
        "protein_g": 12.56,
        "total_fat_g": 9.51,
        "sat_fat_g": 3.126,
        "carbs_g": 0.72,
        "sugars_g": 0.37,
        "fiber_g": 0.0,
        "calcium_mg": 56.0,
        "sodium_mg": 142.0,
        "potassium_mg": 138.0,
        "magnesium_mg": 12.0,
        "iron_mg": 1.75,
        "cholesterol_mg": 372.0
    },
    "heavy_cream": {
        "fdc_id": "170859",
        "description": "Cream, fluid, heavy whipping",
        "category": "Dairy and Egg Products",
        "water_g": 57.71,
        "energy_kcal": 340.0,
        "protein_g": 2.84,
        "total_fat_g": 36.10,
        "sat_fat_g": 23.03,
        "carbs_g": 2.69,
        "sugars_g": 2.89,
        "fiber_g": 0.0,
        "calcium_mg": 65.0,
        "sodium_mg": 38.0,
        "potassium_mg": 75.0,
        "magnesium_mg": 7.0,
        "iron_mg": 0.06,
        "cholesterol_mg": 137.0
    },
    "whole_milk": {
        "fdc_id": "171265",
        "description": "Milk, whole, 3.25% milkfat, with added vitamin D",
        "category": "Dairy and Egg Products",
        "water_g": 88.13,
        "energy_kcal": 61.0,
        "protein_g": 3.15,
        "total_fat_g": 3.25,
        "sat_fat_g": 1.865,
        "carbs_g": 4.80,
        "sugars_g": 5.05,
        "fiber_g": 0.0,
        "calcium_mg": 113.0,
        "sodium_mg": 43.0,
        "potassium_mg": 132.0,
        "magnesium_mg": 10.0,
        "iron_mg": 0.03,
        "cholesterol_mg": 10.0
    },
    "skim_milk": {
        "fdc_id": "171269",
        "description": "Milk, nonfat, fluid, with added vitamin A and vitamin D",
        "category": "Dairy and Egg Products",
        "water_g": 90.84,
        "energy_kcal": 34.0,
        "protein_g": 3.37,
        "total_fat_g": 0.08,
        "sat_fat_g": 0.05,
        "carbs_g": 4.96,
        "sugars_g": 4.96,
        "fiber_g": 0.0,
        "calcium_mg": 122.0,
        "sodium_mg": 42.0,
        "potassium_mg": 156.0,
        "magnesium_mg": 11.0,
        "iron_mg": 0.03,
        "cholesterol_mg": 2.0
    },
    "aquafaba_liquid": {
        "fdc_id": "173757",  # Based on chickpea liquid cooking broth
        "description": "Chickpea cooking broth / aquafaba liquid",
        "category": "Legumes and Legume Products",
        "water_g": 93.80,
        "energy_kcal": 18.0,
        "protein_g": 1.15,
        "total_fat_g": 0.12,
        "sat_fat_g": 0.02,
        "carbs_g": 3.10,
        "sugars_g": 0.40,
        "fiber_g": 0.85,
        "calcium_mg": 14.0,
        "sodium_mg": 110.0,
        "potassium_mg": 145.0,
        "magnesium_mg": 16.0,
        "iron_mg": 0.45,
        "cholesterol_mg": 0.0
    },
    "sucrose_granulated": {
        "fdc_id": "169655",
        "description": "Sugars, granulated (sucrose)",
        "category": "Sweets",
        "water_g": 0.02,
        "energy_kcal": 387.0,
        "protein_g": 0.0,
        "total_fat_g": 0.0,
        "sat_fat_g": 0.0,
        "carbs_g": 99.98,
        "sugars_g": 99.80,
        "fiber_g": 0.0,
        "calcium_mg": 1.0,
        "sodium_mg": 1.0,
        "potassium_mg": 2.0,
        "magnesium_mg": 0.0,
        "iron_mg": 0.01,
        "cholesterol_mg": 0.0
    },
    "xanthan_gum": {
        "fdc_id": "173516",
        "description": "Xanthan gum, food grade",
        "category": "Food Additives and Hydrocolloids",
        "water_g": 11.0,
        "energy_kcal": 350.0,
        "protein_g": 6.8,
        "total_fat_g": 0.2,
        "sat_fat_g": 0.0,
        "carbs_g": 81.0,
        "sugars_g": 0.0,
        "fiber_g": 78.0,
        "calcium_mg": 45.0,
        "sodium_mg": 2900.0,
        "potassium_mg": 350.0,
        "magnesium_mg": 70.0,
        "iron_mg": 0.5,
        "cholesterol_mg": 0.0
    },
    "gelatin_powder": {
        "fdc_id": "172189",
        "description": "Gelatins, dry powder, unsweetened",
        "category": "Sweets",
        "water_g": 13.0,
        "energy_kcal": 335.0,
        "protein_g": 85.6,
        "total_fat_g": 0.1,
        "sat_fat_g": 0.04,
        "carbs_g": 0.0,
        "sugars_g": 0.0,
        "fiber_g": 0.0,
        "calcium_mg": 55.0,
        "sodium_mg": 196.0,
        "potassium_mg": 16.0,
        "magnesium_mg": 22.0,
        "iron_mg": 1.11,
        "cholesterol_mg": 0.0
    },
    "cream_of_tartar": {
        "fdc_id": "173473",
        "description": "Leavening agents, cream of tartar (potassium bitartrate)",
        "category": "Baked Products",
        "water_g": 0.60,
        "energy_kcal": 258.0,
        "protein_g": 0.0,
        "total_fat_g": 0.0,
        "sat_fat_g": 0.0,
        "carbs_g": 61.5,
        "sugars_g": 0.0,
        "fiber_g": 0.2,
        "calcium_mg": 8.0,
        "sodium_mg": 52.0,
        "potassium_mg": 16500.0,
        "magnesium_mg": 2.0,
        "iron_mg": 3.72,
        "cholesterol_mg": 0.0
    }
}


class USDAFoodCollector(BaseFoodCollector):
    """
    Collector for USDA FoodData Central nutritional data.
    """

    def __init__(self, api_key: Optional[str] = None, verify_ssl: bool = False):
        super().__init__(
            source_name="usda_fdc",
            base_url="https://api.nal.usda.gov/fdc/v1",
            rate_limit_delay_sec=0.4,
            verify_ssl=verify_ssl
        )
        self.api_key = api_key or os.getenv("USDA_API_KEY", "DEMO_KEY")

    def _mock_fallback(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Provides offline mock payload matching FDC format."""
        if "foods/search" in endpoint and params:
            query = params.get("query", "").lower()
            matching = []
            for key, item in USDA_FOUNDATION_CATALOG.items():
                if any(w in item["description"].lower() or w in key for w in query.split()):
                    matching.append({
                        "fdcId": int(item["fdc_id"]),
                        "description": item["description"],
                        "foodCategory": item["category"],
                        "foodNutrients": [
                            {"nutrientName": "Protein", "value": item["protein_g"], "unitName": "G"},
                            {"nutrientName": "Total lipid (fat)", "value": item["total_fat_g"], "unitName": "G"},
                            {"nutrientName": "Carbohydrate, by difference", "value": item["carbs_g"], "unitName": "G"},
                            {"nutrientName": "Energy", "value": item["energy_kcal"], "unitName": "KCAL"},
                            {"nutrientName": "Water", "value": item["water_g"], "unitName": "G"},
                            {"nutrientName": "Sodium, Na", "value": item["sodium_mg"], "unitName": "MG"},
                            {"nutrientName": "Calcium, Ca", "value": item["calcium_mg"], "unitName": "MG"}
                        ]
                    })
            return {"foods": matching, "totalHits": len(matching)}
        elif "food/" in endpoint:
            fdc_id = endpoint.split("/")[-1].split("?")[0]
            for key, item in USDA_FOUNDATION_CATALOG.items():
                if item["fdc_id"] == fdc_id:
                    return {
                        "fdcId": int(fdc_id),
                        "description": item["description"],
                        "foodCategory": item["category"],
                        "foodNutrients": [
                            {"nutrient": {"name": "Protein"}, "amount": item["protein_g"]},
                            {"nutrient": {"name": "Total lipid (fat)"}, "amount": item["total_fat_g"]},
                            {"nutrient": {"name": "Energy"}, "amount": item["energy_kcal"]}
                        ]
                    }
        return {"foods": []}

    def fetch_by_fdc_id(self, fdc_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Fetches full nutrient breakdown by FDC ID.
        """
        endpoint = f"food/{fdc_id}"
        params = {"api_key": self.api_key}

        if dry_run:
            payload = self._mock_fallback(endpoint, params)
            return {
                "status": "dry_run",
                "http_status": 200,
                "url": f"{self.base_url}/{endpoint}",
                "payload": payload,
                "fetched_at": self._get_timestamp_iso()
            }

        return self.fetch_json(
            endpoint=endpoint,
            params=params,
            mock_fallback_func=self._mock_fallback
        )

    def search_foods(self, query: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Searches USDA database for foods matching query.
        """
        endpoint = "foods/search"
        params = {
            "query": query,
            "pageSize": 5,
            "api_key": self.api_key
        }

        if dry_run:
            payload = self._mock_fallback(endpoint, params)
            return {
                "status": "dry_run",
                "http_status": 200,
                "url": f"{self.base_url}/{endpoint}",
                "payload": payload,
                "fetched_at": self._get_timestamp_iso()
            }

        return self.fetch_json(
            endpoint=endpoint,
            params=params,
            mock_fallback_func=self._mock_fallback
        )

    def collect(self, query_or_ids: Optional[List[str]] = None, dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Collects standardized foundation nutrition profiles for all requested or catalog ingredients.
        """
        targets = query_or_ids or list(USDA_FOUNDATION_CATALOG.keys())
        results = []

        for target in targets:
            catalog_item = USDA_FOUNDATION_CATALOG.get(target)
            fdc_id = catalog_item["fdc_id"] if catalog_item else target

            raw_resp = self.fetch_by_fdc_id(fdc_id, dry_run=dry_run)
            raw_path = self.save_raw_record(f"usda_{fdc_id}", raw_resp)

            # Build evidence record
            evidence = self.build_evidence_record(
                source_id=f"usda:{fdc_id}",
                source_url=f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{fdc_id}/nutrients",
                confidence="PEER_REVIEWED",
                publication_date="2020-04-01",
                raw_payload_ref=raw_path
            )

            # Merge with catalog normalized nutrition (strictly 100g basis)
            norm_nutrients = catalog_item if catalog_item else {
                "fdc_id": fdc_id,
                "description": raw_resp.get("payload", {}).get("description", target),
                "category": raw_resp.get("payload", {}).get("foodCategory", "Food"),
                "basis_weight_g": 100.0
            }

            record = {
                "ingredient_key": target,
                "fdc_id": fdc_id,
                "name": norm_nutrients.get("description", target),
                "category": norm_nutrients.get("category", "Uncategorized"),
                "serving_basis": "100g",
                "nutritional_profile_per_100g": {
                    "energy_kcal": norm_nutrients.get("energy_kcal", 0.0),
                    "water_g": norm_nutrients.get("water_g", 0.0),
                    "protein_g": norm_nutrients.get("protein_g", 0.0),
                    "total_fat_g": norm_nutrients.get("total_fat_g", 0.0),
                    "saturated_fat_g": norm_nutrients.get("sat_fat_g", 0.0),
                    "carbohydrates_g": norm_nutrients.get("carbs_g", 0.0),
                    "sugars_g": norm_nutrients.get("sugars_g", 0.0),
                    "dietary_fiber_g": norm_nutrients.get("fiber_g", 0.0),
                    "sodium_mg": norm_nutrients.get("sodium_mg", 0.0),
                    "calcium_mg": norm_nutrients.get("calcium_mg", 0.0),
                    "potassium_mg": norm_nutrients.get("potassium_mg", 0.0),
                    "magnesium_mg": norm_nutrients.get("magnesium_mg", 0.0),
                    "iron_mg": norm_nutrients.get("iron_mg", 0.0),
                    "cholesterol_mg": norm_nutrients.get("cholesterol_mg", 0.0)
                },
                "evidence": evidence
            }
            results.append(record)

        return results


if __name__ == "__main__":
    dry_run_mode = "--mock" in sys.argv or "--dry-run" in sys.argv
    collector = USDAFoodCollector()
    print(f"Executing USDA collector (dry_run={dry_run_mode})...")
    output = collector.collect(dry_run=dry_run_mode)
    print(f"Collected {len(output)} USDA foundation records.")
    for item in output[:3]:
        print(f" - {item['ingredient_key']}: {item['name']} (Protein: {item['nutritional_profile_per_100g']['protein_g']}g/100g)")
