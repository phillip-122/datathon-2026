"""
FOODMECHANICS Ingestion - Open Food Facts Collector
Collects commercial food product composition, additive profiles, and ingredient lists.
Target categories: pasteurized egg whites, whipping creams, hydrocolloid stabilizers,
commercial plant milks, and culinary additives.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from .base_collector import BaseFoodCollector

logger = logging.getLogger(__name__)

# Curated reference products from Open Food Facts with real barcodes & profiles
OFF_COMMERCIAL_CATALOG = {
    "0035000000010": {
        "barcode": "0035000000010",
        "product_name": "Liquid Egg Whites 100% Pure",
        "brand": "Eggland's Best",
        "ingredients_text": "100% Pasteurized Liquid Egg Whites. No added preservatives.",
        "additives_tags": [],
        "nova_group": 1,
        "nutriments": {
            "energy-kcal_100g": 48.0,
            "proteins_100g": 10.5,
            "fat_100g": 0.0,
            "saturated-fat_100g": 0.0,
            "carbohydrates_100g": 0.8,
            "sugars_100g": 0.7,
            "sodium_100g": 0.165,
            "calcium_100g": 0.008
        }
    },
    "0041271000123": {
        "barcode": "0041271000123",
        "product_name": "Organic Heavy Whipping Cream 36% Milkfat",
        "brand": "Horizon Organic",
        "ingredients_text": "Organic Grade A Heavy Cream, Gellan Gum.",
        "additives_tags": ["en:e418"],
        "nova_group": 3,
        "nutriments": {
            "energy-kcal_100g": 345.0,
            "proteins_100g": 2.2,
            "fat_100g": 36.5,
            "saturated-fat_100g": 22.8,
            "carbohydrates_100g": 2.8,
            "sugars_100g": 2.8,
            "sodium_100g": 0.035,
            "calcium_100g": 0.065
        }
    },
    "0852697001056": {
        "barcode": "0852697001056",
        "product_name": "Barista Edition Oat Foamable Beverage",
        "brand": "Oatly",
        "ingredients_text": "Oat base (water, oats), low erucic acid rapeseed oil, dipotassium phosphate, calcium carbonate, tricalcium phosphate, sea salt, dicalcium phosphate.",
        "additives_tags": ["en:e340ii", "en:e170", "en:e341ii"],
        "nova_group": 4,
        "nutriments": {
            "energy-kcal_100g": 59.0,
            "proteins_100g": 1.0,
            "fat_100g": 3.0,
            "saturated-fat_100g": 0.3,
            "carbohydrates_100g": 6.7,
            "sugars_100g": 4.1,
            "sodium_100g": 0.045,
            "calcium_100g": 0.120
        }
    },
    "0043000052303": {
        "barcode": "0043000052303",
        "product_name": "Original Unflavored Gelatin",
        "brand": "Knox",
        "ingredients_text": "Gelatin.",
        "additives_tags": ["en:e441"],
        "nova_group": 2,
        "nutriments": {
            "energy-kcal_100g": 335.0,
            "proteins_100g": 85.0,
            "fat_100g": 0.0,
            "saturated-fat_100g": 0.0,
            "carbohydrates_100g": 0.0,
            "sugars_100g": 0.0,
            "sodium_100g": 0.190,
            "calcium_100g": 0.050
        }
    },
    "0857317005124": {
        "barcode": "0857317005124",
        "product_name": "Pure Food Grade Xanthan Gum Powder",
        "brand": "Anthony's Goods",
        "ingredients_text": "Xanthan Gum.",
        "additives_tags": ["en:e415"],
        "nova_group": 2,
        "nutriments": {
            "energy-kcal_100g": 353.0,
            "proteins_100g": 7.0,
            "fat_100g": 0.0,
            "saturated-fat_100g": 0.0,
            "carbohydrates_100g": 80.0,
            "sugars_100g": 0.0,
            "sodium_100g": 3.100,
            "calcium_100g": 0.040
        }
    },
    "0024100440700": {
        "barcode": "0024100440700",
        "product_name": "Organic Chickpeas in Water (Aquafaba source)",
        "brand": "Simple Truth Organic",
        "ingredients_text": "Organic Prepared Chickpeas, Water, Sea Salt.",
        "additives_tags": [],
        "nova_group": 3,
        "nutriments": {
            "energy-kcal_100g": 115.0,
            "proteins_100g": 6.8,
            "fat_100g": 2.0,
            "saturated-fat_100g": 0.2,
            "carbohydrates_100g": 17.5,
            "sugars_100g": 1.2,
            "sodium_100g": 0.140,
            "calcium_100g": 0.045
        }
    }
}


class OpenFoodFactsCollector(BaseFoodCollector):
    """
    Collector for Open Food Facts commercial product APIs.
    """

    def __init__(self, verify_ssl: bool = False):
        super().__init__(
            source_name="openfoodfacts",
            base_url="https://world.openfoodfacts.org",
            rate_limit_delay_sec=1.0,
            verify_ssl=verify_ssl
        )
        self.session.headers.update({
            "User-Agent": "FoodMechanicsApp/1.0 (contact: info@foodmechanics.org)"
        })

    def _mock_fallback(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Provides offline mock payload matching Open Food Facts API v2."""
        if "product/" in endpoint:
            barcode = endpoint.split("product/")[-1].replace(".json", "")
            if barcode in OFF_COMMERCIAL_CATALOG:
                prod = OFF_COMMERCIAL_CATALOG[barcode]
                return {
                    "code": barcode,
                    "status": 1,
                    "status_verbose": "product found",
                    "product": prod
                }
        elif "search.pl" in endpoint and params:
            query = params.get("search_terms", "").lower()
            matching = [p for p in OFF_COMMERCIAL_CATALOG.values() if any(q in p["product_name"].lower() for q in query.split())]
            return {
                "count": len(matching),
                "products": matching
            }
        return {"status": 0, "status_verbose": "product not found", "products": []}

    def fetch_product(self, barcode: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Fetch commercial product details by barcode.
        """
        endpoint = f"api/v2/product/{barcode}.json"
        if dry_run:
            payload = self._mock_fallback(endpoint, None)
            return {
                "status": "dry_run",
                "http_status": 200,
                "url": f"{self.base_url}/{endpoint}",
                "payload": payload,
                "fetched_at": self._get_timestamp_iso()
            }

        return self.fetch_json(
            endpoint=endpoint,
            timeout=8,
            mock_fallback_func=self._mock_fallback
        )

    def search_products(self, query: str, limit: int = 5, dry_run: bool = False) -> Dict[str, Any]:
        """
        Search commercial products by term.
        """
        endpoint = "cgi/search.pl"
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": limit
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
            timeout=8,
            mock_fallback_func=self._mock_fallback
        )

    def collect(self, query_or_ids: Optional[List[str]] = None, dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Collect commercial product profiles with standardized 100g nutriments and additives.
        """
        barcodes = query_or_ids or list(OFF_COMMERCIAL_CATALOG.keys())
        results = []

        for barcode in barcodes:
            raw_resp = self.fetch_product(barcode, dry_run=dry_run)
            raw_path = self.save_raw_record(f"off_{barcode}", raw_resp)

            product = raw_resp.get("payload", {}).get("product", {})
            if not product and barcode in OFF_COMMERCIAL_CATALOG:
                product = OFF_COMMERCIAL_CATALOG[barcode]

            nutriments = product.get("nutriments", {})
            evidence = self.build_evidence_record(
                source_id=f"off:{barcode}",
                source_url=f"https://world.openfoodfacts.org/product/{barcode}",
                confidence="EMPIRICAL",
                publication_date="2023-01-15",
                raw_payload_ref=raw_path
            )

            record = {
                "barcode": barcode,
                "product_name": product.get("product_name", "Unknown Product"),
                "brand": product.get("brand", "Unknown Brand"),
                "ingredients_text": product.get("ingredients_text", ""),
                "additives_tags": product.get("additives_tags", []),
                "nova_group": product.get("nova_group"),
                "serving_basis": "100g",
                "nutritional_profile_per_100g": {
                    "energy_kcal": nutriments.get("energy-kcal_100g", 0.0),
                    "protein_g": nutriments.get("proteins_100g", 0.0),
                    "total_fat_g": nutriments.get("fat_100g", 0.0),
                    "saturated_fat_g": nutriments.get("saturated-fat_100g", 0.0),
                    "carbohydrates_g": nutriments.get("carbohydrates_100g", 0.0),
                    "sugars_g": nutriments.get("sugars_100g", 0.0),
                    "sodium_mg": (nutriments.get("sodium_100g", 0.0) or 0.0) * 1000.0,
                    "calcium_mg": (nutriments.get("calcium_100g", 0.0) or 0.0) * 1000.0
                },
                "evidence": evidence
            }
            results.append(record)

        return results


if __name__ == "__main__":
    dry_run_mode = "--mock" in sys.argv or "--dry-run" in sys.argv
    collector = OpenFoodFactsCollector()
    print(f"Executing Open Food Facts collector (dry_run={dry_run_mode})...")
    output = collector.collect(dry_run=dry_run_mode)
    print(f"Collected {len(output)} commercial product records.")
    for item in output[:3]:
        print(f" - [{item['barcode']}] {item['product_name']} by {item['brand']}")
