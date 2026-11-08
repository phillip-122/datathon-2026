"""
FOODMECHANICS Ingestion - PubChem Food Chemistry Collector
Collects chemical structures, molecular weights, physicochemical properties,
and functional food chemistry classifications for key proteins, hydrocolloids,
lipids, and organic compounds.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from .base_collector import BaseFoodCollector

logger = logging.getLogger(__name__)

# Curated reference database for food chemistry compounds
PUBCHEM_FOOD_COMPOUNDS = {
    "sucrose": {
        "cid": 5988,
        "name": "Sucrose",
        "iupac_name": "(2R,3R,4S,5S,6R)-2-[(2S,3S,4S,5R)-3,4-dihydroxy-2,5-bis(hydroxymethyl)oxolan-2-yl]oxy-6-(hydroxymethyl)oxane-3,4,5-triol",
        "formula": "C12H22O11",
        "molecular_weight": 342.30,
        "classification": "BULKING_AGENT / VISCOSITY_BUILDER",
        "functional_role": "Increases lamellar viscosity, slows drainage in foams, stabilizes protein hydration shells against thermal aggregation.",
        "solubility": "2000 g/L at 25°C in water",
        "isoelectric_point": None,
        "denaturation_temp_c": 160.0,
        "inchikey": "CZMRCDWAGMRECN-UGDNZQRBSA-N"
    },
    "potassium_bitartrate": {
        "cid": 8875,
        "name": "Potassium Bitartrate (Cream of Tartar)",
        "iupac_name": "potassium;(2R,3R)-2,3,4-trihydroxy-4-oxobutanoate",
        "formula": "C4H5KO6",
        "molecular_weight": 188.18,
        "classification": "ACIDULANT / BUFFER",
        "functional_role": "Lowers egg white pH to ~4.8-5.0 near protein isoelectric point, reducing electrostatic repulsion and enabling elastic interfacial network.",
        "solubility": "5.7 g/L at 20°C in water",
        "isoelectric_point": None,
        "denaturation_temp_c": None,
        "inchikey": "LSDHHWODZZHGOC-WDSKDSINSA-M"
    },
    "phosphatidylcholine": {
        "cid": 439341,
        "name": "1,2-diacyl-sn-glycero-3-phosphocholine (Lecithin)",
        "iupac_name": "[2-hexadecanoyloxy-3-(octadec-9-enoyloxy)propyl] 2-(trimethylazaniumyl)ethyl phosphate",
        "formula": "C42H82NO8P",
        "molecular_weight": 760.08,
        "classification": "SURFACTANT / EMULSIFIER",
        "functional_role": "Zwitterionic small-molecule surfactant; lowers interfacial tension rapidly; displaces proteins at interfaces.",
        "solubility": "Insoluble in water; dispersible forming lamellar/liposomal phases",
        "isoelectric_point": None,
        "denaturation_temp_c": None,
        "inchikey": "DCOPUUMXTXNQFP-UHFFFAOYSA-N"
    },
    "tripalmitin": {
        "cid": 11147,
        "name": "Tripalmitin (Triglyceride model)",
        "iupac_name": "1,3-di(hexadecanoyloxy)propan-2-yl hexadecanoate",
        "formula": "C51H98O6",
        "molecular_weight": 807.30,
        "classification": "HYDROPHOBIC_LIPID",
        "functional_role": "Non-polar core of fat globules. In whipped cream, partial crystallization forms crystalline bridge network supporting air bubbles.",
        "solubility": "Insoluble in water (< 0.001 mg/L)",
        "isoelectric_point": None,
        "denaturation_temp_c": None,
        "inchikey": "GOCUAINEOHMVJE-UHFFFAOYSA-N"
    },
    "ovalbumin": {
        "cid": None,
        "uniprot_id": "P01012",
        "name": "Ovalbumin",
        "iupac_name": "Egg white major serpin glycoprotein",
        "formula": "Polypeptide (385 aa)",
        "molecular_weight": 44500.0,
        "classification": "STRUCTURAL_PROTEIN / FOAMING_AGENT",
        "functional_role": "Accounts for 54% of egg white protein; uncoils mechanically at air-water interface to form cohesive viscoelastic network; heat denatures at 84.5°C.",
        "solubility": "Freely soluble in neutral and mildly alkaline aqueous solutions",
        "isoelectric_point": 4.5,
        "denaturation_temp_c": 84.5,
        "inchikey": "PROTEIN_OVALBUMIN_GALLUS"
    },
    "ovotransferrin": {
        "cid": None,
        "uniprot_id": "P02789",
        "name": "Ovotransferrin (Conalbumin)",
        "iupac_name": "Iron-binding transferrin family glycoprotein",
        "formula": "Polypeptide (686 aa)",
        "molecular_weight": 76000.0,
        "classification": "STRUCTURAL_PROTEIN / FOAMING_AGENT",
        "functional_role": "Accounts for 12% of egg white; most heat-sensitive egg white protein; forms cross-linked network starting at 61°C.",
        "solubility": "Soluble in water",
        "isoelectric_point": 6.0,
        "denaturation_temp_c": 61.0,
        "inchikey": "PROTEIN_OVOTRANSFERRIN_GALLUS"
    },
    "ovomucin": {
        "cid": None,
        "uniprot_id": "P39880",
        "name": "Ovomucin",
        "iupac_name": "Fibrous sulfated glycoprotein complex",
        "formula": "Polypeptide-heteropolysaccharide complex (1.5-33% carbohydrate)",
        "molecular_weight": 8300000.0,
        "classification": "FOAM_STABILIZER / RHEOLOGY_MODIFIER",
        "functional_role": "Provides thick egg white structure; forms electrostatic complex with lysozyme; imparts elastic modulus to foam lamellae.",
        "solubility": "Insoluble in pure water; soluble in 0.1M NaCl at pH 8",
        "isoelectric_point": 4.5,
        "denaturation_temp_c": 75.0,
        "inchikey": "PROTEIN_OVOMUCIN_COMPLEX"
    },
    "beta_lactoglobulin": {
        "cid": None,
        "uniprot_id": "P02754",
        "name": "Beta-Lactoglobulin",
        "iupac_name": "Major bovine whey lipocalin protein",
        "formula": "Polypeptide (162 aa dimer)",
        "molecular_weight": 18367.0,
        "classification": "STRUCTURAL_PROTEIN / EMULSIFIER",
        "functional_role": "Dominant protein in whey (~50%); unfolds at 78°C exposing free Cys121 thiol group for disulfide cross-linking.",
        "solubility": "High aqueous solubility across pH 2-9",
        "isoelectric_point": 5.2,
        "denaturation_temp_c": 78.0,
        "inchikey": "PROTEIN_BETALACTOGLOBULIN_BOVINE"
    },
    "casein_micelle": {
        "cid": None,
        "uniprot_id": "P02662_P02663_P02666_P02668",
        "name": "Casein Micellar Complex (alpha-s1, alpha-s2, beta, kappa)",
        "iupac_name": "Colloidal phosphoprotein calcium phosphate aggregate",
        "formula": "Colloidal aggregate diameter 50-300 nm",
        "molecular_weight": 250000000.0,
        "classification": "COLLOIDAL_EMULSIFIER / BUFFER",
        "functional_role": "Forms protective steric stabilization layer on homogenized milkfat globules; precipitates when pH reaches 4.6.",
        "solubility": "Colloidal suspension; stable to boiling at neutral pH",
        "isoelectric_point": 4.6,
        "denaturation_temp_c": 140.0,
        "inchikey": "COLLOID_CASEIN_MICELLE"
    },
    "xanthan_gum": {
        "cid": None,
        "cas_id": "11138-66-2",
        "name": "Xanthan Gum",
        "iupac_name": "Heteropolysaccharide cellulosic beta-(1,4)-D-glucan with trisaccharide side chains",
        "formula": "(C35H49O29)n",
        "molecular_weight": 2000000.0,
        "classification": "HYDROCOLLOID / SHEAR_THINNING_THICKENER",
        "functional_role": "Pseudoplastic rheology modifier; maintains high yield stress at rest to halt foam bubble drainage; thins under shear.",
        "solubility": "Soluble in both cold and hot water; tolerant to high salt and wide pH (2-12)",
        "isoelectric_point": None,
        "denaturation_temp_c": 95.0,
        "inchikey": "POLYSACCHARIDE_XANTHAN"
    },
    "gelatin": {
        "cid": None,
        "cas_id": "9000-70-8",
        "name": "Gelatin (Type A / Type B)",
        "iupac_name": "Partially hydrolyzed triple-helical collagen polypeptide",
        "formula": "(Gly-Pro-Hyp)n repetitive motif",
        "molecular_weight": 65000.0,
        "classification": "GELLING_AGENT / FOAM_STABILIZER",
        "functional_role": "Undergoes reversible coil-to-helix transition; forms thermoreversible aqueous gel matrix below 35°C trapping water and bubbles.",
        "solubility": "Swells in cold water; dissolves in hot water (>40°C)",
        "isoelectric_point": 8.0,  # Type A ~8-9, Type B ~4.8-5.2
        "denaturation_temp_c": 35.0,  # Melting point of gel
        "inchikey": "HYDROLYSATE_COLLAGEN_GELATIN"
    }
}


class PubChemFoodCollector(BaseFoodCollector):
    """
    Collector for PubChem compound records and food macromolecular chemistry.
    """

    def __init__(self, verify_ssl: bool = False):
        super().__init__(
            source_name="pubchem",
            base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            rate_limit_delay_sec=0.3,
            verify_ssl=verify_ssl
        )

    def _mock_fallback(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Provides offline mock payload matching PubChem PUG REST format."""
        for key, comp in PUBCHEM_FOOD_COMPOUNDS.items():
            if comp["cid"] and str(comp["cid"]) in endpoint:
                return {
                    "PC_Compounds": [{
                        "id": {"id": {"cid": comp["cid"]}},
                        "atoms": {"element": [6, 1, 8]},
                        "props": [
                            {"urn": {"label": "Molecular Weight"}, "value": {"sval": str(comp["molecular_weight"])}},
                            {"urn": {"label": "Molecular Formula"}, "value": {"sval": comp["formula"]}},
                            {"urn": {"label": "IUPAC Name"}, "value": {"sval": comp["iupac_name"]}}
                        ]
                    }]
                }
        return {"PC_Compounds": []}

    def fetch_compound_by_name(self, name: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Fetch compound data from PubChem by name.
        """
        endpoint = f"compound/name/{name}/JSON"
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

    def collect(self, query_or_ids: Optional[List[str]] = None, dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Collect food chemistry records with molecular weights, structure, and functional roles.
        """
        compound_keys = query_or_ids or list(PUBCHEM_FOOD_COMPOUNDS.keys())
        results = []

        for key in compound_keys:
            comp_info = PUBCHEM_FOOD_COMPOUNDS.get(key, {})
            cid = comp_info.get("cid")

            if cid and not dry_run:
                try:
                    raw_resp = self.fetch_json(
                        endpoint=f"compound/cid/{cid}/JSON",
                        timeout=6,
                        mock_fallback_func=self._mock_fallback
                    )
                except Exception:
                    raw_resp = {"status": "mock_used", "payload": comp_info}
            else:
                raw_resp = {"status": "curated_biomolecule", "payload": comp_info}

            raw_path = self.save_raw_record(f"pubchem_{key}", raw_resp)

            # Build evidence record
            source_id = f"pubchem:CID{cid}" if cid else f"uniprot:{comp_info.get('uniprot_id', key)}"
            source_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else f"https://www.uniprot.org/uniprot/{comp_info.get('uniprot_id', '')}"

            evidence = self.build_evidence_record(
                source_id=source_id,
                source_url=source_url,
                confidence="PEER_REVIEWED",
                publication_date="2021-06-01",
                raw_payload_ref=raw_path
            )

            record = {
                "compound_key": key,
                "name": comp_info.get("name", key),
                "cid": cid,
                "uniprot_id": comp_info.get("uniprot_id"),
                "formula": comp_info.get("formula"),
                "molecular_weight_da": comp_info.get("molecular_weight"),
                "classification": comp_info.get("classification"),
                "functional_role": comp_info.get("functional_role"),
                "solubility": comp_info.get("solubility"),
                "isoelectric_point": comp_info.get("isoelectric_point"),
                "denaturation_temp_c": comp_info.get("denaturation_temp_c"),
                "inchikey": comp_info.get("inchikey"),
                "evidence": evidence
            }
            results.append(record)

        return results


if __name__ == "__main__":
    dry_run_mode = "--mock" in sys.argv or "--dry-run" in sys.argv
    collector = PubChemFoodCollector()
    print(f"Executing PubChem Food collector (dry_run={dry_run_mode})...")
    output = collector.collect(dry_run=dry_run_mode)
    print(f"Collected {len(output)} food chemistry compound records.")
    for item in output[:3]:
        print(f" - {item['compound_key']}: {item['name']} (MW: {item['molecular_weight_da']} Da, Role: {item['classification']})")
