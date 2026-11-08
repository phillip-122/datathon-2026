# FOODMECHANICS — Data Sources & Ingestion Specifications

## 1. Executive Overview

FOODMECHANICS relies on an evidence-first, multi-tier data acquisition architecture designed to provide verifiable, reproducible food science and culinary colloid intelligence. Every nutritional quantity, chemical property, and interfacial mechanism is traced to an authoritative origin:
- **Nutritional Standards:** United States Department of Agriculture (USDA FoodData Central)
- **Commercial Formulations:** Open Food Facts (OFF)
- **Chemical Compounds & Biomacromolecules:** PubChem PUG REST & UniProt Knowledgebase
- **Interfacial Mechanics & Evidence:** Crossref Peer-Reviewed Food Colloid Literature

All collectors implement dual-mode execution (`--live` and `--mock`/`--dry-run`) and automatically store untransformed raw payloads into `FOODMECHANICS/data/raw/` preserving HTTP headers, query parameters, and capture timestamps.

---

## 2. Ingestion Source Catalog

### 2.1 USDA FoodData Central (FDC)

- **Source Identifier:** `usda_fdc`
- **Base URL:** `https://api.nal.usda.gov/fdc/v1`
- **Primary Use Case:** Authoritative foundation nutritional data normalized strictly to a **100g serving basis**.
- **Key Endpoints:**
  - `GET /food/{fdcId}`: Retrieves full micro- and macronutrient breakdowns for a specific foundation food.
  - `GET /foods/search`: Full-text search with facets for foundation foods, SR Legacy, and survey foods.
- **Authentication:** `api_key` query parameter or `X-Api-Key` HTTP header. Defaults to public `DEMO_KEY` if `USDA_API_KEY` is unset.
- **Rate Limits:**
  - `DEMO_KEY`: 30 requests/hour, 50 requests/day.
  - Registered API Key: 1,000 requests/hour.
- **Collector Implementation:** `FOODMECHANICS/ingestion/sources/usda_collector.py`
- **Foundation Food Coverage:**
  | Food Item | FDC ID | USDA Data Type | Standard Basis | Key Analytes |
  |---|---|---|---|---|
  | Egg white, raw | `172183` | SR Legacy | 100g | 10.9g protein, 87.6g water, 0.17g fat |
  | Egg white, dried powder | `172186` | SR Legacy | 100g | 82.4g protein, 5.8g water, 0.04g fat |
  | Whole egg, raw | `171287` | SR Legacy | 100g | 12.6g protein, 9.51g fat, 76.2g water |
  | Heavy whipping cream (36% fat) | `170859` | SR Legacy | 100g | 36.1g fat, 2.84g protein, 57.7g water |
  | Whole milk (3.25% fat) | `171265` | SR Legacy | 100g | 3.25g fat, 3.15g protein, 113mg Ca |
  | Skim milk (nonfat) | `171269` | SR Legacy | 100g | 0.08g fat, 3.37g protein, 122mg Ca |
  | Chickpea broth / aquafaba | `173757` | Foundation | 100g | 1.15g protein, 3.10g carb, 93.8g water |
  | Granulated sucrose | `169655` | SR Legacy | 100g | 99.98g sucrose, 387 kcal |
  | Xanthan gum | `173516` | Additive/Survey | 100g | 78g soluble fiber, 2900mg Na |
  | Unflavored gelatin powder | `172189` | SR Legacy | 100g | 85.6g protein, 0g carb, 0.1g fat |
  | Cream of tartar (potassium bitartrate) | `173473` | SR Legacy | 100g | 16,500mg K, 61.5g organic acid |

---

### 2.2 Open Food Facts (OFF)

- **Source Identifier:** `openfoodfacts`
- **Base URL:** `https://world.openfoodfacts.org`
- **Primary Use Case:** Commercial branded composition, additive tagging (E-numbers), and ultra-processing classifications (NOVA score).
- **Key Endpoints:**
  - `GET /api/v2/product/{barcode}.json`: Complete product record including ingredients hierarchy, nutriments per 100g, and additives.
  - `GET /cgi/search.pl`: Search query endpoint for branded products across categories.
- **Authentication:** None required. Mandatory User-Agent header specifying application name and contact email.
- **Rate Limits:** 100 requests/minute.
- **Collector Implementation:** `FOODMECHANICS/ingestion/sources/openfoodfacts_collector.py`
- **Representative Products:**
  - `0035000000010`: Eggland's Best Liquid Egg Whites 100% Pure (NOVA 1)
  - `0041271000123`: Horizon Organic Heavy Whipping Cream 36% (E418 gellan stabilized, NOVA 3)
  - `0852697001056`: Oatly Barista Edition Foamable Oat Beverage (Dipotassium phosphate buffer, NOVA 4)
  - `0043000052303`: Knox Original Unflavored Gelatin (NOVA 2)
  - `0857317005124`: Anthony's Pure Food Grade Xanthan Gum (E415, NOVA 2)
  - `0024100440700`: Simple Truth Organic Chickpeas in Water / Aquafaba (NOVA 3)

---

### 2.3 PubChem Food Chemistry & Macromolecules

- **Source Identifier:** `pubchem` (and `uniprot`)
- **Base URL:** `https://pubchem.ncbi.nlm.nih.gov/rest/pug`
- **Primary Use Case:** Pure chemical structures, exact molecular weights, InChIKeys, solubility limits, and biological macromolecular classifications.
- **Key Endpoints:**
  - `GET /compound/cid/{cid}/JSON`: Full compound description, canonical SMILES, formulas.
  - `GET /compound/name/{name}/JSON`: Look up compound records by chemical name.
- **Authentication:** None.
- **Rate Limits:** Maximum 5 requests/second. Request volume should not exceed 400 requests/minute.
- **Collector Implementation:** `FOODMECHANICS/ingestion/sources/pubchem_food_collector.py`
- **Curated Compound Scope:**
  - `Sucrose` (CID 5988): Disaccharide, MW 342.30 Da, water-binding viscosity agent.
  - `Potassium Bitartrate` (CID 8875): Tartaric acid monopotassium salt, MW 188.18 Da, pH 3.5-4.8 buffer.
  - `Phosphatidylcholine / Lecithin` (CID 439341): Zwitterionic phospholipid, MW ~760 Da, fast-adsorbing surfactant.
  - `Tripalmitin` (CID 11147): Saturated triglyceride model, MW 807.30 Da, hydrophobic crystal network former.
  - `Ovalbumin` (UniProt P01012): Major egg serpin glycoprotein, MW 44.5 kDa, pI 4.5, mechanical unfolding foaming.
  - `Ovotransferrin / Conalbumin` (UniProt P02789): Iron-binding glycoprotein, MW 76 kDa, pI 6.0, 61°C thermal gel onset.
  - `Ovomucin` (UniProt P39880): Sulfated fibrous glycoprotein, MW 8.3 MDa, elastic lamellar stabilizer.
  - `Beta-Lactoglobulin` (UniProt P02754): Bovine whey lipocalin dimer, MW 18.4 kDa, pI 5.2, thiol-active cross-linker at 78°C.
  - `Casein Micelle Complex` (UniProt P02662/P02663/P02666/P02668): Colloidal phosphoprotein (50-300 nm), steric stabilizer.
  - `Xanthan Polysaccharide` (CAS 11138-66-2): Rigid 5-fold helical cellulosic gum, MW 2-4 MDa, yield stress builder.
  - `Gelatin Polypeptide` (CAS 9000-70-8): Hydrolyzed collagen triple-helix, MW 20-250 kDa, reversible 35°C sol-gel network.

---

### 2.4 Scientific Literature Corpus (Crossref REST API)

- **Source Identifier:** `scientific_lit`
- **Base URL:** `https://api.crossref.org`
- **Primary Use Case:** Peer-reviewed food colloid citations, empirical interfacial mechanics evidence, and kinetic constants.
- **Key Endpoints:**
  - `GET /works/{doi}`: Detailed metadata, authors, publication date, journal, citation count.
  - `GET /works?query={query}&rows=5`: Bibliographic keyword search.
- **Authentication:** None. Header `User-Agent: AppName/1.0 (mailto:team@org.com)` grants access to Polite Pool.
- **Rate Limits:** 50 requests/second on the polite pool.
- **Collector Implementation:** `FOODMECHANICS/ingestion/sources/scientific_lit_collector.py`
- **Foundational Citations:**
  - **Foegeding, Luck, & Davis (2006):** *Factors determining the physical properties of protein foams.* Food Hydrocolloids 20(2-3), 284-292. DOI: `10.1016/j.foodhyd.2005.03.014`.
  - **Dickinson (1999):** *Adsorbed protein layers at fluid interfaces: interactions, structure and surface rheology.* Colloids and Surfaces B: Biointerfaces 15(2), 161-176. DOI: `10.1016/S0927-7765(99)00042-9`.
  - **Damodaran (2005):** *Protein stabilization of emulsions and foams.* Journal of Food Science 70(3), R54-R66. DOI: `10.1111/j.1365-2621.2005.tb07150.x`.
  - **Raikos (2010):** *Milk heat treatment: Effects on the heat stability of milk proteins and the rheological properties of milk gels.* Food Chemistry 122(1), 1-10. DOI: `10.1016/j.foodchem.2010.02.046`.
  - **Murray (2007):** *Stabilization of foams and emulsions by mixtures of surface active food components.* Curr. Opin. Colloid & Interface Sci. 12(4-5), 232-241. DOI: `10.1016/j.cocis.2007.07.009`.
  - **Mustapha & Stathopoulos (2020):** *Aquafaba, a promising plant-based functional alternative to egg white: A review.* Food Science and Technology International 26(4), 312-324. DOI: `10.1177/1082013219890696`.

---

## 3. Raw Payload Architecture

All collectors inherit from `BaseFoodCollector` (`FOODMECHANICS/ingestion/sources/base_collector.py`), which guarantees:
1. **Raw Storage Isolation:** Every query response is saved to `FOODMECHANICS/data/raw/{source_name}/{item_id}_{timestamp}.json`.
2. **Untransformed Audit Log:** Original HTTP status, target URL, and raw JSON response body are retained without truncation.
3. **Execution Mode Flexibility:**
   - Command-line execution with `--live` fetches real HTTP responses.
   - Command-line execution with `--mock` or `--dry-run` re-plays high-fidelity cached payloads with identical schema structures.
4. **Standard Provenance Generation:** Generates `EvidenceRecord` compliant with `coordination/INTEGRATION_CONTRACT.md`.
