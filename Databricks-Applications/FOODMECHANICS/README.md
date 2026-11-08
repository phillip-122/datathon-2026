# FOODMECHANICS: Food Science Investigation Platform

An evidence-grounded food science, interfacial mechanics, and formulation intelligence platform. Built with a dual-engine data architecture that seamlessly bridges local in-memory DuckDB analytical querying with live Databricks Serverless SQL and Unity Catalog.

---

## 🌟 Executive Summary & Problem Solved

### The Killer Demonstration:
> **"How can we make a lower-fat whipped dessert while preserving volume and structural stability?"**

Conventional whipped dessert foams rely heavily on **heavy whipping cream (36% milkfat)**, where air bubble walls are mechanically stabilized by a 3D network of **partially coalesced fat globules**. This produces high caloric density (345 kcal/100g) and saturated fat (23g/100g), and the foam collapses whenever temperature rises above 18–20°C as fat crystals melt into antifoaming liquid oil.

**FOODMECHANICS demonstrates the scientific solution:**
By replacing heavy cream with an engineered aqueous protein-hydrocolloid system (**Skim Milk + Xanthan Gum + Type A Gelatin + Sucrose**):
1. **99.8% Fat Reduction:** Drops total lipids from 32.8g to **0.07g per 100g**.
2. **73% Calorie Reduction:** Lowers caloric density from 344 kcal to **93 kcal per 100g**.
3. **Overrun & Aeration:** Milk whey proteins unfold rapidly at air-water interfaces to drive initial overrun (>300%).
4. **4x Drainage Resistance (Syneresis Arrest):** Xanthan gum builds continuous-phase pseudoplastic yield stress, retarding gravity drainage (Darcy's Law), while gelatin forms a thermo-reversible triple-helix gel scaffold that preserves structure up to 32°C and melts cleanly at human body temperature (35°C).

---

## 🏛️ Architecture & Integration Standards

FOODMECHANICS adheres strictly to the architectural standards defined in `coordination/INTEGRATION_CONTRACT.md`:

```
┌─────────────────────────────────────────────────────────────┐
│                       Modern Web UI                         │
│  (Dashboard, Matrix, Mechanism Explorer, Simulator, Genie)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│               `FOODMECHANICS/app/main.py`                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   FoodLocalDataProvider     │     │ FoodDatabricksDataProvider  │
│ (In-Memory DuckDB + JSON)   │     │ (databricks-sql-connector)  │
│ - Instant Sub-ms Response   │     │ - Unity Catalog Schemas     │
│ - Offline First Parity      │     │ - Serverless Starter WH     │
└─────────────────────────────┘     └─────────────────────────────┘
```

1. **Dual-Engine `DataProvider` Abstraction:** The backend communicates exclusively through `DataProvider` and `GenieProvider` interfaces. When Databricks credentials are present and reachable, it connects to Databricks SQL; if offline or unconfigured, it automatically falls back to in-memory DuckDB and local JSON fixtures without errors.
2. **Standard 100g Mass Basis Normalization:** All nutritional comparisons across all ingredients are strictly normalized to a standard **100g mass basis** (USDA FoodData Central standard), preventing misleading comparisons caused by differing serving sizes.
3. **Evidence & Provenance Tracking:** 100% of chemical claims, physical mechanisms, and nutrient figures maintain explicit attribution including source IDs, canonical DOIs, authors, publication years, journals, and confidence levels (`PEER_REVIEWED`, `EVALUATED`).

---

## 🚀 Quickstart & Running the Application

### 1. Prerequisites
- Python 3.10+
- Dependencies installed via `uv` or `pip`:
  ```bash
  uv pip install fastapi uvicorn pydantic duckdb pandas databricks-sql-connector python-dotenv httpx
  ```

### 2. Start the Web Server
Run the standalone launch script:
```bash
./FOODMECHANICS/app/run.sh
```

Or run directly with Python:
```bash
cd FOODMECHANICS/app
../../.venv/bin/python main.py
```

Open your browser to:
👉 **`http://localhost:8080`**

---

## 📡 REST API Reference

The FastAPI application provides structured JSON endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | `GET` | Health check and engine mode (`Databricks SQL` vs `Local DuckDB`). |
| `/api/metrics` | `GET` | Dashboard KPIs (tracked ingredients, mechanisms, evidence count, tests). |
| `/api/ingredients` | `GET` | Filter and search ingredients by category and keyword. |
| `/api/ingredients/{ingredient_id}` | `GET` | Detailed 100g nutritional profile, functional properties, and biochemical compounds. |
| `/api/compare?ids=...` | `GET` | Side-by-side comparison of 2 or more ingredients on 100g basis with trade-offs. |
| `/api/mechanisms` | `GET` | Catalog of physical food chemistry mechanisms. |
| `/api/mechanisms/{mechanism_id}/evidence`| `GET` | Supporting peer-reviewed literature records and DOIs for a mechanism. |
| `/api/evidence` | `GET` | Full catalog of scientific literature, DOIs, journals, and citations. |
| `/api/formulations/benchmarks` | `GET` | Pre-calculated formulation benchmarks (meringues, creams, vegan foams). |
| `/api/formulation/simulate` | `POST` | Physical chemistry simulation engine predicting mass balance, overrun, stability, and deltas. |
| `/api/genie/query` | `POST` | Natural language food science assistant with generated SQL and evidence sources. |
| `/api/search?q=...` | `GET` | Global keyword search across ingredients, mechanisms, and papers. |

### Example Formulation Simulation Request:
```bash
curl -X POST http://localhost:8080/api/formulation/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "lower_fat_whipped_dessert",
    "base_ingredients": [
      {"ingredient_id": "ing_heavy_cream", "mass_g": 100.0},
      {"ingredient_id": "ing_sucrose_granulated", "mass_g": 10.0}
    ],
    "substitutions": [
      {"ingredient_id": "ing_skim_milk", "mass_g": 100.0},
      {"ingredient_id": "ing_xanthan_gum", "mass_g": 0.25},
      {"ingredient_id": "ing_gelatin_powder", "mass_g": 1.2},
      {"ingredient_id": "ing_sucrose_granulated", "mass_g": 15.0}
    ]
  }'
```

---

## 🧪 Interactive Frontend Capabilities

The application frontend in `FOODMECHANICS/app/static/` is built with a reactive, dark/light mode responsive architecture:
- **Dashboard & KPIs:** Instant high-level metrics for ingredients, mechanisms, evidence records, and benchmark tests.
- **Normalized Ingredient Matrix:** Filterable table and card view comparing protein, fat, carbohydrates, water content, overrun %, and drainage rates across dairy, egg, plant, and hydrocolloid systems.
- **Side-by-Side Comparison Modal:** Interactive Chart.js multi-bar visualization comparing any selected set of ingredients on a strict 100g basis with automated physical chemistry trade-off commentary.
- **Physical Mechanism Explorer:** Molecular physics breakdowns of interfacial denaturation, Marangoni lipid dewetting, acid charge minimization, sugar osmotic stabilization, partial fat coalescence, and polymer yield stress.
- **Formulation / Substitution Simulator:** Real-time simulation of recipe substitutions with delta cards (-99.8% fat, -73% calories), comparative nutrient charts, and sensory texture predictions.
- **Evidence Provenance Inspector:** Full bibliographic details including DOIs, authors, journals, publication years, and direct links to dx.doi.org.
- **Genie Investigation Drawer:** Slide-over assistant allowing users to ask food chemistry questions with live SQL generation, query results, and evidence sources.

---

## 🔒 Verification & Test Suite

Run the end-to-end test verification script:
```bash
.venv/bin/python -c "
from fastapi.testclient import TestClient
from FOODMECHANICS.app.main import app

client = TestClient(app)
assert client.get('/api/status').status_code == 200
assert client.get('/api/metrics').status_code == 200
assert client.get('/api/ingredients').status_code == 200
assert client.get('/api/mechanisms').status_code == 200
assert client.get('/api/evidence').status_code == 200
print('All test assertions passed successfully!')
"
```
