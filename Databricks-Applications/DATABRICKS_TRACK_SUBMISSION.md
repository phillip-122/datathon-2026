# TAMU Datathon 2026: Databricks Challenge Track Submission

**Challenge Benchmark:** *Build a project that creatively uses Databricks for data engineering, analytics, machine learning or AI, either on its own or as part of your solution to another TAMU Datathon challenge. Judges look at creativity, technical implementation, and how meaningfully Databricks improves your solution.*

---

## Executive Summary

Modern AI engineering and scientific formulation are plagued by fragmented, unstructured, and unverified data. Rather than building another toy dashboard or a generic chatbot wrapper, we engineered **two production-grade intelligence platforms powered natively by a Databricks Unity Catalog Medallion Lakehouse**:

1. **SIGNAL (AI Landscape Observatory)** – A dynamic AI intelligence terminal tracking 26 frontier reasoning models, 195 Monte Carlo benchmark simulation sweeps, protocol graphs (MCP, AGY, LangGraph), and Pareto frontier efficiency (reasoning performance vs. inference cost). Features an interactive **Research Ingestion & Impact Radius Tracer ("What-If Release")**, an **Enterprise Architecture Advisor & LLM Matchmaker**, and a **Head-to-Head Battleground**.
2. **FOODMECHANICS (Food Science Colloid Studio)** – An evidence-based molecular formulation workbench simulating 5 complex colloidal architectures (foams, emulsions, viscoelastic doughs, gels, spherified beads) across 33 USDA Foundation ingredients, powered by 120 Databricks Spark SQL multi-concentration parameter sweeps. Features a dynamic **AI & Serverless Inverse Formulation Solver** plotting **24-hour Stokes' Law drainage kinetics curves** and an **Any-to-Any Ingredient Substitution Studio**.

Both platforms run against a live **Databricks Serverless SQL Warehouse** and feature interactive **Lakehouse SQL Studios** directly in their web interfaces, giving judges and domain practitioners transparent, sub-second execution transparency.

---

## Live System Architecture

```
                                  DATABRICKS LAKEHOUSE PLATFORM
                                 (Serverless SQL Warehouse Active)
                                               │
                ┌──────────────────────────────┴──────────────────────────────┐
                ▼                                                             ▼
      FOOD SCIENCE DOMAIN                                            AI ECOSYSTEM DOMAIN
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│  BRONZE LAYER (Immutable)    │                              │  BRONZE LAYER (Immutable)    │
│  workspace.foodmechanics_raw │                              │  workspace.signal_raw        │
│  - USDA FoodData Central     │                              │  - arXiv Preprints           │
│  - Food Hydrocolloids DOIs   │                              │  - Hugging Face / GitHub     │
│  - Raw Formulation Batches   │                              │  - Provider Benchmark Specs  │
└──────────────┬───────────────┘                              └──────────────┬───────────────┘
               ▼                                                             ▼
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│  SILVER LAYER (Conformed)    │                              │  SILVER LAYER (Conformed)    │
│  - silver_ingredients (33)   │                              │  - silver_models (26)        │
│  - silver_nutritional (33)   │                              │  - silver_benchmarks (8)     │
│    (Strict 100g USDA norm)   │                              │  - silver_evaluations (104)  │
│  - silver_functional_props   │                              │  - silver_tools (9)          │
│  - silver_food_mechanisms    │                              │  - silver_relationships (8)  │
│  - silver_scientific_evidence│                              │  - silver_developments (6)   │
└──────────────┬───────────────┘                              └──────────────┬───────────────┘
               ▼                                                             ▼
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│  GOLD LAYER (Analytics & AI) │                              │  GOLD LAYER (Analytics & AI) │
│  - gold_formulation_sims     │                              │  - gold_synthetic_benchmarks │
│    (120 Spark SQL sweeps)    │                              │    (195 Monte Carlo sweeps)  │
│  - v_ingredient_comparisons  │                              │  - v_model_comparisons       │
│  - Stokes' Law Rheology View │                              │  - Pareto Efficiency Frontier│
└──────────────┬───────────────┘                              └──────────────┬───────────────┘
               │                                                             │
               └──────────────────────────────┬──────────────────────────────┘
                                              ▼
                                 DUAL-ENGINE DATA ADAPTER
                           (Databricks REST API + DuckDB Fallback)
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               ▼                                                             ▼
     FOODMECHANICS APP                                              SIGNAL APP
     Port: 8080                                                     Port: 8000
     - ⚡ Dynamic Inverse Formulation Solver                        - ⚡ Research Ingestion & Impact Tracer
     - 📈 24h Stokes' Law Drainage Curves                           - 🎯 Enterprise Architecture Advisor
     - 🔄 Any-to-Any Substitution Studio                            - ⚔️ Head-to-Head Arena
     - 🧪 Multi-Component Colloid Scratchpad                        - 🌐 Ecosystem Relationship Radar
     - 🎛️ 5 Core Benchmark Architectures                            - 📊 Pareto Efficiency Frontier
     - 🏛️ Embedded Databricks SQL Studio                            - 🏛️ Embedded Databricks SQL Studio
```

---

## How Meaningfully Databricks Improves the Solution

### 1. High-Scale Synthetic Simulation Sweeps (Spark SQL)
In both food formulation and AI evaluation, real-world trials are costly and high-latency. We leveraged Databricks compute to generate high-dimensional synthetic analytical spaces:
- **Food Formulation Sweeps (`workspace.foodmechanics_gold.gold_formulation_simulations` - 120 rows):** Evaluated multi-parameter combinations of continuous phase water, fat, protein, and xanthan hydrocolloid percentages across shear rates ($10^{-1}$ to $10^{3} \text{ s}^{-1}$) to model Stokes' gravitational drainage velocity:
  $$v(t) = \frac{2 r^2 (\rho_p - \rho_f) g}{9 \eta(t)}$$
- **AI Reasoning Sweeps (`workspace.signal_gold.gold_synthetic_benchmark_runs` - 195 rows):** Conducted Monte Carlo parameter sweeps crossing reasoning effort tiers (`low`, `medium`, `high`, `extreme`), temperatures ($0.0$ to $1.0$), and sampling budgets to chart token variance and Pareto frontiers.

### 2. Dynamic Inverse Formulation Solver (Interactive AI on Databricks)
Instead of static canned recipes, any user or judge can specify arbitrary goals:
- Set target limits for Max Fat (g/100g), Min Protein (g/100g), Target Overrun (%), Min Stability (Hours), and Dietary Exclusions (Vegan, Gluten-Free, Dairy-Free, Low-Sugar).
- Databricks Serverless queries the 120-point gold parameter sweep, solves the multi-component mass balance across 33 USDA Foundation ingredients, and renders a **dynamic 24-hour Stokes' Law drainage curve** alongside the molecular physics explanation.

### 3. Dynamic "What-If" Release Ingestion & Graph Ripple Analytics
- Users can simulate any new model release (e.g. *DeepSeek-R2 Preview*, *Claude 3.7 Sonnet*, or custom specs).
- Databricks recalibrates the **Pareto Efficiency Frontier** in real-time, identifies **Economically Displaced Incumbent Models** (e.g. models costing 14.6x more for equivalent reasoning), and traverses Unity Catalog graph edges to compute downstream ripple effects on runtime engines (vLLM kernels, SGLang DualPipe, Ollama edge quantizations).

### 4. Governed Medallion Architecture (Unity Catalog)
- **Zero Loss of Provenance:** Every single datum in Silver and Gold is linked back to a persistent `EvidenceRecord` containing DOI citations, USDA FDC IDs, or original paper URLs.
- **Strict Data Contracts:** All food ingredients are normalized to a strict **100g USDA mass basis** to eliminate misleading arbitrary serving size skew.

### 5. Interactive Lakehouse & SQL Studio Built for Judges
Rather than hiding Databricks behind abstract buzzwords or generic chatbots, we surfaced a **Databricks Lakehouse & Governed SQL Studio modal** inside both frontends:
- Judges can click **"⚡ Databricks Lakehouse & SQL Studio"** in the top navigation bar.
- Pre-populated queries allow 1-click execution against live Unity Catalog tables.
- Displays live serverless latency, Serverless Query ID, executed SQL, and tabular result schemas.

---

## Verifying the Solution Live (Judge Walkthrough)

### Step 1: Ensure Local Apps are Running
Both applications run concurrently on the host machine:
- **FOODMECHANICS (Food Science Studio):** [http://localhost:8080](http://localhost:8080)
- **SIGNAL (AI Landscape Observatory):** [http://localhost:8000](http://localhost:8000)

To check health via curl:
```bash
curl http://localhost:8080/api/status
curl http://localhost:8000/api/status
```
Both return `databricks_connected: true` and `engine_mode: "Databricks Serverless SQL"`.

### Step 2: Test FOODMECHANICS (Port 8080)
1. **Dynamic Inverse Formulation Solver:**
   - Under Tab 1, click any 1-click preset (e.g., *🍨 Zero-Fat Vegan Meringue* or *🍞 Gluten-Free Viscoelastic Bread*) or adjust the sliders for Max Fat, Min Protein, Target Overrun, and Stability.
   - Click **"⚡ Solve Formulation on Databricks"**.
   - Observe the solved mass recipe table, the **dynamic 24-hour Stokes' Law drainage curve** (green curve staying $< 6\%$ syneresis vs red control collapsing to $100\%$), the 100g USDA macros, and the sub-second Databricks execution latency.
2. **Any-to-Any Substitution Studio:**
   - Switch to Tab 2 (**"🔄 Any-to-Any Substitution Studio"**).
   - Select *Classic Chantilly Cream*, replace *Heavy Whipping Cream (36% Fat)* with *Chickpea Cooking Liquid (Aquafaba)*, and add *Xanthan Gum* as a compensator.
   - Click **"Run Substitution Analysis"** to see instant $\Delta$ calories, $\Delta$ fat, $\Delta$ overrun, and the food chemistry risk and compensation analysis backed by DOI literature.
3. **Open the Databricks SQL Studio:**
   - Click **"⚡ Databricks Lakehouse & SQL Studio"** in the header to run live DBSQL queries against `workspace.foodmechanics_gold` and `workspace.foodmechanics_silver`.

### Step 3: Test SIGNAL AI Observatory (Port 8000)
1. **Live Research Ingestion & Ripple Tracer ("What-If Release"):**
   - Under Tab 1, click any preset (e.g., *⚡ DeepSeek-R2 Preview* or *⚡ Claude 3.7 Sonnet*).
   - Click **"⚡ Run Databricks Ingestion & Impact Analysis"**.
   - Observe the **Economically Displaced Incumbent Models** table (demonstrating up to 14.6x cost savings), the Pareto Frontier recalculation, and the traversed inference runtime kernel impacts (vLLM, SGLang, Ollama).
2. **Enterprise Architecture Advisor & LLM Matchmaker:**
   - Switch to Tab 2 (**"🎯 Enterprise Architecture Advisor"**).
   - Set your monthly budget ($500), monthly token volume (50M), and reasoning thresholds.
   - Click **"Match Optimal Architecture"** to receive a dual-tier deployment cascade, runtime memory footprint, and **Projected Monthly Savings vs GPT-4o**.
3. **Head-to-Head Battleground:**
   - Switch to Tab 3 (**"⚔️ Head-to-Head Battleground"**).
   - Select Challenger A (e.g. *DeepSeek-R1-Distill-Qwen-14B*) vs Benchmark B (e.g. *OpenAI o1*).
   - Click **"Compare Models"** to view dimensional deltas across SWE-bench, MATH-500, GPQA, pricing, and context size.

---

## Automated Test Verification

Our solution enforces rigorous test-driven engineering across both domains. Run the full pytest suite in the active environment:

```bash
source .venv/bin/activate
pytest SIGNAL/tests/ FOODMECHANICS/tests/
```

**Results:**
```
======================= 47 passed, 17 warnings in 14.94s =======================
```
- `SIGNAL/tests/test_acquisition.py`: 5 passed (Relational integrity, evidence schema, collector contracts)
- `SIGNAL/tests/test_signal_pipeline.py`: 16 passed (Bronze hashing, Silver conformed tables, Gold views, Ingest simulation, Advisor match, Head-to-head)
- `FOODMECHANICS/tests/test_collectors_and_fixtures.py`: 5 passed (USDA envelope normalization, evidence links)
- `FOODMECHANICS/tests/test_food_pipeline.py`: 21 passed (100g unit contracts, mass conservation, substitution physics, inverse optimizer solver, recipes catalog)

---

## Unity Catalog Schema Inventory

### 1. AI Intelligence Observatory (`workspace.signal_*`)
| Table / View | Layer | Rows | Purpose |
|---|---|---|---|
| `workspace.signal_bronze.bronze_signal_raw` | Bronze | 16 | Content-hash deduplicated raw acquisition payloads |
| `workspace.signal_silver.silver_models` | Silver | 26 | Conformed AI models with parameter counts and licenses |
| `workspace.signal_silver.silver_benchmarks` | Silver | 8 | Evaluated benchmark definitions and scoring criteria |
| `workspace.signal_silver.silver_evaluations` | Silver | 104 | Rigorous benchmark results linked to evidence DOIs |
| `workspace.signal_silver.silver_tools` | Silver | 9 | Protocols, runtime engines, and agent harnesses |
| `workspace.signal_silver.silver_relationships` | Silver | 8 | Ecosystem graph edges connecting models and tools |
| `workspace.signal_gold.gold_synthetic_benchmark_runs`| Gold | 195 | Databricks Spark Monte Carlo reasoning simulations |
| `workspace.signal_gold.v_model_comparisons` | Gold View | Active | Flattened analytical view with pricing and scores |

### 2. Food Science Rheology Studio (`workspace.foodmechanics_*`)
| Table / View | Layer | Rows | Purpose |
|---|---|---|---|
| `workspace.foodmechanics_bronze.bronze_food_raw` | Bronze | 33 | Raw USDA FoodData Central and literature extractions |
| `workspace.foodmechanics_silver.silver_ingredients` | Silver | 33 | Standardized food ingredient catalog |
| `workspace.foodmechanics_silver.silver_nutritional_profiles` | Silver | 33 | Normalized strictly to 100g edible portion basis |
| `workspace.foodmechanics_silver.silver_functional_properties`| Silver | 14 | Overrun %, drainage $T_{50}$, and continuous yield stress |
| `workspace.foodmechanics_silver.silver_food_mechanisms` | Silver | 6 | Thermodynamic equations and kinetic governing rules |
| `workspace.foodmechanics_gold.gold_formulation_simulations` | Gold | 120 | Databricks Spark SQL multi-concentration sweeps |
| `workspace.foodmechanics_gold.v_ingredient_comparisons` | Gold View | Active | 100g basis ingredient physical and nutritional comparison |

---

## Conclusion

By grounding complex scientific and AI domains into **governed Unity Catalog schemas, Spark parameter sweeps, dynamic multi-objective solvers, and sub-second Serverless SQL interfaces**, this submission demonstrates the transformative capability of Databricks in converting fragmented data into production-ready decision intelligence.
