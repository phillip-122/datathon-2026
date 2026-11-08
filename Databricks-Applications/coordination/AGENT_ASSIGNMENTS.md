# Agent Assignments & Scope Ownership

## Architecture & Coordination
- **Lead / Orchestrator:** Oversees integration contracts, shared standards, Databricks connectivity, and milestone progress.

---

## SIGNAL Agent Team (AI Landscape Intelligence)

### Agent S1 — Domain Research & Data Acquisition
- **Role:** Build the AI landscape data acquisition layer.
- **Sources:** arXiv API (CS.AI/CL), Hugging Face model & dataset APIs, GitHub release/repo metadata, official release announcements.
- **Deliverables:**
  - `SIGNAL/ingestion/sources/` (arXiv, Hugging Face, GitHub collectors)
  - `SIGNAL/data/raw/` (JSON source payloads preserving metadata & timestamps)
  - `SIGNAL/data/fixtures/` (Realistic local sample datasets for offline use)
  - `SIGNAL/docs/SOURCES.md` & `SIGNAL/docs/DATA_DICTIONARY.md`

### Agent S2 — Data Engineering & Pipeline
- **Role:** Build normalized knowledge model and incremental processing pipeline.
- **Deliverables:**
  - `SIGNAL/pipelines/` (Bronze raw capture, Silver normalization/deduplication, Gold analytical aggregates)
  - `SIGNAL/analytics/` (Gold views: recent developments, capability trends, model comparisons, ecosystem graph)
  - `SIGNAL/docs/SCHEMA.md` & `SIGNAL/tests/`

### Agent S3 — SIGNAL Interactive Application
- **Role:** Implement the user-facing web application.
- **Deliverables:**
  - `SIGNAL/app/` (FastAPI / Node + Modern Web UI: Timeline, Ecosystem Graph, Model Comparison, Genie Investigation interface)
  - `SIGNAL/app/data_adapter.py` (Local & Databricks DataProvider implementations)
  - `SIGNAL/README.md` & local run scripts

### Agent S4 — Genie Agent Configuration & Evaluation
- **Role:** Design, validate, and evaluate the SIGNAL Genie Agent.
- **Deliverables:**
  - `SIGNAL/genie/GENIE_GUIDE.md` (System instructions, semantic terms, table descriptions)
  - `SIGNAL/genie/EXAMPLE_QUERIES.sql` (Verified complex analytical queries)
  - `SIGNAL/evaluation/QUESTIONS.json` & `SIGNAL/evaluation/EXPECTED_RESULTS.md`

---

## FOODMECHANICS Agent Team (Food Intelligence)

### Agent F1 — Food Science Research & Data Acquisition
- **Role:** Identify and implement reliable food information sources.
- **Sources:** USDA FoodData Central (FDC API), Open Food Facts, PubChem, scientific literature metadata (Crossref/OpenAlex), curated culinary chemistry experiments.
- **Deliverables:**
  - `FOODMECHANICS/ingestion/sources/` (USDA, Open Food Facts, Scientific collectors)
  - `FOODMECHANICS/data/raw/` (Raw JSON payloads with full metadata)
  - `FOODMECHANICS/data/fixtures/` (Curated offline fixtures: egg white, milk, emulsions, foams, hydrocolloids)
  - `FOODMECHANICS/docs/SOURCES.md` & `FOODMECHANICS/docs/DATA_DICTIONARY.md`

### Agent F2 — Data Engineering & Pipeline
- **Role:** Build the food knowledge model and incremental pipelines.
- **Deliverables:**
  - `FOODMECHANICS/pipelines/` (Bronze raw capture, Silver normalization with 100g basis, Gold mechanism/evidence views)
  - `FOODMECHANICS/analytics/` (Gold views: ingredient comparisons, functional property mappings, formulation trade-offs)
  - `FOODMECHANICS/docs/SCHEMA.md` & `FOODMECHANICS/tests/`

### Agent F3 — FOODMECHANICS Interactive Application
- **Role:** Build the user-facing food science application.
- **Deliverables:**
  - `FOODMECHANICS/app/` (FastAPI / Node + Modern Web UI: Ingredient Explorer, Nutritional Comparison, Mechanism Map, Substitution Simulator, Genie interface)
  - `FOODMECHANICS/app/data_adapter.py` (Local & Databricks DataProvider implementations)
  - `FOODMECHANICS/README.md` & local run scripts

### Agent F4 — Genie Agent Configuration & Evaluation
- **Role:** Design, validate, and evaluate the FOODMECHANICS Genie Agent.
- **Deliverables:**
  - `FOODMECHANICS/genie/GENIE_GUIDE.md` (Domain instructions, serving basis rules, scientific claim guidelines)
  - `FOODMECHANICS/genie/EXAMPLE_QUERIES.sql` (Verified food chemistry & nutrition queries)
  - `FOODMECHANICS/evaluation/QUESTIONS.json` & `FOODMECHANICS/evaluation/EXPECTED_RESULTS.md`
