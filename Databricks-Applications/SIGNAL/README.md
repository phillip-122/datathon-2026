# SIGNAL — AI Landscape Observatory

> **An evolving intelligence system tracking frontier AI models, research breakthroughs, developer tools, benchmark evaluations, and ecosystem relationships.**

---

## 1. Overview & Vision

**SIGNAL** is an AI landscape observatory. It continuously collects developments across multiple technical domains, extracts structured knowledge, connects related technologies into an evolving graph, and maintains a verifiable historical representation of the AI ecosystem.

Unlike generic news aggregators or ungrounded chatbots, SIGNAL's distinguishing capability is **understanding and visualizing relationships between developments**—connecting research papers to foundation models, models to tools and agent frameworks, and claims to verifiable source evidence.

### What It Tracks

| Domain | Tracked Information |
|---|---|
| **Models** | Releases, versions, architectures, context windows, open weights, serving pricing |
| **Research** | Preprints, techniques, mathematical derivations, empirical discoveries |
| **Benchmarks** | Standardized evaluations (SWE-bench Verified, GPQA Diamond, MATH-500, GAIA), methodology rigor |
| **Tools & Protocols** | Multi-agent frameworks (LangGraph, CrewAI, AutoGen, Smolagents), open protocols (MCP), serving engines (vLLM) |
| **Relationships** | Derivations, integrations, evaluations, competitive dynamics |
| **Evidence** | Source URLs, publication dates, ingestion timestamps, confidence classifications |

---

## 2. Architecture & Data Contracts

SIGNAL is decoupled from physical storage through the abstract `DataProvider` interface defined in `coordination/contracts/provider.py`.

```
                    ┌──────────────────────────────────────────────┐
                    │            SIGNAL Web Application            │
                    │   (FastAPI Backend + Modern HTML5/CSS/JS)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                           ┌───────────────▼───────────────┐
                           │      DataProvider Factory     │
                           │     (SIGNAL/app/data_adapter) │
                           └───────┬───────────────┬───────┘
                                   │               │
                 [Databricks Online]               [Offline Fallback]
                                   │               │
                 ┌─────────────────▼─────┐   ┌─────▼─────────────────────────┐
                 │ Databricks Gold SQL   │   │ SignalLocalDataProvider       │
                 │ - v_recent_developments│   │ - Curated verified fixtures   │
                 │ - v_model_comparisons │   │ - In-memory DuckDB tables     │
                 │ - v_ecosystem_graph   │   │ - Sub-millisecond text search │
                 └───────────────────────┘   └───────────────────────────────┘
```

### Evidence Provenance Schema
Every development, model, tool, and benchmark evaluation adheres strictly to the `EvidenceRecord` standard:
- `source_id`: Canonical unique identifier (e.g. `arxiv:2501.12948`, `github:modelcontextprotocol/specification`)
- `source_type`: `arxiv`, `huggingface`, `github`, `official_announcement`
- `source_url`: Verifiable web URL
- `evidence_confidence`: `ANNOUNCEMENT`, `EVALUATED`, `PEER_REVIEWED`, `EMPIRICAL`
- `publication_date`, `ingestion_timestamp`, `extraction_timestamp`
- `raw_payload_ref`: Location or hash of bronze raw payload

---

## 3. Web Application Features

The user interface in `SIGNAL/app/static/` is designed for high-density, interactive technical analysis:

### 1. KPI Summary Banner
- **Active Models**: Frontier proprietary and open-weights models under tracking.
- **Tracked Tools & Protocols**: Open standards (MCP) and agent frameworks.
- **Benchmark Evaluations**: Verified score entries across standardized test suites.
- **30-Day Ecosystem Velocity**: Percentage change in ecosystem activity and new releases.
- **Evidence Verification Ratio**: Visual breakdown of `EVALUATED` and `EMPIRICAL` vs `ANNOUNCEMENT`.

### 2. Chronological 30-Day Timeline
- Filter by technical theme: **All**, **Models**, **Tools**, **Research**, **Benchmarks**, **Infrastructure**.
- Filter by evidence confidence level.
- Live impact score indicators (1 to 10 scale).
- Direct access to **Inspect Evidence** for any card.

### 3. Frontier Model Benchmark Matrix
- Multi-model comparison across **SWE-bench Verified**, **GPQA Diamond**, **MATH-500**, **GAIA**, and **HumanEval**.
- Visual score bars, context window specifications, and pricing per 1M tokens.
- **Comparability Rigor Rule**: Callout enforcing identical evaluation splits and no-human-intervention conditions.

### 4. Ecosystem Relationship Map
- Visual network graph connecting:
  $$\text{Research Papers} \longrightarrow \text{Foundation Models} \longrightarrow \text{Tools \& Protocols} \longrightarrow \text{Autonomous Agents}$$
- Color-coded node categories with interactive pan, zoom, and node search.
- Click any node to isolate connected sub-graphs, view relationship types, and inspect evidence.

### 5. Databricks Genie Investigation Drawer
- Natural language query interface over Databricks Unity Catalog schemas.
- Pre-configured investigation queries:
  - *"What has changed in the AI agent ecosystem over the last 30 days?"*
  - *"Which developments are announcements versus independently evaluated results?"*
  - *"Compare DeepSeek-R1 and Claude 3.5 Sonnet on coding and reasoning."*
  - *"Which tools depend on or integrate with Model Context Protocol (MCP)?"*
- Displays synthesis answer, executed Databricks SQL dialect with copy button, query results table, and verified source provenance.

### 6. Evidence Inspector Modal
- Complete provenance card displaying canonical source URL, timestamps, verification status, and claim excerpts.

---

## 4. API Endpoints Reference

The FastAPI server exposes clean REST endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the interactive web interface |
| `/api/status` | `GET` | System health, provider type, Databricks connectivity status |
| `/api/metrics` | `GET` | Observatory KPI summary statistics |
| `/api/developments/recent` | `GET` | Filterable recent developments timeline (`limit`, `category`) |
| `/api/models/comparison` | `GET` | Benchmark comparison matrix across models |
| `/api/ecosystem/graph` | `GET` | Graph nodes and edges for ecosystem visualization (`category`, `search`) |
| `/api/search` | `GET` | Multi-faceted entity and claim search (`q`, `category`, `confidence`) |
| `/api/genie/query` | `POST` | Natural language investigation queries with SQL and evidence |
| `/api/evidence/{record_id}` | `GET` | Source provenance inspection for a record |
| `/api/entities/{entity_id}` | `GET` | Complete entity detail with adjacent relationships |

---

## 5. Running the Application

### Option A: Standalone Launcher (Recommended)
From the project root:
```bash
./SIGNAL/app/run.sh
```

### Option B: Direct Python / Uvicorn
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Launch FastAPI on port 8000
python -m uvicorn SIGNAL.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
```
http://localhost:8000
```

---

## 6. Databricks Connectivity & Fallback Behavior

SIGNAL operates under a zero-blockers philosophy:
1. When `.env` contains valid Databricks credentials (`DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`), the factory connects to the Databricks workspace.
2. If the `workspace.signal_gold` schema tables are populated by data engineering pipelines, it executes live queries directly on Databricks serverless SQL.
3. If Databricks is offline, credentials are not provided, or tables are undergoing initial materialization, the system **automatically and seamlessly serves verified offline fixtures** through an in-memory DuckDB engine.
4. The live UI header displays an active status badge indicating whether data is served from **Databricks Gold** or **Verified Local Fixtures**.
