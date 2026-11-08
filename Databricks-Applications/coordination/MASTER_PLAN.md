# Master Plan: Dual Intelligence Products (SIGNAL & FOODMECHANICS)

## Overview & Architecture
This workspace implements two independent, evidence-grounded intelligence applications powered by Databricks and modern reactive frontend architectures:
1. **SIGNAL** (AI Landscape Intelligence): An observatory tracking research, models, tools, capabilities, benchmarks, and their evolving ecosystem relationships.
2. **FOODMECHANICS** (Food Science & Culinary Intelligence): An evidence-backed food science investigation platform linking nutrition, functional ingredient properties, chemical compounds, mechanisms, and formulations.

## Execution Principles
- **Total Independence:** SIGNAL and FOODMECHANICS run on separate data models, separate ingestion pipelines, separate analytical engines, separate Genie Agents, and independent UI apps.
- **Offline / Local First:** Both products feature complete local data fixtures, local DuckDB/JSON adapters, and local interactive application servers so that everything is demonstrable immediately without waiting for compute.
- **Databricks Cloud Parity:** Dual-mode DataProviders (`LocalDataProvider` and `DatabricksDataProvider`) allow instant transition between local fixtures and Databricks SQL / Unity Catalog / Genie.
- **Provenance & Evidence First:** Every claim, benchmark score, and mechanism must preserve source identifiers, source URLs, publication timestamps, and ingestion metadata.

## Milestones & Timeline
- **Milestone 1:** Architecture, Directory Setup & Integration Contracts (Current)
- **Milestone 2:** Parallel Agent Launch (S1-S4 for SIGNAL, F1-F4 for FOODMECHANICS)
- **Milestone 3:** Local End-to-End Pipeline & Usable Applications with Local Fixtures
- **Milestone 4:** Databricks Unity Catalog Schemas, Bronze/Silver/Gold DDLs & Genie Agent Configuration
- **Milestone 5:** Live Demonstration & Verification Runs
