"""
SIGNAL Gold Analytical Pipeline
Generates application-ready analytical tables and views:
- v_recent_developments (30-day rolling timeline with evidence attribution)
- v_model_comparisons (Normalized benchmark evaluations with methodology matching)
- v_ecosystem_relationships (Directed knowledge graph: Research -> Models -> Tools -> Capabilities)
- v_agent_trends (30-day velocity and momentum metrics for agent frameworks)

Supports both local DuckDB execution and Databricks Unity Catalog DDL export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb

logger = logging.getLogger("signal.pipelines.gold")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


SQL_V_RECENT_DEVELOPMENTS_DUCKDB = """
CREATE OR REPLACE VIEW v_recent_developments AS
SELECT 
    d.development_id,
    d.title,
    d.summary,
    d.category,
    d.date,
    d.confidence_level,
    d.primary_entity_id,
    COALESCE(m.name, t.name, c.name, d.primary_entity_id) AS primary_entity_name,
    CASE 
        WHEN m.model_id IS NOT NULL THEN 'model'
        WHEN t.tool_id IS NOT NULL THEN 'tool'
        WHEN c.capability_id IS NOT NULL THEN 'capability'
        ELSE 'general'
    END AS primary_entity_type,
    d.source_id,
    COALESCE(e.source_type, 'announcement') AS source_type,
    COALESCE(e.source_url, '') AS source_url,
    COALESCE(e.claim_text, d.summary) AS claim_text,
    d.extraction_timestamp
FROM silver_developments d
LEFT JOIN silver_evidence e ON d.source_id = e.source_id
LEFT JOIN silver_models m ON d.primary_entity_id = m.model_id
LEFT JOIN silver_tools t ON d.primary_entity_id = t.tool_id
LEFT JOIN silver_capabilities c ON d.primary_entity_id = c.capability_id
WHERE d.date >= CURRENT_DATE - INTERVAL 30 DAY
ORDER BY d.date DESC, d.development_id ASC;
"""

SQL_V_MODEL_COMPARISONS_DUCKDB = """
CREATE OR REPLACE VIEW v_model_comparisons AS
SELECT
    m.model_id,
    m.name AS model_name,
    m.provider,
    m.release_date,
    m.parameter_count,
    m.architecture,
    m.context_window,
    m.license,
    b.benchmark_id,
    b.name AS benchmark_name,
    b.task_type,
    b.evaluation_metric,
    ev.eval_id,
    ev.score,
    ev.methodology,
    ev.eval_date,
    ev.source_url,
    ROUND(ev.score - AVG(ev.score) OVER (PARTITION BY b.benchmark_id, ev.methodology), 2) AS delta_vs_methodology_avg,
    DENSE_RANK() OVER (PARTITION BY b.benchmark_id, ev.methodology ORDER BY ev.score DESC) AS rank_in_benchmark_methodology
FROM silver_models m
JOIN silver_evaluations ev ON m.model_id = ev.model_id
JOIN silver_benchmarks b ON ev.benchmark_id = b.benchmark_id
ORDER BY b.benchmark_id ASC, ev.methodology ASC, ev.score DESC;
"""

SQL_V_ECOSYSTEM_RELATIONSHIPS_DUCKDB = """
CREATE OR REPLACE VIEW v_ecosystem_relationships AS
SELECT
    r.source_entity_id,
    COALESCE(sm.name, st.name, sc.name, r.source_entity_id) AS source_entity_name,
    CASE 
        WHEN sm.model_id IS NOT NULL THEN 'model'
        WHEN st.tool_id IS NOT NULL THEN 'tool'
        WHEN sc.capability_id IS NOT NULL THEN 'capability'
        WHEN r.source_entity_id LIKE 'arxiv:%' THEN 'research_publication'
        ELSE 'external'
    END AS source_type,
    r.relationship_type,
    r.target_entity_id,
    COALESCE(tm.name, tt.name, tc.name, r.target_entity_id) AS target_entity_name,
    CASE 
        WHEN tm.model_id IS NOT NULL THEN 'model'
        WHEN tt.tool_id IS NOT NULL THEN 'tool'
        WHEN tc.capability_id IS NOT NULL THEN 'capability'
        WHEN r.target_entity_id LIKE 'arxiv:%' THEN 'research_publication'
        ELSE 'external'
    END AS target_type,
    r.strength,
    r.evidence_id,
    COALESCE(ev.source_url, '') AS source_url,
    COALESCE(ev.claim_text, '') AS claim_text,
    COALESCE(ev.confidence, 'ANNOUNCEMENT') AS confidence,
    r.extraction_timestamp
FROM silver_relationships r
LEFT JOIN silver_models sm ON r.source_entity_id = sm.model_id
LEFT JOIN silver_tools st ON r.source_entity_id = st.tool_id
LEFT JOIN silver_capabilities sc ON r.source_entity_id = sc.capability_id
LEFT JOIN silver_models tm ON r.target_entity_id = tm.model_id
LEFT JOIN silver_tools tt ON r.target_entity_id = tt.tool_id
LEFT JOIN silver_capabilities tc ON r.target_entity_id = tc.capability_id
LEFT JOIN silver_evidence ev ON r.evidence_id = ev.evidence_id;
"""

SQL_V_AGENT_TRENDS_DUCKDB = """
CREATE OR REPLACE VIEW v_agent_trends AS
WITH agent_devs AS (
    SELECT 
        primary_entity_id AS tool_id,
        COUNT(*) AS recent_developments_count,
        MAX(date) AS latest_development_date
    FROM silver_developments
    WHERE (category = 'agent_framework' OR category = 'infra_tool')
      AND date >= CURRENT_DATE - INTERVAL 30 DAY
    GROUP BY primary_entity_id
),
agent_relations AS (
    SELECT 
        t.tool_id,
        COUNT(DISTINCT r.source_entity_id || r.target_entity_id) AS total_integrations
    FROM silver_tools t
    LEFT JOIN silver_relationships r 
      ON t.tool_id = r.source_entity_id OR t.tool_id = r.target_entity_id
    GROUP BY t.tool_id
)
SELECT 
    t.tool_id,
    t.name,
    t.category,
    t.release_version,
    t.stars,
    t.ecosystem_role,
    COALESCE(d.recent_developments_count, 0) AS developments_last_30d,
    COALESCE(d.latest_development_date, CURRENT_DATE - INTERVAL 30 DAY) AS latest_development_date,
    COALESCE(r.total_integrations, 0) AS total_integrations,
    ROUND(
        (COALESCE(t.stars, 0) / 1000.0) * 0.4 + 
        (COALESCE(d.recent_developments_count, 0) * 15.0) + 
        (COALESCE(r.total_integrations, 0) * 5.0), 
        2
    ) AS momentum_score,
    CASE 
        WHEN COALESCE(d.recent_developments_count, 0) >= 3 THEN 'ACCELERATING'
        WHEN COALESCE(d.recent_developments_count, 0) >= 1 THEN 'STEADY'
        WHEN COALESCE(t.stars, 0) >= 10000 THEN 'MATURE'
        ELSE 'EMERGING'
    END AS velocity_category
FROM silver_tools t
LEFT JOIN agent_devs d ON t.tool_id = d.tool_id
LEFT JOIN agent_relations r ON t.tool_id = r.tool_id
WHERE t.category IN ('agent_orchestration', 'serving_inference', 'eval_observability')
ORDER BY momentum_score DESC;
"""


class GoldPipeline:
    """
    Gold analytical pipeline for SIGNAL.
    Creates application-ready views and materialized tables, and provides
    fast analytical query accessors.
    """

    def __init__(self, db_path: Optional[str] = None, connection: Optional[duckdb.DuckDBPyConnection] = None):
        if connection is not None:
            self.con = connection
        elif db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(db_path)
        else:
            self.con = duckdb.connect(":memory:")

        self.init_views()

    def init_views(self) -> None:
        """Executes DDL to build or refresh all Gold analytical views."""
        self.con.execute(SQL_V_RECENT_DEVELOPMENTS_DUCKDB)
        self.con.execute(SQL_V_MODEL_COMPARISONS_DUCKDB)
        self.con.execute(SQL_V_ECOSYSTEM_RELATIONSHIPS_DUCKDB)
        self.con.execute(SQL_V_AGENT_TRENDS_DUCKDB)
        logger.info("Gold analytical views created successfully.")

    def materialize_gold_tables(self) -> None:
        """Creates physical tables from the views for persistent downstream access."""
        self.con.execute("CREATE OR REPLACE TABLE gold_recent_developments AS SELECT * FROM v_recent_developments;")
        self.con.execute("CREATE OR REPLACE TABLE gold_model_comparisons AS SELECT * FROM v_model_comparisons;")
        self.con.execute("CREATE OR REPLACE TABLE gold_ecosystem_relationships AS SELECT * FROM v_ecosystem_relationships;")
        self.con.execute("CREATE OR REPLACE TABLE gold_agent_trends AS SELECT * FROM v_agent_trends;")
        logger.info("Materialized all Gold tables successfully.")

    def export_gold_parquet(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Exports Gold analytical views into Parquet files for distribution or lakehouse loading."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        views = [
            "v_recent_developments",
            "v_model_comparisons",
            "v_ecosystem_relationships",
            "v_agent_trends",
        ]
        for v in views:
            dest = out / f"{v}.parquet"
            self.con.execute(f"COPY (SELECT * FROM {v}) TO '{dest}' (FORMAT PARQUET);")
            paths[v] = str(dest)
        return paths

    # --------------------------------------------------------------------------
    # Python Analytical Accessors (for UI / DataProvider)
    # --------------------------------------------------------------------------

    def get_recent_developments(self, limit: int = 50, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches the latest AI developments from v_recent_developments."""
        query = "SELECT * FROM v_recent_developments"
        params = []
        if category:
            query += " WHERE category = ?"
            params.append(category)
        query += " LIMIT ?"
        params.append(limit)

        df = self.con.execute(query, params).fetchdf()
        # Ensure dates are serialized to ISO string for JSON serialization
        records = df.to_dict(orient="records")
        for r in records:
            if "date" in r and r["date"] is not None:
                r["date"] = str(r["date"]).split(" ")[0].split("T")[0]
            if "extraction_timestamp" in r and r["extraction_timestamp"] is not None:
                r["extraction_timestamp"] = str(r["extraction_timestamp"])
        return records

    def get_model_comparisons(
        self,
        benchmark_id: Optional[str] = None,
        methodology: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetches normalized model evaluations with methodology matching.
        Enforces methodology matching rule.
        """
        clauses = []
        params = []
        if benchmark_id:
            clauses.append("benchmark_id = ?")
            params.append(benchmark_id)
        if methodology:
            clauses.append("methodology = ?")
            params.append(methodology)
        if model_ids:
            placeholders = ",".join(["?"] * len(model_ids))
            clauses.append(f"model_id IN ({placeholders})")
            params.extend(model_ids)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM v_model_comparisons{where}"
        df = self.con.execute(query, params).fetchdf()
        records = df.to_dict(orient="records")
        for r in records:
            for k in ["release_date", "eval_date"]:
                if k in r and r[k] is not None:
                    r[k] = str(r[k]).split(" ")[0].split("T")[0]
        return records

    def get_ecosystem_graph(self, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches edges for ecosystem graph. If entity_id is provided, filters for connected nodes."""
        query = "SELECT * FROM v_ecosystem_relationships"
        params = []
        if entity_id:
            query += " WHERE source_entity_id = ? OR target_entity_id = ?"
            params.extend([entity_id, entity_id])

        df = self.con.execute(query, params).fetchdf()
        records = df.to_dict(orient="records")
        for r in records:
            if "extraction_timestamp" in r and r["extraction_timestamp"] is not None:
                r["extraction_timestamp"] = str(r["extraction_timestamp"])
        return records

    def get_agent_trends(self) -> List[Dict[str, Any]]:
        """Fetches velocity and momentum metrics for AI agent frameworks."""
        df = self.con.execute("SELECT * FROM v_agent_trends").fetchdf()
        records = df.to_dict(orient="records")
        for r in records:
            if "latest_development_date" in r and r["latest_development_date"] is not None:
                r["latest_development_date"] = str(r["latest_development_date"]).split(" ")[0].split("T")[0]
        return records

    def get_metrics(self) -> Dict[str, Any]:
        """Returns high-level summary KPIs across the SIGNAL knowledge base."""
        total_devs = self.con.execute("SELECT COUNT(*) FROM silver_developments").fetchone()[0]
        recent_devs = self.con.execute("SELECT COUNT(*) FROM v_recent_developments").fetchone()[0]
        total_models = self.con.execute("SELECT COUNT(*) FROM silver_models").fetchone()[0]
        total_tools = self.con.execute("SELECT COUNT(*) FROM silver_tools").fetchone()[0]
        total_evals = self.con.execute("SELECT COUNT(*) FROM silver_evaluations").fetchone()[0]
        total_edges = self.con.execute("SELECT COUNT(*) FROM silver_relationships").fetchone()[0]

        top_agent = self.con.execute("SELECT name, momentum_score FROM v_agent_trends LIMIT 1").fetchone()

        return {
            "total_developments": total_devs,
            "recent_30d_developments": recent_devs,
            "total_models": total_models,
            "total_tools": total_tools,
            "total_evaluations": total_evals,
            "total_graph_edges": total_edges,
            "top_trending_agent": top_agent[0] if top_agent else "N/A",
            "top_agent_momentum": top_agent[1] if top_agent else 0.0,
        }

    # --------------------------------------------------------------------------
    # Databricks Unity Catalog DDL Generator
    # --------------------------------------------------------------------------

    @staticmethod
    def generate_unity_catalog_ddl(catalog: str = "workspace", schema_prefix: str = "signal") -> str:
        """
        Generates complete Databricks Unity Catalog SQL DDL with ANSI SQL compliance,
        table comments, and column constraints.
        """
        return f"""-- =============================================================================
-- SIGNAL: AI Landscape Intelligence — Databricks Unity Catalog DDL
-- Target Catalog: {catalog}
-- Target Schemas: {catalog}.{schema_prefix}_bronze, {schema_prefix}_silver, {schema_prefix}_gold
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_prefix}_bronze;
CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_prefix}_silver;
CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_prefix}_gold;

-- -----------------------------------------------------------------------------
-- BRONZE LAYER
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_bronze.bronze_raw_feeds (
    source_id STRING NOT NULL COMMENT 'Canonical unique source identifier (arXiv ID, HF slug, GitHub repo)',
    source_type STRING NOT NULL COMMENT 'Origin platform: arxiv, huggingface, github, announcement',
    payload_json STRING NOT NULL COMMENT 'Unaltered raw JSON payload preserving all original attributes',
    ingestion_timestamp TIMESTAMP NOT NULL COMMENT 'UTC timestamp when record was appended to bronze',
    content_hash STRING NOT NULL COMMENT 'Canonical SHA-256 digest of payload for deduplication',
    CONSTRAINT pk_bronze_raw PRIMARY KEY (source_id, content_hash)
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
)
COMMENT 'Bronze capture table for raw AI landscape feeds';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_bronze.bronze_checkpoints (
    feed_name STRING PRIMARY KEY NOT NULL COMMENT 'Feed identifier stream',
    last_ingested_at TIMESTAMP NOT NULL COMMENT 'Timestamp of last ingested item',
    record_count BIGINT NOT NULL COMMENT 'Running total of ingested records',
    checkpoint_offset STRING COMMENT 'Cursor, pagination token, or git hash'
)
USING DELTA
COMMENT 'Tracks incremental ingestion high-water marks';

-- -----------------------------------------------------------------------------
-- SILVER LAYER
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_evidence (
    evidence_id STRING PRIMARY KEY NOT NULL COMMENT 'Unique evidence identifier',
    source_id STRING NOT NULL COMMENT 'Reference back to bronze source_id',
    source_type STRING NOT NULL COMMENT 'Source origin type',
    source_url STRING NOT NULL COMMENT 'Direct canonical URL of origin paper, repo, or blog',
    publication_date TIMESTAMP COMMENT 'Official release or publication timestamp',
    claim_text STRING NOT NULL COMMENT 'Verbatim or extracted claim text',
    confidence STRING NOT NULL COMMENT 'Confidence: ANNOUNCEMENT, EVALUATED, PEER_REVIEWED, HYPOTHESIS, EMPIRICAL',
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Silver normalized evidence records ensuring scientific and journalistic provenance';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_developments (
    development_id STRING PRIMARY KEY NOT NULL COMMENT 'Unique development ID',
    title STRING NOT NULL COMMENT 'Concise development headline',
    summary STRING NOT NULL COMMENT 'Detailed technical summary',
    category STRING NOT NULL COMMENT 'Category: model_release, agent_framework, benchmark, research_breakthrough, infra_tool',
    date DATE NOT NULL COMMENT 'Official announcement date',
    confidence_level STRING NOT NULL COMMENT 'Confidence tier',
    primary_entity_id STRING COMMENT 'Foreign key to primary model_id or tool_id',
    source_id STRING NOT NULL,
    raw_ref STRING NOT NULL COMMENT 'Pointer to bronze content_hash',
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized AI developments and release milestones';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_models (
    model_id STRING PRIMARY KEY NOT NULL COMMENT 'Model slug (e.g. deepseek-r1)',
    name STRING NOT NULL COMMENT 'Full model name',
    provider STRING NOT NULL COMMENT 'Organization or lab',
    release_date DATE NOT NULL,
    parameter_count BIGINT COMMENT 'Active parameter count',
    architecture STRING COMMENT 'MoE, Dense Transformer, SSM, etc.',
    context_window INT COMMENT 'Context token window length',
    license STRING NOT NULL COMMENT 'Open Weights / Commercial / Research',
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized AI models catalog';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_tools (
    tool_id STRING PRIMARY KEY NOT NULL COMMENT 'Tool slug (e.g. langgraph)',
    name STRING NOT NULL,
    category STRING NOT NULL COMMENT 'agent_orchestration, serving_inference, etc.',
    repo_url STRING,
    release_version STRING,
    stars INT,
    ecosystem_role STRING,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized developer tools and frameworks';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_benchmarks (
    benchmark_id STRING PRIMARY KEY NOT NULL COMMENT 'Benchmark slug (e.g. mmlu_pro)',
    name STRING NOT NULL,
    evaluation_metric STRING NOT NULL COMMENT 'Metric: Accuracy (%), Pass@1 (%), etc.',
    dataset_size INT,
    task_type STRING NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized AI evaluation benchmark definitions';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_evaluations (
    eval_id STRING PRIMARY KEY NOT NULL,
    model_id STRING NOT NULL,
    benchmark_id STRING NOT NULL,
    score DOUBLE NOT NULL,
    eval_date DATE NOT NULL,
    methodology STRING NOT NULL COMMENT '0-shot CoT, 5-shot direct, etc.',
    source_url STRING NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Empirical evaluation measurements with strict methodology tracking';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_capabilities (
    capability_id STRING PRIMARY KEY NOT NULL,
    name STRING NOT NULL,
    parent_capability STRING,
    description STRING,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'AI capability taxonomy and reasoning skill hierarchy';

CREATE TABLE IF NOT EXISTS {catalog}.{schema_prefix}_silver.silver_relationships (
    source_entity_id STRING NOT NULL,
    target_entity_id STRING NOT NULL,
    relationship_type STRING NOT NULL COMMENT 'implements, depends_on, evaluates_on, improves_upon, integrated_with, derived_from',
    strength DOUBLE NOT NULL,
    evidence_id STRING NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL,
    CONSTRAINT pk_silver_relationships PRIMARY KEY (source_entity_id, target_entity_id, relationship_type)
)
USING DELTA
COMMENT 'Directed knowledge graph edges connecting AI entities';

-- -----------------------------------------------------------------------------
-- GOLD LAYER ANALYTICAL VIEWS
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW {catalog}.{schema_prefix}_gold.v_recent_developments AS
SELECT 
    d.development_id,
    d.title,
    d.summary,
    d.category,
    d.date,
    d.confidence_level,
    d.primary_entity_id,
    COALESCE(m.name, t.name, c.name, d.primary_entity_id) AS primary_entity_name,
    CASE 
        WHEN m.model_id IS NOT NULL THEN 'model'
        WHEN t.tool_id IS NOT NULL THEN 'tool'
        WHEN c.capability_id IS NOT NULL THEN 'capability'
        ELSE 'general'
    END AS primary_entity_type,
    d.source_id,
    COALESCE(e.source_type, 'announcement') AS source_type,
    COALESCE(e.source_url, '') AS source_url,
    COALESCE(e.claim_text, d.summary) AS claim_text,
    d.extraction_timestamp
FROM {catalog}.{schema_prefix}_silver.silver_developments d
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_evidence e ON d.source_id = e.source_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_models m ON d.primary_entity_id = m.model_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_tools t ON d.primary_entity_id = t.tool_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_capabilities c ON d.primary_entity_id = c.capability_id
WHERE d.date >= CURRENT_DATE() - INTERVAL 30 DAYS;

CREATE OR REPLACE VIEW {catalog}.{schema_prefix}_gold.v_model_comparisons AS
SELECT
    m.model_id,
    m.name AS model_name,
    m.provider,
    m.release_date,
    m.parameter_count,
    m.architecture,
    m.context_window,
    m.license,
    b.benchmark_id,
    b.name AS benchmark_name,
    b.task_type,
    b.evaluation_metric,
    ev.eval_id,
    ev.score,
    ev.methodology,
    ev.eval_date,
    ev.source_url,
    ROUND(ev.score - AVG(ev.score) OVER (PARTITION BY b.benchmark_id, ev.methodology), 2) AS delta_vs_methodology_avg,
    DENSE_RANK() OVER (PARTITION BY b.benchmark_id, ev.methodology ORDER BY ev.score DESC) AS rank_in_benchmark_methodology
FROM {catalog}.{schema_prefix}_silver.silver_models m
JOIN {catalog}.{schema_prefix}_silver.silver_evaluations ev ON m.model_id = ev.model_id
JOIN {catalog}.{schema_prefix}_silver.silver_benchmarks b ON ev.benchmark_id = b.benchmark_id;

CREATE OR REPLACE VIEW {catalog}.{schema_prefix}_gold.v_ecosystem_relationships AS
SELECT
    r.source_entity_id,
    COALESCE(sm.name, st.name, sc.name, r.source_entity_id) AS source_entity_name,
    CASE 
        WHEN sm.model_id IS NOT NULL THEN 'model'
        WHEN st.tool_id IS NOT NULL THEN 'tool'
        WHEN sc.capability_id IS NOT NULL THEN 'capability'
        WHEN r.source_entity_id LIKE 'arxiv:%' THEN 'research_publication'
        ELSE 'external'
    END AS source_type,
    r.relationship_type,
    r.target_entity_id,
    COALESCE(tm.name, tt.name, tc.name, r.target_entity_id) AS target_entity_name,
    CASE 
        WHEN tm.model_id IS NOT NULL THEN 'model'
        WHEN tt.tool_id IS NOT NULL THEN 'tool'
        WHEN tc.capability_id IS NOT NULL THEN 'capability'
        WHEN r.target_entity_id LIKE 'arxiv:%' THEN 'research_publication'
        ELSE 'external'
    END AS target_type,
    r.strength,
    r.evidence_id,
    COALESCE(ev.source_url, '') AS source_url,
    COALESCE(ev.claim_text, '') AS claim_text,
    COALESCE(ev.confidence, 'ANNOUNCEMENT') AS confidence,
    r.extraction_timestamp
FROM {catalog}.{schema_prefix}_silver.silver_relationships r
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_models sm ON r.source_entity_id = sm.model_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_tools st ON r.source_entity_id = st.tool_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_capabilities sc ON r.source_entity_id = sc.capability_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_models tm ON r.target_entity_id = tm.model_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_tools tt ON r.target_entity_id = tt.tool_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_capabilities tc ON r.target_entity_id = tc.capability_id
LEFT JOIN {catalog}.{schema_prefix}_silver.silver_evidence ev ON r.evidence_id = ev.evidence_id;

CREATE OR REPLACE VIEW {catalog}.{schema_prefix}_gold.v_agent_trends AS
WITH agent_devs AS (
    SELECT 
        primary_entity_id AS tool_id,
        COUNT(*) AS recent_developments_count,
        MAX(date) AS latest_development_date
    FROM {catalog}.{schema_prefix}_silver.silver_developments
    WHERE (category = 'agent_framework' OR category = 'infra_tool')
      AND date >= CURRENT_DATE() - INTERVAL 30 DAYS
    GROUP BY primary_entity_id
),
agent_relations AS (
    SELECT 
        t.tool_id,
        COUNT(DISTINCT CONCAT(r.source_entity_id, r.target_entity_id)) AS total_integrations
    FROM {catalog}.{schema_prefix}_silver.silver_tools t
    LEFT JOIN {catalog}.{schema_prefix}_silver.silver_relationships r 
      ON t.tool_id = r.source_entity_id OR t.tool_id = r.target_entity_id
    GROUP BY t.tool_id
)
SELECT 
    t.tool_id,
    t.name,
    t.category,
    t.release_version,
    t.stars,
    t.ecosystem_role,
    COALESCE(d.recent_developments_count, 0) AS developments_last_30d,
    COALESCE(d.latest_development_date, CURRENT_DATE() - INTERVAL 30 DAYS) AS latest_development_date,
    COALESCE(r.total_integrations, 0) AS total_integrations,
    ROUND(
        (COALESCE(t.stars, 0) / 1000.0) * 0.4 + 
        (COALESCE(d.recent_developments_count, 0) * 15.0) + 
        (COALESCE(r.total_integrations, 0) * 5.0), 
        2
    ) AS momentum_score,
    CASE 
        WHEN COALESCE(d.recent_developments_count, 0) >= 3 THEN 'ACCELERATING'
        WHEN COALESCE(d.recent_developments_count, 0) >= 1 THEN 'STEADY'
        WHEN COALESCE(t.stars, 0) >= 10000 THEN 'MATURE'
        ELSE 'EMERGING'
    END AS velocity_category
FROM {catalog}.{schema_prefix}_silver.silver_tools t
LEFT JOIN agent_devs d ON t.tool_id = d.tool_id
LEFT JOIN agent_relations r ON t.tool_id = r.tool_id
WHERE t.category IN ('agent_orchestration', 'serving_inference', 'eval_observability');
"""
