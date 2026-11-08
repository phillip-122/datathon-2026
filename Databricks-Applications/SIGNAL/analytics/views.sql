-- =============================================================================
-- SIGNAL: AI Landscape Intelligence — Analytical SQL Views & DDL Architecture
-- 
-- Supported Dialects:
-- 1. Databricks Unity Catalog (workspace.signal_bronze / silver / gold)
-- 2. Local DuckDB (signal.duckdb)
--
-- Medallion Structure:
-- - Bronze: Raw feeds capture with content hashing & deduplication
-- - Silver: Normalized models, tools, developments, benchmarks, evaluations, capabilities, relationships, evidence
-- - Gold: Application-ready analytical views (v_recent_developments, v_model_comparisons, v_ecosystem_relationships, v_agent_trends)
-- =============================================================================


-- =============================================================================
-- SECTION 1: DATABRICKS UNITY CATALOG DDL (workspace.signal_*)
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS workspace.signal_bronze
COMMENT 'Bronze raw ingestion layer for AI landscape feeds';

CREATE SCHEMA IF NOT EXISTS workspace.signal_silver
COMMENT 'Silver normalized knowledge model for AI entities, evaluations, and relationships';

CREATE SCHEMA IF NOT EXISTS workspace.signal_gold
COMMENT 'Gold analytical views for application rendering and Genie natural language querying';


-- -----------------------------------------------------------------------------
-- 1.1 BRONZE TABLES (Unity Catalog)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS workspace.signal_bronze.bronze_raw_feeds (
    source_id STRING NOT NULL COMMENT 'Canonical unique source identifier (e.g. arxiv:2412.01234, gh:langchain-ai/langgraph)',
    source_type STRING NOT NULL COMMENT 'Origin platform: arxiv, huggingface, github, announcement',
    payload_json STRING NOT NULL COMMENT 'Unaltered raw JSON payload preserving all original attributes',
    ingestion_timestamp TIMESTAMP NOT NULL COMMENT 'UTC timestamp when record was appended to bronze',
    content_hash STRING NOT NULL COMMENT 'Canonical SHA-256 digest of payload for idempotent deduplication',
    CONSTRAINT pk_bronze_raw PRIMARY KEY (source_id, content_hash)
)
USING DELTA
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
)
COMMENT 'Bronze capture table for raw AI landscape feeds';

CREATE TABLE IF NOT EXISTS workspace.signal_bronze.bronze_checkpoints (
    feed_name STRING PRIMARY KEY NOT NULL COMMENT 'Feed identifier stream (e.g. arxiv_cs_ai, github_releases)',
    last_ingested_at TIMESTAMP NOT NULL COMMENT 'Timestamp of last ingested item',
    record_count BIGINT NOT NULL COMMENT 'Running total of ingested records',
    checkpoint_offset STRING COMMENT 'Cursor, pagination token, or git hash'
)
USING DELTA
COMMENT 'Tracks incremental ingestion high-water marks for pollers';


-- -----------------------------------------------------------------------------
-- 1.2 SILVER TABLES (Unity Catalog)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_evidence (
    evidence_id STRING PRIMARY KEY NOT NULL COMMENT 'Unique evidence reference identifier',
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

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_developments (
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

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_models (
    model_id STRING PRIMARY KEY NOT NULL COMMENT 'Model slug (e.g. deepseek-r1, claude-3-5-sonnet)',
    name STRING NOT NULL COMMENT 'Full model name',
    provider STRING NOT NULL COMMENT 'Organization or lab (Anthropic, DeepSeek, Meta, OpenAI, Google)',
    release_date DATE NOT NULL,
    parameter_count BIGINT COMMENT 'Active parameter count',
    architecture STRING COMMENT 'MoE, Dense Transformer, SSM, etc.',
    context_window INT COMMENT 'Context token window length',
    license STRING NOT NULL COMMENT 'Open Weights / Commercial / Research',
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized AI models catalog';

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_tools (
    tool_id STRING PRIMARY KEY NOT NULL COMMENT 'Tool slug (e.g. langgraph, vllm, autogen)',
    name STRING NOT NULL,
    category STRING NOT NULL COMMENT 'agent_orchestration, serving_inference, eval_observability, prompt_engineering',
    repo_url STRING,
    release_version STRING,
    stars INT,
    ecosystem_role STRING,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized developer tools and frameworks';

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_benchmarks (
    benchmark_id STRING PRIMARY KEY NOT NULL COMMENT 'Benchmark slug (e.g. mmlu_pro, swe_bench_verified)',
    name STRING NOT NULL,
    evaluation_metric STRING NOT NULL COMMENT 'Metric: Accuracy (%), Pass@1 (%), etc.',
    dataset_size INT,
    task_type STRING NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Normalized AI evaluation benchmark definitions';

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_evaluations (
    eval_id STRING PRIMARY KEY NOT NULL,
    model_id STRING NOT NULL,
    benchmark_id STRING NOT NULL,
    score DOUBLE NOT NULL,
    eval_date DATE NOT NULL,
    methodology STRING NOT NULL COMMENT '0-shot CoT, 5-shot direct, Scaffolded Agent (Pass@1)',
    source_url STRING NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Empirical evaluation measurements with strict methodology tracking';

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_capabilities (
    capability_id STRING PRIMARY KEY NOT NULL,
    name STRING NOT NULL,
    parent_capability STRING,
    description STRING,
    extraction_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'AI capability taxonomy and reasoning skill hierarchy';

CREATE TABLE IF NOT EXISTS workspace.signal_silver.silver_relationships (
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
-- 1.3 GOLD ANALYTICAL VIEWS (Unity Catalog)
-- -----------------------------------------------------------------------------

-- View 1: Recent Developments (30-day timeline with evidence attribution)
CREATE OR REPLACE VIEW workspace.signal_gold.v_recent_developments AS
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
FROM workspace.signal_silver.silver_developments d
LEFT JOIN workspace.signal_silver.silver_evidence e ON d.source_id = e.source_id
LEFT JOIN workspace.signal_silver.silver_models m ON d.primary_entity_id = m.model_id
LEFT JOIN workspace.signal_silver.silver_tools t ON d.primary_entity_id = t.tool_id
LEFT JOIN workspace.signal_silver.silver_capabilities c ON d.primary_entity_id = c.capability_id
WHERE d.date >= CURRENT_DATE() - INTERVAL 30 DAYS;

-- View 2: Model Comparisons (Normalized benchmark evaluations with methodology matching)
CREATE OR REPLACE VIEW workspace.signal_gold.v_model_comparisons AS
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
FROM workspace.signal_silver.silver_models m
JOIN workspace.signal_silver.silver_evaluations ev ON m.model_id = ev.model_id
JOIN workspace.signal_silver.silver_benchmarks b ON ev.benchmark_id = b.benchmark_id;

-- View 3: Ecosystem Relationships (Connected knowledge graph)
CREATE OR REPLACE VIEW workspace.signal_gold.v_ecosystem_relationships AS
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
FROM workspace.signal_silver.silver_relationships r
LEFT JOIN workspace.signal_silver.silver_models sm ON r.source_entity_id = sm.model_id
LEFT JOIN workspace.signal_silver.silver_tools st ON r.source_entity_id = st.tool_id
LEFT JOIN workspace.signal_silver.silver_capabilities sc ON r.source_entity_id = sc.capability_id
LEFT JOIN workspace.signal_silver.silver_models tm ON r.target_entity_id = tm.model_id
LEFT JOIN workspace.signal_silver.silver_tools tt ON r.target_entity_id = tt.tool_id
LEFT JOIN workspace.signal_silver.silver_capabilities tc ON r.target_entity_id = tc.capability_id
LEFT JOIN workspace.signal_silver.silver_evidence ev ON r.evidence_id = ev.evidence_id;

-- View 4: Agent Trends (30-day velocity and momentum metrics)
CREATE OR REPLACE VIEW workspace.signal_gold.v_agent_trends AS
WITH agent_devs AS (
    SELECT 
        primary_entity_id AS tool_id,
        COUNT(*) AS recent_developments_count,
        MAX(date) AS latest_development_date
    FROM workspace.signal_silver.silver_developments
    WHERE (category = 'agent_framework' OR category = 'infra_tool')
      AND date >= CURRENT_DATE() - INTERVAL 30 DAYS
    GROUP BY primary_entity_id
),
agent_relations AS (
    SELECT 
        t.tool_id,
        COUNT(DISTINCT CONCAT(r.source_entity_id, r.target_entity_id)) AS total_integrations
    FROM workspace.signal_silver.silver_tools t
    LEFT JOIN workspace.signal_silver.silver_relationships r 
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
FROM workspace.signal_silver.silver_tools t
LEFT JOIN agent_devs d ON t.tool_id = d.tool_id
LEFT JOIN agent_relations r ON t.tool_id = r.tool_id
WHERE t.category IN ('agent_orchestration', 'serving_inference', 'eval_observability');


-- =============================================================================
-- SECTION 2: DUCKDB LOCAL SQL DDL (signal.duckdb)
-- =============================================================================

-- Bronze
CREATE TABLE IF NOT EXISTS bronze_raw_feeds (
    source_id VARCHAR NOT NULL,
    source_type VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL,
    ingestion_timestamp TIMESTAMP NOT NULL,
    content_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_id, content_hash)
);

CREATE TABLE IF NOT EXISTS bronze_checkpoints (
    feed_name VARCHAR PRIMARY KEY,
    last_ingested_at TIMESTAMP NOT NULL,
    record_count BIGINT NOT NULL DEFAULT 0,
    checkpoint_offset VARCHAR
);

-- Silver
CREATE TABLE IF NOT EXISTS silver_evidence (
    evidence_id VARCHAR PRIMARY KEY,
    source_id VARCHAR NOT NULL,
    source_type VARCHAR NOT NULL,
    source_url VARCHAR NOT NULL,
    publication_date TIMESTAMP,
    claim_text VARCHAR NOT NULL,
    confidence VARCHAR NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_developments (
    development_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    summary VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    date DATE NOT NULL,
    confidence_level VARCHAR NOT NULL,
    primary_entity_id VARCHAR,
    source_id VARCHAR NOT NULL,
    raw_ref VARCHAR NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_models (
    model_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    release_date DATE NOT NULL,
    parameter_count BIGINT,
    architecture VARCHAR,
    context_window INTEGER,
    license VARCHAR NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_tools (
    tool_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    repo_url VARCHAR,
    release_version VARCHAR,
    stars INTEGER,
    ecosystem_role VARCHAR,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_benchmarks (
    benchmark_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    evaluation_metric VARCHAR NOT NULL,
    dataset_size INTEGER,
    task_type VARCHAR NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_evaluations (
    eval_id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    benchmark_id VARCHAR NOT NULL,
    score DOUBLE NOT NULL,
    eval_date DATE NOT NULL,
    methodology VARCHAR NOT NULL,
    source_url VARCHAR NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_capabilities (
    capability_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    parent_capability VARCHAR,
    description VARCHAR,
    extraction_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS silver_relationships (
    source_entity_id VARCHAR NOT NULL,
    target_entity_id VARCHAR NOT NULL,
    relationship_type VARCHAR NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    evidence_id VARCHAR NOT NULL,
    extraction_timestamp TIMESTAMP NOT NULL,
    PRIMARY KEY (source_entity_id, target_entity_id, relationship_type)
);

-- Gold Views (DuckDB syntax)
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


-- =============================================================================
-- SECTION 3: VERIFIED ANALYTICAL QUERIES (Genie & UI Benchmarks)
-- =============================================================================

-- Query 1: Top Recent Developments in the Last 30 Days by Category
-- Used by: Timeline View & Genie ("What are the latest AI developments?")
SELECT 
    category,
    COUNT(*) AS developments_count,
    STRING_AGG(title, ' | ') AS highlights
FROM v_recent_developments
GROUP BY category
ORDER BY developments_count DESC;

-- Query 2: Comparable Model Benchmark Evaluation Matrix
-- Enforces: Methodology Matching Rule (0-shot CoT vs 0-shot CoT)
-- Used by: Model Comparison View & Genie ("Compare DeepSeek R1 and Claude 3.5 Sonnet on benchmarks")
SELECT
    benchmark_name,
    task_type,
    methodology,
    model_name,
    score,
    delta_vs_methodology_avg,
    rank_in_benchmark_methodology
FROM v_model_comparisons
WHERE methodology = '0-shot CoT'
ORDER BY benchmark_name ASC, rank_in_benchmark_methodology ASC;

-- Query 3: Connected Ecosystem Knowledge Graph
-- Used by: Ecosystem Graph View & Genie ("Which tools and capabilities depend on DeepSeek R1?")
SELECT
    source_entity_name,
    source_type,
    relationship_type,
    target_entity_name,
    target_type,
    strength,
    confidence,
    source_url
FROM v_ecosystem_relationships
WHERE source_entity_id = 'deepseek-r1' OR target_entity_id = 'deepseek-r1'
ORDER BY strength DESC;

-- Query 4: AI Agent Framework Velocity & Momentum
-- Used by: Agent Trends View & Genie ("Which AI agent frameworks are gaining momentum?")
SELECT
    name,
    category,
    release_version,
    stars,
    developments_last_30d,
    total_integrations,
    momentum_score,
    velocity_category
FROM v_agent_trends
ORDER BY momentum_score DESC;

-- Query 5: Scientific Provenance & Confidence Audit
-- Distinguishes ANNOUNCEMENT, EVALUATED, and PEER_REVIEWED claims
SELECT 
    confidence,
    COUNT(*) AS evidence_count,
    COUNT(DISTINCT source_type) AS source_types_count
FROM silver_evidence
GROUP BY confidence
ORDER BY evidence_count DESC;
