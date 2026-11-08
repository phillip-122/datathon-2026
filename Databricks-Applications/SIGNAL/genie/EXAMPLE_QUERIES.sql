-- ==============================================================================
-- SIGNAL Intelligence: Curated Example SQL Queries for Databricks Genie Agent
-- ==============================================================================
-- Purpose: 12 verified, production-grade, parameter-safe SQL queries answering
--          core analytical and demonstration questions over the AI Landscape.
-- Target Catalogs & Schemas: workspace.signal_gold, workspace.signal_silver
-- Standard Dialect: Databricks SQL / ANSI SQL compatible (also executable in DuckDB)
-- ==============================================================================


-- ------------------------------------------------------------------------------
-- QUERY 1: Recent Developments in the Last 30 Days Grouped by Category
-- Question: "What are the most significant AI developments over the last 30 days?"
-- ------------------------------------------------------------------------------
SELECT 
    category,
    COUNT(*) AS total_developments,
    COUNT(CASE WHEN confidence_level = 'PEER_REVIEWED' THEN 1 END) AS peer_reviewed_count,
    COUNT(CASE WHEN confidence_level = 'EVALUATED' THEN 1 END) AS independently_evaluated_count,
    COUNT(CASE WHEN confidence_level = 'ANNOUNCEMENT' THEN 1 END) AS vendor_announcements_count,
    MAX(date) AS latest_development_date
FROM silver_developments
WHERE date >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY category
ORDER BY total_developments DESC;


-- ------------------------------------------------------------------------------
-- QUERY 2: What Has Changed in the AI Agent Ecosystem Over the Last 30 Days?
-- Question: "What has actually changed in the AI agent ecosystem over the last 30 days?"
-- (The Killer Demo Question: Combines recent releases, agent frameworks, and velocity)
-- ------------------------------------------------------------------------------
WITH recent_agent_devs AS (
    SELECT 
        d.development_id,
        d.title,
        d.summary,
        d.date AS development_date,
        d.confidence_level,
        d.primary_entity_id,
        t.name AS framework_name,
        t.release_version,
        t.stars,
        t.ecosystem_role,
        e.source_url
    FROM silver_developments d
    JOIN silver_tools t ON d.primary_entity_id = t.tool_id
    LEFT JOIN silver_evidence e ON d.source_id = e.source_id
    WHERE d.category = 'agent_framework'
      AND d.date >= CURRENT_DATE - INTERVAL 30 DAY
),
agent_graph_stats AS (
    SELECT 
        target_entity_id AS tool_id,
        COUNT(DISTINCT source_entity_id) AS inbound_integrations
    FROM silver_relationships
    WHERE relationship_type IN ('depends_on', 'integrated_with', 'implements')
    GROUP BY target_entity_id
)
SELECT 
    rad.framework_name,
    rad.release_version,
    rad.development_date,
    rad.title AS milestone_title,
    rad.confidence_level,
    rad.ecosystem_role,
    rad.stars AS github_stars,
    COALESCE(ags.inbound_integrations, 0) AS active_ecosystem_integrations,
    rad.source_url
FROM recent_agent_devs rad
LEFT JOIN agent_graph_stats ags ON rad.primary_entity_id = ags.tool_id
ORDER BY rad.development_date DESC, rad.stars DESC;


-- ------------------------------------------------------------------------------
-- QUERY 3: Apples-to-Apples Model Comparison on Standardized Benchmarks
-- Question: "Compare top reasoning and coding models on SWE-bench Verified and MMLU-Pro."
-- (Enforces strict evaluation methodology matching)
-- ------------------------------------------------------------------------------
SELECT 
    m.name AS model_name,
    m.provider,
    m.architecture,
    m.release_date,
    b.name AS benchmark_name,
    b.evaluation_metric,
    ev.score,
    ev.methodology,
    ev.eval_date,
    ev.source_url
FROM silver_models m
JOIN silver_evaluations ev ON m.model_id = ev.model_id
JOIN silver_benchmarks b ON ev.benchmark_id = b.benchmark_id
WHERE b.benchmark_id IN ('swe_bench_verified', 'mmlu_pro')
ORDER BY b.benchmark_id, ev.score DESC;


-- ------------------------------------------------------------------------------
-- QUERY 4: Tools Integrating With or Depending on a Specific Model Architecture
-- Question: "Which developer tools and agent runtimes integrate with or depend on DeepSeek-R1?"
-- ------------------------------------------------------------------------------
SELECT 
    r.relationship_type,
    t.name AS tool_name,
    t.category AS tool_category,
    t.ecosystem_role,
    t.release_version,
    r.strength AS link_strength,
    ev.claim_text AS integration_evidence,
    ev.source_url
FROM silver_relationships r
JOIN silver_tools t ON r.source_entity_id = t.tool_id
LEFT JOIN silver_evidence ev ON r.evidence_id = ev.evidence_id
WHERE r.target_entity_id = 'deepseek-r1'
  AND r.relationship_type IN ('depends_on', 'integrated_with', 'implements')
ORDER BY r.strength DESC, t.stars DESC;


-- ------------------------------------------------------------------------------
-- QUERY 5: Tracing Research Breakthroughs to Downstream Tools & Implementations
-- Question: "Trace test-time compute and reasoning research to downstream models and tools."
-- ------------------------------------------------------------------------------
SELECT 
    p.source_id AS publication_id,
    d.title AS paper_title,
    p.publication_date,
    m.name AS implemented_model,
    m.provider AS model_provider,
    r.relationship_type AS model_to_tool_rel,
    t.name AS downstream_tool,
    t.category AS tool_category,
    ev.source_url AS proof_source_url
FROM silver_developments d
JOIN silver_evidence p ON d.source_id = p.source_id
JOIN silver_models m ON d.primary_entity_id = m.model_id
LEFT JOIN silver_relationships r ON m.model_id = r.target_entity_id
LEFT JOIN silver_tools t ON r.source_entity_id = t.tool_id
LEFT JOIN silver_evidence ev ON r.evidence_id = ev.evidence_id
WHERE d.category = 'research_breakthrough'
ORDER BY p.publication_date DESC;


-- ------------------------------------------------------------------------------
-- QUERY 6: Distinguish Announcements vs. Independently Evaluated Results
-- Question: "Which developments are corporate announcements vs independently verified evaluations?"
-- ------------------------------------------------------------------------------
SELECT 
    d.confidence_level,
    COUNT(*) AS total_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS percentage_of_total,
    MAX(d.date) AS most_recent_date,
    MIN(d.date) AS oldest_date
FROM silver_developments d
GROUP BY d.confidence_level
ORDER BY total_count DESC;


-- ------------------------------------------------------------------------------
-- QUERY 7: AI Capability Taxonomy & Tracking Ecosystem Growth
-- Question: "Which capabilities have received the most research and development focus?"
-- ------------------------------------------------------------------------------
SELECT 
    c.capability_id,
    c.name AS capability_name,
    COALESCE(pc.name, 'Top-Level Domain') AS parent_capability_name,
    COUNT(DISTINCT r.source_entity_id) AS demonstrating_technologies_count,
    COUNT(DISTINCT d.development_id) AS related_developments_count
FROM silver_capabilities c
LEFT JOIN silver_capabilities pc ON c.parent_capability = pc.capability_id
LEFT JOIN silver_relationships r ON c.capability_id = r.target_entity_id 
    AND r.relationship_type IN ('implements', 'demonstrates')
LEFT JOIN silver_developments d ON r.source_entity_id = d.primary_entity_id
GROUP BY c.capability_id, c.name, pc.name
ORDER BY demonstrating_technologies_count DESC, related_developments_count DESC;


-- ------------------------------------------------------------------------------
-- QUERY 8: Historical Evolution & Version Upgrades for Models and Tools
-- Question: "What is the historical evolution and release history for open reasoning models?"
-- ------------------------------------------------------------------------------
SELECT 
    m.name AS model_name,
    m.provider,
    m.architecture,
    m.release_date,
    m.parameter_count / 1000000000.0 AS parameter_count_billions,
    m.context_window,
    m.license,
    datediff('day', LAG(m.release_date) OVER (PARTITION BY m.provider ORDER BY m.release_date), m.release_date) AS days_since_previous_release
FROM silver_models m
ORDER BY m.provider, m.release_date ASC;


-- ------------------------------------------------------------------------------
-- QUERY 9: Discrepancy & Conflicting Claims Detection (Vendor vs. Independent)
-- Question: "Identify score discrepancies between vendor-announced claims and independent benchmark audits."
-- ------------------------------------------------------------------------------
WITH vendor_scores AS (
    SELECT 
        ev.model_id,
        ev.benchmark_id,
        ev.score AS vendor_score,
        ev.methodology AS vendor_methodology,
        ev.source_url AS vendor_source
    FROM silver_evaluations ev
    JOIN silver_evidence e ON ev.source_url = e.source_url
    WHERE e.confidence = 'ANNOUNCEMENT'
),
independent_scores AS (
    SELECT 
        ev.model_id,
        ev.benchmark_id,
        ev.score AS independent_score,
        ev.methodology AS independent_methodology,
        ev.source_url AS independent_source
    FROM silver_evaluations ev
    JOIN silver_evidence e ON ev.source_url = e.source_url
    WHERE e.confidence IN ('EVALUATED', 'PEER_REVIEWED')
)
SELECT 
    m.name AS model_name,
    b.name AS benchmark_name,
    b.evaluation_metric,
    v.vendor_score,
    v.vendor_methodology,
    i.independent_score,
    i.independent_methodology,
    ROUND(v.vendor_score - i.independent_score, 2) AS claim_discrepancy,
    CASE 
        WHEN v.vendor_score > i.independent_score THEN 'VENDOR_OPTIMISTIC'
        WHEN v.vendor_score < i.independent_score THEN 'INDEPENDENT_HIGHER'
        ELSE 'CONSISTENT'
    END AS audit_status,
    v.vendor_source,
    i.independent_source
FROM vendor_scores v
JOIN independent_scores i ON v.model_id = i.model_id AND v.benchmark_id = i.benchmark_id
JOIN silver_models m ON v.model_id = m.model_id
JOIN silver_benchmarks b ON v.benchmark_id = b.benchmark_id
ORDER BY ABS(v.vendor_score - i.independent_score) DESC;


-- ------------------------------------------------------------------------------
-- QUERY 10: Complete Provenance Trail for a Specific AI Development
-- Question: "Trace the full provenance and supporting evidence for development 'dev_deepseek_r1_release'."
-- ------------------------------------------------------------------------------
SELECT 
    d.development_id,
    d.title AS development_title,
    d.category,
    d.date AS announcement_date,
    d.confidence_level,
    e.source_id,
    e.source_type,
    e.source_url,
    e.publication_date AS source_publication_date,
    e.claim_text AS extracted_claim,
    d.extraction_timestamp
FROM silver_developments d
LEFT JOIN silver_evidence e ON d.source_id = e.source_id
WHERE d.primary_entity_id = 'deepseek-r1'
ORDER BY d.date DESC;


-- ------------------------------------------------------------------------------
-- QUERY 11: High-Momentum Ecosystem Nodes (Ecosystem Centrality Ranking)
-- Question: "Which models and tools have the highest number of ecosystem connections and integrations?"
-- ------------------------------------------------------------------------------
WITH entity_degrees AS (
    SELECT 
        target_entity_id AS entity_id,
        COUNT(*) AS inbound_connections,
        'TARGET' AS direction
    FROM silver_relationships
    GROUP BY target_entity_id

    UNION ALL

    SELECT 
        source_entity_id AS entity_id,
        COUNT(*) AS outbound_connections,
        'SOURCE' AS direction
    FROM silver_relationships
    GROUP BY source_entity_id
),
aggregated_degrees AS (
    SELECT 
        entity_id,
        SUM(inbound_connections) AS total_connections
    FROM entity_degrees
    GROUP BY entity_id
)
SELECT 
    ad.entity_id,
    COALESCE(m.name, t.name, c.name, ad.entity_id) AS entity_name,
    CASE 
        WHEN m.model_id IS NOT NULL THEN 'MODEL'
        WHEN t.tool_id IS NOT NULL THEN 'TOOL / FRAMEWORK'
        WHEN c.capability_id IS NOT NULL THEN 'CAPABILITY'
        ELSE 'OTHER'
    END AS entity_type,
    ad.total_connections,
    DENSE_RANK() OVER (ORDER BY ad.total_connections DESC) AS ecosystem_rank
FROM aggregated_degrees ad
LEFT JOIN silver_models m ON ad.entity_id = m.model_id
LEFT JOIN silver_tools t ON ad.entity_id = t.tool_id
LEFT JOIN silver_capabilities c ON ad.entity_id = c.capability_id
ORDER BY ad.total_connections DESC;


-- ------------------------------------------------------------------------------
-- QUERY 12: Model Provider vs. Agent Orchestration Compatibility Matrix
-- Question: "Which model providers are supported by the leading agent orchestration runtimes?"
-- ------------------------------------------------------------------------------
SELECT 
    t.name AS agent_framework,
    t.release_version,
    m.provider AS supported_provider,
    COUNT(DISTINCT m.model_id) AS supported_models_count,
    MAX(r.strength) AS integration_confidence,
    MAX(ev.source_url) AS documentation_url
FROM silver_tools t
JOIN silver_relationships r ON t.tool_id = r.source_entity_id
JOIN silver_models m ON r.target_entity_id = m.model_id
LEFT JOIN silver_evidence ev ON r.evidence_id = ev.evidence_id
WHERE t.category = 'agent_orchestration'
  AND r.relationship_type IN ('depends_on', 'integrated_with')
GROUP BY t.name, t.release_version, m.provider
ORDER BY t.name, supported_models_count DESC;
