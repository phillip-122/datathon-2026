# SIGNAL Evaluation: Golden Expected Results & Methodology Guide

This document establishes the ground-truth criteria, expected analytical outcomes, SQL validation assertions, and critical edge-case pitfalls for evaluating the **SIGNAL Intelligence** Databricks Genie Agent and local analytics engine.

---

## 1. Evaluation Philosophy & Epistemic Principles

SIGNAL is an AI landscape observatory designed for critical, evidence-grounded intelligence. A response is considered **correct** if and only if it adheres to these five core principles:

1. **Epistemic Humility & Evidence Grounding:** Factual assertions about models, tools, and benchmarks must cite traceable source records in `silver_evidence`. An unverified corporate blog post must be explicitly labeled as an `ANNOUNCEMENT`, whereas peer-reviewed or independently replicated scores receive `PEER_REVIEWED` or `EVALUATED` status.
2. **Strict Methodology Matching (No Apples-to-Oranges):** A model evaluated under `0-shot CoT` cannot be directly ranked against a model running with a multi-step agentic scaffold (e.g., SWE-bench multi-attempt agent loop) without explicitly stating the difference in methodology.
3. **Temporal Precision:** Technological milestones must be indexed by `release_date`, `publication_date`, or `date`. Using `ingestion_timestamp` or `extraction_timestamp` to represent the historical timing of an innovation is considered an evaluation failure.
4. **Graph Edge Provenance:** Relationships such as `depends_on` or `implements` must be backed by concrete code commits, release documentation, or technical papers, not speculative associations.
5. **Absence of Evidence is Not Evidence of Absence:** If a tool or model is missing from the dataset, the system must explain what data is currently tracked rather than hallucinating that the tool does not exist or has zero users.

---

## 2. Common Pitfalls & Anti-Patterns Catalog

| Pitfall / Failure Mode | Why It Fails | Correct Analytical Behavior |
|---|---|---|
| **Prompting Strategy Conflation** | Comparing a 0-shot direct benchmark score with a 5-shot CoT or agentic scaffold score creates a distorted, invalid comparison. | Always filter or group by `methodology` (e.g., `WHERE ev.methodology = '0-shot CoT'`) or display methodology as an explicit comparison dimension. |
| **Date Field Conflation** | Conflating the date an article was ingested into the database (`extraction_timestamp`) with the date the model or paper was released (`date`). | Use `silver_developments.date` or `silver_models.release_date` for historical timelines and rolling 30-day windows. |
| **Treating Marketing as Evaluation** | Vendor press releases frequently report best-case cherry-picked benchmarks that fail independent reproduction. | Flag vendor scores as `ANNOUNCEMENT` / `SELF_REPORTED` and highlight third-party audits (`EVALUATED` / `PEER_REVIEWED`) where available. |
| **Speculative Momentum Claims** | Claiming a technology is "exploding in popularity" based on 1 or 2 press releases. | Compute `momentum_score` across multiple dimensions: 30-day release count, GitHub stars, and knowledge graph integrations. Explicitly report the timeframe. |
| **Reversing Knowledge Graph Edges** | Asking "which tools integrate with DeepSeek-R1" and filtering `WHERE source_entity_id = 'deepseek-r1'` instead of `WHERE target_entity_id = 'deepseek-r1'`. | Respect edge direction: Tools (`source_entity_id`) depend on or integrate with Models (`target_entity_id`). |
| **Hallucination of Missing Tech** | Stating "Company X has ceased AI development" simply because no paper was ingested this week. | State: *"No new records for Company X were captured in the current 30-day window. Tracked records reflect available sources."* |

---

## 3. Detailed Expected Results for Evaluation Suite

### EVAL-SIG-001: 30-Day Developments by Category
- **Question:** *"What are the most significant AI developments in the last 30 days grouped by category?"*
- **Expected SQL Pattern:**
  ```sql
  SELECT category, COUNT(*) AS count
  FROM silver_developments
  WHERE date >= CURRENT_DATE - INTERVAL 30 DAY
  GROUP BY category ORDER BY count DESC;
  ```
- **Golden Result Expectations:**
  - Distinct categories: `model_release`, `agent_framework`, `research_breakthrough`, `benchmark`.
  - Temporal filter uses `date >= CURRENT_DATE - INTERVAL 30 DAY`.
  - Output shows distribution across categories with latest dates.

---

### EVAL-SIG-002: AI Agent Ecosystem Changes (The Killer Demo)
- **Question:** *"What has actually changed in the AI agent ecosystem over the last 30 days?"*
- **Expected SQL Pattern:**
  Joins `silver_developments`, `silver_tools`, and `silver_relationships` filtered by `category = 'agent_framework'` and rolling 30 days.
- **Golden Result Expectations:**
  - Identifies active frameworks: `LangGraph`, `AutoGen`, `CrewAI`, `DSPy`.
  - Surfaces recent version upgrades (e.g. LangGraph v0.2.60+, CrewAI multi-agent flows).
  - Summarizes architectural shifts (e.g., transition toward graph-based stateful orchestration and integration with reasoning models).
  - Preserves source provenance links for each milestone.

---

### EVAL-SIG-003: Model Comparison on SWE-bench Verified & MMLU-Pro
- **Question:** *"Compare Claude 3.5 Sonnet and DeepSeek R1 across comparable standardized benchmarks."*
- **Expected SQL Pattern:**
  ```sql
  SELECT m.name, b.name, ev.score, ev.methodology, ev.eval_date
  FROM silver_models m
  JOIN silver_evaluations ev ON m.model_id = ev.model_id
  JOIN silver_benchmarks b ON ev.benchmark_id = b.benchmark_id
  WHERE m.model_id IN ('claude-3-5-sonnet', 'deepseek-r1')
    AND b.benchmark_id IN ('swe_bench_verified', 'mmlu_pro')
  ORDER BY b.benchmark_id, ev.score DESC;
  ```
- **Golden Result Expectations:**
  - `SWE-bench Verified`: Both models evaluated on identical benchmark sets; reports resolve rate percentage.
  - `MMLU-Pro`: Reports reasoning accuracy.
  - Methodology column clearly displays whether 0-shot CoT or scaffold was used.
  - Explicitly states that while both achieve state-of-the-art results, DeepSeek-R1 utilizes test-time compute reinforcement learning while Claude 3.5 Sonnet operates as a high-efficiency dense/hybrid model.

---

### EVAL-SIG-004: Tools Integrating with DeepSeek-R1
- **Question:** *"Which developer tools and agent frameworks integrate with or depend on DeepSeek-R1?"*
- **Expected SQL Pattern:**
  ```sql
  SELECT t.name, t.category, r.relationship_type, ev.claim_text, ev.source_url
  FROM silver_relationships r
  JOIN silver_tools t ON r.source_entity_id = t.tool_id
  LEFT JOIN silver_evidence ev ON r.evidence_id = ev.evidence_id
  WHERE r.target_entity_id = 'deepseek-r1'
    AND r.relationship_type IN ('depends_on', 'integrated_with', 'implements')
  ORDER BY t.stars DESC;
  ```
- **Golden Result Expectations:**
  - Inference runtimes: `vLLM` (implements MoE kernel & deepseek support), `SGLang` (optimized MLA attention).
  - Orchestration frameworks: `LangGraph`, `Ollama` (local deployment).
  - Each item must display the verified evidence citation link.

---

### EVAL-SIG-005: Trace Research Papers to Downstream Tools
- **Question:** *"Trace research publications on test-time compute to downstream models and developer tools."*
- **Expected SQL Pattern:**
  Multi-hop graph traversal: Research Paper $\rightarrow$ Model (implements) $\rightarrow$ Tool (integrates/serves).
- **Golden Result Expectations:**
  - Traces DeepSeek-R1 technical report / test-time compute research $\rightarrow$ `deepseek-r1` $\rightarrow$ `vLLM` / `LangGraph`.
  - Traces step-by-step reasoning papers $\rightarrow$ reasoning models $\rightarrow$ developer agent runtimes.
  - Returns complete path with edge types and timestamps showing chronological progression from paper release to tool integration.

---

### EVAL-SIG-006: Announcements vs. Independently Evaluated Results
- **Question:** *"Which developments in the dataset are vendor announcements versus independently evaluated results?"*
- **Expected SQL Pattern:**
  ```sql
  SELECT confidence_level, COUNT(*) AS total,
         ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
  FROM silver_developments
  GROUP BY confidence_level;
  ```
- **Golden Result Expectations:**
  - Accurately splits records by `ANNOUNCEMENT`, `EVALUATED`, `PEER_REVIEWED`, `EMPIRICAL`.
  - Explains the distinction: Announcements represent unverified vendor claims; Evaluated/Peer-Reviewed records represent audited or published findings.

---

### EVAL-SIG-007: Capability Taxonomy & Research Attention
- **Question:** *"Which capabilities have received the most new research and implementation attention?"*
- **Expected SQL Pattern:**
  Aggregates inbound relationships and developments per capability in `silver_capabilities`.
- **Golden Result Expectations:**
  - Top capabilities highlighted: `cap_test_time_compute` (Reasoning & Search), `cap_agent_planning` (Agentic Tool Loops), `cap_code_gen` (Autonomous Software Engineering).
  - Ranks capabilities by count of implementing models and developer tools.

---

### EVAL-SIG-008: Release Chronology of Reasoning Models
- **Question:** *"What is the release chronology and architectural evolution of open-weight reasoning models?"*
- **Expected SQL Pattern:**
  Sorts models by `release_date ASC` where architecture indicates MoE or reasoning.
- **Golden Result Expectations:**
  - Chronological timeline starting from early open models to recent MoE reasoning architectures.
  - Displays parameter count, active vs total parameters, context window, and license.

---

### EVAL-SIG-009: Vendor Claims vs. Independent Audits Discrepancies
- **Question:** *"Identify score discrepancies between vendor-announced claims and independent benchmark audits."*
- **Expected SQL Pattern:**
  Compares vendor self-reported score records against independent third-party evaluation records on matching models and benchmarks.
- **Golden Result Expectations:**
  - Identifies variance between initial corporate announcements and independent verification runs.
  - Flags whether variance is due to differences in prompting (e.g. cherry-picked $n$-pass vs single-pass) or data contamination.

---

### EVAL-SIG-010: Provenance Trail for a Specific Milestone
- **Question:** *"Trace the complete evidence trail and source provenance for development record 'dev_deepseek_r1_release'."*
- **Expected SQL Pattern:**
  Retrieves development headline joined with `silver_evidence` matching `source_id`.
- **Golden Result Expectations:**
  - Returns: `source_type` (`arxiv`), `source_url` (canonical paper link), `publication_date`, `claim_text`, and Bronze payload reference hash.
  - Validates that no claim exists without source evidence.

---

### EVAL-SIG-011: Ecosystem Centrality & Integration Hubs
- **Question:** *"Which models and tools have the highest degree centrality across the ecosystem?"*
- **Expected SQL Pattern:**
  Calculates inbound and outbound degree centrality across `silver_relationships`.
- **Golden Result Expectations:**
  - Top hub nodes: `deepseek-r1`, `langgraph`, `vllm`, `claude-3-5-sonnet`.
  - Explains why high-centrality nodes represent ecosystem foundational standards.

---

### EVAL-SIG-012: Multi-Provider Agent Frameworks
- **Question:** *"Which agent orchestration frameworks support multiple foundation model providers?"*
- **Expected SQL Pattern:**
  Aggregates distinct providers supported per agent tool from `silver_relationships` with `HAVING COUNT(DISTINCT provider) > 1`.
- **Golden Result Expectations:**
  - Identifies vendor-agnostic frameworks (`LangGraph`, `AutoGen`, `CrewAI`).
  - Lists the matrix of supported model providers (Anthropic, OpenAI, Meta, DeepSeek).

---

### EVAL-SIG-013: Open Weights vs. Proprietary Model Licensing Distribution
- **Question:** *"What is the distribution of open-weights versus proprietary API licenses among tracked foundation models?"*
- **Expected SQL Pattern:**
  Groups `silver_models` into Open Weights vs Proprietary / API.
- **Golden Result Expectations:**
  - Computes counts and percentages.
  - Accurately classifies Llama 3.3 and DeepSeek R1 as open-weights and Claude 3.5 as proprietary API.

---

### EVAL-SIG-014: Benchmark Coverage Across Tracked Models
- **Question:** *"Which benchmarks have the highest coverage across evaluated models?"*
- **Expected SQL Pattern:**
  Counts distinct models evaluated per benchmark in `silver_evaluations`.
- **Golden Result Expectations:**
  - SWE-bench Verified and MMLU-Pro emerge as the industry standard benchmarks with the broadest multi-model coverage.

---

### EVAL-SIG-015: 30-Day Momentum Metrics for Agent Frameworks
- **Question:** *"What is the calculated 30-day velocity score for top agent orchestration frameworks?"*
- **Expected SQL Pattern:**
  Queries `v_agent_trends` for `momentum_score` and `velocity_category`.
- **Golden Result Expectations:**
  - Ranks frameworks by composite score combining release activity, GitHub community size, and active knowledge graph integrations.
  - Classifies frameworks into `ACCELERATING`, `STEADY`, `MATURE`, or `EMERGING`.

---

### EVAL-SIG-016: Data Quality & Referential Integrity Audit
- **Question:** *"Are there any orphan developments or unverified entities lacking evidence records?"*
- **Expected SQL Pattern:**
  Outer joins `silver_developments` to `silver_evidence` filtering `WHERE e.evidence_id IS NULL`.
- **Golden Result Expectations:**
  - Zero orphan rows returned.
  - Confirms referential integrity across the Medallion architecture.
