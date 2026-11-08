# Databricks Genie Agent Configuration Guide: SIGNAL Intelligence

This guide provides the complete specification, system instructions, schema catalog annotations, semantic dictionary, and query design principles for the **SIGNAL Intelligence** Databricks Genie Agent.

---

## 1. Genie Space Metadata

| Attribute | Specification |
|---|---|
| **Space Name** | `SIGNAL Intelligence` |
| **Description** | An analytical observatory agent for querying the evolving AI landscape: tracking models, tools, research breakthroughs, benchmarks, capabilities, and dependencies. |
| **Default Catalog** | `workspace` |
| **Target Schemas** | `workspace.signal_gold`, `workspace.signal_silver` |
| **Recommended Compute** | Databricks SQL Serverless Warehouse (e.g., `01b90cfaa061e148`) |
| **Primary Tables / Views** | `v_recent_developments`, `v_model_comparisons`, `v_ecosystem_relationships`, `v_agent_trends`, `silver_developments`, `silver_models`, `silver_tools`, `silver_benchmarks`, `silver_evaluations`, `silver_capabilities`, `silver_relationships`, `silver_evidence` |

---

## 2. General Instructions

Copy and configure the following system prompt verbatim as the **General Instructions** in the Genie Agent settings:

```text
You are SIGNAL Intelligence, an analytical assistant for exploring
the evolving AI technology ecosystem.

Your purpose is to help users understand developments, technologies,
capabilities, research, benchmarks, and relationships across the
AI landscape.

DATA RULES:

1. Base factual answers about tracked developments on the available
   datasets and their supporting source records.

2. Distinguish announcements, research claims, independently
   evaluated results, and community observations.

3. Do not treat an announced capability as independently verified.

4. Do not compare benchmark scores unless the benchmark,
   evaluation methodology, model version, and relevant conditions
   are sufficiently comparable.

5. Distinguish publication dates, release dates, discovery dates,
   and ingestion dates.

6. Do not assume that missing records indicate that a technology
   does not exist.

7. Preserve source attribution and identify the supporting sources
   for important claims.

8. When describing trends, specify the time period and available
   evidence. Avoid claiming momentum from a small number of records.

9. Distinguish a direct relationship from a speculative relationship.

10. Do not invent model capabilities, release dates, benchmark
    results, tool integrations, or adoption statistics.

ANALYTICAL BEHAVIOR:

- Explain why developments are related.
- Compare versions and technologies when comparable evidence exists.
- Identify meaningful changes over time.
- Identify conflicting claims.
- Explain what additional evidence would be needed to establish
  a conclusion.
- Prefer structured comparisons over generic summaries.

If a question cannot be answered from the available datasets,
state what information is missing rather than fabricating an answer.
```

---

## 3. Dataset Catalog & Table Annotations

The Genie Agent accesses curated views in `workspace.signal_gold` as well as normalized relational tables in `workspace.signal_silver`.

### 3.1 Gold Analytical Views (Primary Query Targets)

#### `workspace.signal_gold.v_recent_developments`
- **Description:** Pre-joined view of AI developments from the last 30 rolling days, linking headlines to primary entities (models or tools), confidence levels, and provenance source URLs.
- **When to Use:** Use for queries about "what's new", "recent developments", "announcements in the last month", or "weekly breakdown of AI activity".
- **Column Annotations:**
  - `development_id` (STRING, PK): Unique canonical identifier for the development record (e.g., `dev_2026_0101`).
  - `title` (STRING): Clear headline summarizing the milestone.
  - `summary` (STRING): Normalized analytical summary highlighting technical significance.
  - `category` (STRING): High-level category: `model_release`, `agent_framework`, `benchmark`, `research_breakthrough`, `infra_tool`.
  - `date` (DATE): The official announcement, publication, or release date. Filter on this for temporal windows.
  - `confidence_level` (STRING): Rigor tier: `ANNOUNCEMENT` (vendor marketing/blog), `EVALUATED` (empirical testing performed), `PEER_REVIEWED` (formal research paper), `EMPIRICAL` (community benchmarks), `HYPOTHESIS` (unverified claim).
  - `primary_entity_id` (STRING, FK): Identifier of the main model (`model_id`) or tool (`tool_id`) featured.
  - `primary_entity_name` (STRING): Human-readable entity name (e.g., `DeepSeek R1`, `LangGraph`).
  - `primary_entity_type` (STRING): Entity classification (`model`, `tool`, or `general`).
  - `source_type` (STRING): Ingestion origin (`arxiv`, `huggingface`, `github`, `official_announcement`).
  - `source_url` (STRING): Canonical web link to the original publication, GitHub repo, or release notice.
  - `claim_text` (STRING): Exact finding or excerpt extracted from the source document.
  - `extraction_timestamp` (TIMESTAMP): Time when the record was processed into the silver layer.

#### `workspace.signal_gold.v_model_comparisons`
- **Description:** Normalized benchmark comparison matrix joining models, benchmarks, and evaluated scores.
- **When to Use:** Use whenever comparing models (e.g., "compare Claude 3.5 Sonnet and DeepSeek R1 on coding", "highest scoring models on SWE-bench Verified", "MMLU-Pro leaderboard").
- **Column Annotations:**
  - `model_id` (STRING): Unique model slug.
  - `model_name` (STRING): Formal model name.
  - `provider` (STRING): Model creator / lab (e.g., `Anthropic`, `DeepSeek`, `OpenAI`, `Meta`, `Google`).
  - `release_date` (DATE): Model public release date.
  - `parameter_count` (BIGINT): Total or active parameters (NULL if proprietary/unreleased).
  - `architecture` (STRING): Underlying model architecture (e.g., `MoE`, `Dense Transformer`).
  - `context_window` (INTEGER): Context window size in tokens.
  - `license` (STRING): License type (`Open Weights / MIT`, `Proprietary / API`, etc.).
  - `benchmark_id` (STRING): Benchmark identifier (e.g., `mmlu_pro`, `swe_bench_verified`, `gpqa_diamond`).
  - `benchmark_name` (STRING): Human-readable benchmark title.
  - `task_type` (STRING): Domain category (`Reasoning & Knowledge`, `Autonomous Software Engineering`, `Code Synthesis`).
  - `evaluation_metric` (STRING): Metric name and scale (e.g., `Accuracy (%)`, `Resolved Rate (%)`).
  - `score` (DOUBLE): Numeric score achieved.
  - `methodology` (STRING): Evaluation setup (e.g., `0-shot CoT`, `5-shot direct`, `Scaffolded Agent (Pass@1)`).
  - `eval_date` (DATE): Date of the benchmark run.
  - `source_url` (STRING): Provenance link for verification.

#### `workspace.signal_gold.v_ecosystem_relationships`
- **Description:** Directed knowledge graph connecting research papers, model architectures, developer tools, and capabilities with verified evidence edges.
- **When to Use:** Use for tracing dependencies ("which tools depend on model X", "what research inspired framework Y", "how does technology A connect to capability B").
- **Column Annotations:**
  - `source_entity_id` (STRING): Subject entity slug.
  - `source_entity_name` (STRING): Human-readable subject entity name.
  - `source_type` (STRING): Subject entity classification (`model`, `tool`, `capability`, `research`).
  - `relationship_type` (STRING): Relationship semantic edge (`implements`, `depends_on`, `evaluates_on`, `improves_upon`, `integrated_with`, `derived_from`).
  - `target_entity_id` (STRING): Object entity slug.
  - `target_entity_name` (STRING): Human-readable object entity name.
  - `target_type` (STRING): Object entity classification (`model`, `tool`, `capability`, `research`).
  - `strength` (DOUBLE): Numeric confidence/weight of the link (0.0 to 1.0).
  - `evidence_id` (STRING, FK): Pointer to the supporting evidence record.
  - `source_url` (STRING): Web link to the technical report or code commit validating the link.
  - `claim_text` (STRING): Textual justification for the relationship.
  - `confidence` (STRING): Evidence confidence level.

#### `workspace.signal_gold.v_agent_trends`
- **Description:** Rolling 30-day velocity and ecosystem activity scores for autonomous agent orchestration frameworks and inference runtimes.
- **When to Use:** Use for queries regarding "agent ecosystem momentum", "most active agent frameworks", "fastest growing developer tools".
- **Column Annotations:**
  - `tool_id` (STRING): Tool identifier.
  - `name` (STRING): Framework title.
  - `category` (STRING): Category (`agent_orchestration`, `serving_inference`).
  - `release_version` (STRING): Current tracked semantic release tag.
  - `stars` (INTEGER): GitHub star count.
  - `ecosystem_role` (STRING): Operational specialty.
  - `developments_last_30d` (INTEGER): Number of distinct releases or announcements in the last 30 days.
  - `total_integrations` (INTEGER): Number of connected model/tool integrations in the knowledge graph.
  - `momentum_score` (DOUBLE): Composite score combining GitHub activity, release velocity, and graph integrations.
  - `velocity_category` (STRING): Classification: `ACCELERATING`, `STEADY`, `MATURE`, `EMERGING`.

---

### 3.2 Silver Normalized Tables

When deeper investigation or multi-hop relational aggregations are necessary, Genie can query the Silver tables directly:

| Table Name | Description | Key Join Columns |
|---|---|---|
| `silver_developments` | Atomic event record for each tracked announcement, paper, or release | `primary_entity_id`, `source_id` |
| `silver_models` | Normalized AI models catalog with architecture and parameter specs | `model_id` |
| `silver_tools` | Developer tools, agent frameworks, and inference runtimes | `tool_id` |
| `silver_benchmarks` | Benchmark specifications, task types, and metric definitions | `benchmark_id` |
| `silver_evaluations` | Benchmark results tagged with exact evaluation methodology | `model_id`, `benchmark_id` |
| `silver_capabilities` | Hierarchical AI capability taxonomy | `capability_id`, `parent_capability` |
| `silver_relationships`| Directed knowledge graph edges connecting entities | `source_entity_id`, `target_entity_id`, `evidence_id` |
| `silver_evidence` | Provenance records preserving source URLs, claims, and confidence | `evidence_id`, `source_id` |

---

## 4. Join Path Rules & Disambiguation

Genie must strictly follow these join patterns to prevent Cartesian products or misleading aggregations:

1. **Model Comparisons:**
   ```sql
   silver_models m
   JOIN silver_evaluations ev ON m.model_id = ev.model_id
   JOIN silver_benchmarks b ON ev.benchmark_id = b.benchmark_id
   ```
   *Strict Condition:* ALWAYS filter by `ev.methodology` or group by methodology when comparing models to avoid contrasting zero-shot direct answers with multi-turn scaffolded execution.

2. **Developments & Evidence:**
   ```sql
   silver_developments d
   LEFT JOIN silver_evidence e ON d.source_id = e.source_id
   ```

3. **Relationship Graph Traversal:**
   ```sql
   silver_relationships r
   JOIN silver_evidence ev ON r.evidence_id = ev.evidence_id
   ```

4. **Multi-Hop Research to Tools:**
   To connect a research paper to a tool:
   ```sql
   silver_relationships r1  -- Research to Model or Capability
   JOIN silver_relationships r2 ON r1.target_entity_id = r2.source_entity_id -- Model to Tool
   ```

---

## 5. Semantic Dictionary & AI Domain Lexicon

When interpreting user prompts, Genie must adhere to these precise operational definitions:

| Term | Operational Definition in SIGNAL | Querying Guidance |
|---|---|---|
| **Momentum** | A measured composite of release velocity (developments in the last 30 days), community engagement (GitHub stars), and ecosystem adoption (inbound/outbound graph integrations). | Query `v_agent_trends`. Do NOT assert momentum based on a single press release or isolated tweet. Explicitly state the 30-day observation period. |
| **Announcement** | A vendor-authored press release, corporate blog post, or social media post reporting capabilities without independent verification. | Filter for `confidence_level = 'ANNOUNCEMENT'`. Treat all reported numbers as self-reported claims. |
| **Independent Benchmark** | An evaluation conducted by a neutral third party (e.g. LMSYS Chatbot Arena, SWE-bench authors, Scale AI SEAL, METR) or verified peer-reviewed replication. | Filter for `confidence_level IN ('EVALUATED', 'PEER_REVIEWED')` or query `evaluator_type = 'INDEPENDENT'`. |
| **Agent Framework** | Software libraries and runtimes designed to compose multi-step tool execution, state machines, and LLM orchestration (e.g. LangGraph, CrewAI, AutoGen, DSPy). | Query `silver_tools` WHERE `category = 'agent_orchestration'`. |
| **Reasoning Model** | A foundation model explicitly trained to utilize test-time compute scaling (e.g., DeepSeek-R1, OpenAI o1/o3-mini) generating extended internal chains of thought prior to response. | Check `silver_models.architecture LIKE '%Reasoning%'` or relationships connecting to capability `cap_test_time_compute`. |
| **Test-Time Compute** | Dynamic allocation of additional computational cycles during inference (search over token trees, self-critique, verification). | Target capability `cap_test_time_compute` in `silver_capabilities` and related edges. |
| **MoE (Mixture of Experts)** | An architecture where only a subset of total parameters is activated per token (e.g., DeepSeek-V3, Mixtral 8x7B). Total parameters $\ne$ active parameters. | Filter `silver_models.architecture = 'MoE'`. Note that active parameter count is smaller than total parameter footprint. |
| **Publication Date vs Release Date vs Ingestion Date** | - *Publication Date:* Academic paper release date (e.g., arXiv submission timestamp).<br>- *Release Date:* Date code repository or model API weights were published.<br>- *Ingestion Date:* Timestamp when SIGNAL recorded the entry into Bronze. | Always query `date` or `release_date` when answering questions about technology history. NEVER use `extraction_timestamp` or `ingestion_timestamp` as the historical date of an innovation. |

---

## 6. Curated Demonstration Query Mappings

For the core demo scenarios, Genie should map user queries to these canonical SQL patterns:

1. *"What has actually changed in the AI agent ecosystem over the last 30 days?"*
   $\rightarrow$ Aggregate `v_recent_developments` WHERE `category = 'agent_framework'` AND `date >= CURRENT_DATE - INTERVAL 30 DAYS`, joined with `v_agent_trends`.
2. *"Compare models across comparable benchmark evaluations (e.g. MMLU-Pro, SWE-bench)."*
   $\rightarrow$ Query `v_model_comparisons` matching `benchmark_id` and ensuring `methodology` is strictly equal.
3. *"Which tools integrate with or depend on a specific model architecture?"*
   $\rightarrow$ Query `v_ecosystem_relationships` WHERE `target_entity_id = :model_id` AND `relationship_type IN ('depends_on', 'integrated_with')`.
4. *"Trace research papers to downstream tools and implementations."*
   $\rightarrow$ Perform graph traversal across `silver_relationships` linking research publications to model architectures and down to developer tools.
5. *"Identify announcements vs independently evaluated benchmark scores."*
   $\rightarrow$ Group and pivot `silver_developments` or `silver_evaluations` by `confidence_level` / `evaluator_type`.
