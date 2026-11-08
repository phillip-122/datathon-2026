# SIGNAL Data Dictionary & Entity Reference

This document provides the formal data dictionary for all entities, junction records, and graph relationships within the SIGNAL (AI Landscape Intelligence) platform.

All fields are strictly typed, nullability-constrained, and cross-referenced. Downstream pipelines (Agent S2), interactive UI views (Agent S3), and Genie conversational agents (Agent S4) adhere to these definitions.

---

## 1. Entity Relationship Overview

```
       +--------------------+
       |  Research Papers   |
       +---------+----------+
                 | (INTRODUCED_IN / BASED_ON_METHOD)
                 v
+----------------+---+       (EVALUATED_ON)       +-------------------+
|       Models       | <------------------------> |    Benchmarks     |
+----------------+---+      [Evaluations Score]   +-------------------+
                 ^
                 | (ORCHESTRATES / SERVES_MODEL)
                 |
+----------------+---+
|  Developer Tools   |
+--------------------+
```

---

## 2. Models (`models`)

Represents frontier and foundation artificial intelligence models across proprietary and open-weights ecosystems.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `model_id` | `VARCHAR(64)` | NO | Unique slug identifying the model instance (Primary Key). | `deepseek-r1`, `llama-3.3-70b-instruct` |
| `name` | `VARCHAR(128)` | NO | Human-readable official display title. | `"DeepSeek R1"`, `"Claude 3.5 Sonnet"` |
| `organization` | `VARCHAR(64)` | NO | Creator lab or company. | `"DeepSeek"`, `"Meta"`, `"Anthropic"`, `"Google"`, `"OpenAI"` |
| `release_date` | `DATE` | NO | Date of public announcement or release (YYYY-MM-DD). | `2025-01-20` |
| `license` | `VARCHAR(64)` | NO | Legal distribution license. | `"MIT"`, `"Apache-2.0"`, `"Llama 3.3 Community License"`, `"Proprietary"` |
| `architecture` | `VARCHAR(64)` | NO | High-level neural network architecture. | `"MoE (Multi-head Latent Attention)"`, `"Dense Transformer"` |
| `parameter_count_total_b` | `FLOAT` | YES | Total parameters across all layers in Billions. | `671.0`, `70.0`, `14.0` |
| `parameter_count_active_b` | `FLOAT` | YES | Active parameters per forward pass token in Billions. | `37.0`, `70.0`, `14.0` |
| `context_window_tokens` | `INTEGER` | NO | Maximum token capacity supported per inference session. | `128000`, `200000`, `1048576` |
| `input_modalities` | `ARRAY<VARCHAR>` | NO | Supported input formats. | `["text", "code"]`, `["text", "code", "image", "audio", "video"]` |
| `output_modalities` | `ARRAY<VARCHAR>` | NO | Supported output formats. | `["text", "code"]`, `["text", "code", "audio"]` |
| `capabilities` | `ARRAY<VARCHAR>` | NO | Distinct algorithmic features and training objectives. | `["reasoning", "chain-of-thought", "code_generation", "tool_use"]` |
| `open_weights` | `BOOLEAN` | NO | `TRUE` if model weights are publicly downloadable; `FALSE` if API-only. | `true`, `false` |
| `pricing_input_per_m_tokens` | `FLOAT` | YES | Cost in USD per 1 Million prompt tokens (API standard). | `0.55`, `3.00`, `0.10` |
| `pricing_output_per_m_tokens`| `FLOAT` | YES | Cost in USD per 1 Million completion tokens. | `2.19`, `15.00`, `0.40` |
| `huggingface_repo_id` | `VARCHAR(128)` | YES | Upstream Hugging Face repo ID if applicable. | `"deepseek-ai/DeepSeek-R1"`, `NULL` |
| `description` | `TEXT` | NO | Curated architectural and capability summary. | `"Open reasoning foundation model trained via pure RL..."` |
| `evidence` | `STRUCT` | NO | Provenance attribution record conforming to EvidenceRecord. | (See Section 9) |

---

## 3. Benchmarks (`benchmarks`)

Standardized evaluation suites and test batteries used to measure model competence.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `benchmark_id` | `VARCHAR(64)` | NO | Unique slug for the benchmark (Primary Key). | `swe-bench-verified`, `mmlu-pro`, `math-500` |
| `name` | `VARCHAR(128)` | NO | Official benchmark title. | `"SWE-bench Verified"`, `"MATH-500"` |
| `domain` | `VARCHAR(64)` | NO | Knowledge domain or evaluation category. | `"Software Engineering"`, `"Academic Reasoning"`, `"Mathematics"` |
| `metric_name` | `VARCHAR(64)` | NO | Primary metric measured. | `"Resolved Pass Rate"`, `"Accuracy"`, `"Pass@1"` |
| `metric_unit` | `VARCHAR(16)` | NO | Display unit of measurement. | `"%"` |
| `evaluation_type` | `VARCHAR(64)` | NO | Method of verification. | `"Execution-Based Unit Testing"`, `"LLM-as-a-Judge Elo"` |
| `description` | `TEXT` | NO | Detailed description of the benchmark methodology. | `"500 human-validated GitHub issues requiring end-to-end patches..."` |
| `dataset_url` | `VARCHAR(256)` | NO | Authoritative repository or dataset link. | `"https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified"` |
| `evidence` | `STRUCT` | NO | Provenance attribution record. | (See Section 9) |

---

## 4. Evaluations (`evaluations`)

Factual performance observations linking a specific model to a specific benchmark score.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `evaluation_id` | `VARCHAR(128)` | NO | Composite identifier: `eval:{model_id}:{benchmark_id}`. | `eval:deepseek-r1:swe-bench-verified` |
| `model_id` | `VARCHAR(64)` | NO | Foreign key referencing `models.model_id`. | `deepseek-r1` |
| `benchmark_id` | `VARCHAR(64)` | NO | Foreign key referencing `benchmarks.benchmark_id`. | `swe-bench-verified` |
| `score` | `FLOAT` | NO | Numerical performance value normalized to standard metric (0.0 to 100.0). | `49.2`, `97.3`, `84.0` |
| `metric_unit` | `VARCHAR(16)` | NO | Unit of score (usually `"%"`). | `"%"` |
| `evaluation_date` | `DATE` | NO | Date the evaluation was conducted or published (YYYY-MM-DD). | `2025-01-20` |
| `evaluated_by` | `VARCHAR(128)` | NO | Entity conducting or verifying the evaluation. | `"Official Technical Report"`, `"LMSYS Community Benchmark"` |
| `is_verified` | `BOOLEAN` | NO | `TRUE` if confirmed by official paper or independent public leaderboard. | `true` |
| `evidence` | `STRUCT` | NO | Provenance attribution record. | (See Section 9) |

---

## 5. Developer Tools & Frameworks (`tools`)

Open source and commercial developer frameworks for orchestrating agents, serving inference, and structuring model output.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `tool_id` | `VARCHAR(64)` | NO | Unique slug for the tool (Primary Key). | `langgraph`, `vllm`, `crewai`, `ollama` |
| `name` | `VARCHAR(128)` | NO | Official tool name. | `"LangGraph"`, `"vLLM"`, `"CrewAI"` |
| `organization` | `VARCHAR(64)` | NO | Authoring company, lab, or open-source foundation. | `"LangChain"`, `"vLLM Project / UC Berkeley"`, `"Ollama Inc."` |
| `category` | `VARCHAR(64)` | NO | Functional classification. | `"Agent Orchestration & State Graphs"`, `"High-Throughput LLM Serving Runtime"` |
| `github_repo` | `VARCHAR(128)` | NO | Upstream GitHub repository identifier (`owner/repo`). | `"langchain-ai/langgraph"`, `"vllm-project/vllm"` |
| `github_stars` | `INTEGER` | NO | Stargazer count on GitHub. | `18950`, `42500`, `118000` |
| `license` | `VARCHAR(64)` | NO | Open source software license. | `"MIT"`, `"Apache-2.0"`, `"CC-BY-4.0"` |
| `latest_version` | `VARCHAR(32)` | NO | Most recent stable semantic release tag. | `"v0.2.74"`, `"v0.7.3"` |
| `latest_release_date` | `DATE` | NO | Date of most recent release. | `2025-01-20` |
| `supported_protocols` | `ARRAY<VARCHAR>` | NO | Standard protocols supported. | `["Model Context Protocol (MCP)", "OpenAI Tool Calling", "gRPC"]` |
| `description` | `TEXT` | NO | Purpose and technical features. | `"Cyclic state machine and graph runtime for robust multi-actor agents..."` |
| `evidence` | `STRUCT` | NO | Provenance attribution record. | (See Section 9) |

---

## 6. Research Papers (`research_papers`)

Landmark scientific publications in artificial intelligence and machine learning.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `paper_id` | `VARCHAR(64)` | NO | Canonical paper identifier (e.g. `arxiv:{id}`). | `arxiv:2501.12948`, `arxiv:1706.03762` |
| `arxiv_id` | `VARCHAR(32)` | NO | Pure arXiv numeric identifier. | `"2501.12948"`, `"1706.03762"` |
| `title` | `VARCHAR(256)` | NO | Full publication title. | `"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"` |
| `authors` | `ARRAY<VARCHAR>` | NO | List of contributing authors. | `["DeepSeek-AI", "Daya Guo", "Dejian Yang"]` |
| `primary_category` | `VARCHAR(16)` | NO | Primary arXiv classification category. | `"cs.AI"`, `"cs.CL"`, `"cs.LG"`, `"cs.SE"` |
| `published_date` | `DATE` | NO | Date published or submitted to arXiv. | `2025-01-22` |
| `canonical_url` | `VARCHAR(256)` | NO | Web landing page URL. | `"https://arxiv.org/abs/2501.12948"` |
| `pdf_url` | `VARCHAR(256)` | NO | Direct PDF download link. | `"https://arxiv.org/pdf/2501.12948.pdf"` |
| `key_contributions` | `ARRAY<TEXT>` | NO | Summary bullet points of novel technical breakthroughs. | `["Discovers emergent reflection via pure RL...", "..."]` |
| `abstract` | `TEXT` | NO | Complete published abstract text. | `"We introduce our first-generation reasoning models..."` |
| `evidence` | `STRUCT` | NO | Provenance attribution record. | (See Section 9) |

---

## 7. Relationships (`relationships`)

Directed graph edges modeling connections across models, tools, papers, and benchmarks.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `source_entity_id` | `VARCHAR(64)` | NO | Origin entity ID. | `deepseek-r1`, `langgraph`, `vllm` |
| `target_entity_id` | `VARCHAR(64)` | NO | Destination entity ID. | `arxiv:2501.12948`, `arxiv:2210.03629`, `deepseek-r1` |
| `relationship_type` | `VARCHAR(32)` | NO | Semantic edge relation (See Taxonomy below). | `INTRODUCED_IN`, `BASED_ON_METHOD`, `SERVES_MODEL`, `ORCHESTRATES` |
| `strength` | `FLOAT` | NO | Relationship confidence / coupling weight (0.0 to 1.0). | `1.0`, `0.95`, `0.85` |
| `properties` | `MAP<STRING, STRING>` | YES | Key-value attributes describing the relationship details. | `{"method": "ReAct Thought-Action Loops"}` |
| `evidence` | `STRUCT` | NO | Provenance attribution record. | (See Section 9) |

### Relationship Taxonomy & Semantics
- `INTRODUCED_IN`: A model or architecture was first described in a specific research paper (e.g. `deepseek-r1` -> `arxiv:2501.12948`).
- `BASED_ON_METHOD`: A developer tool or model implements a algorithmic method introduced by a paper (e.g. `langgraph` -> `arxiv:2210.03629` [ReAct]).
- `DISTILLED_FROM`: A smaller or dense model learned its weights from a larger teacher model (e.g. `deepseek-r1-distill-qwen-32b` -> `deepseek-r1`).
- `BUILT_ON_BASE`: A fine-tuned or aligned model was initialized from a base foundation model (e.g. `deepseek-r1` -> `deepseek-v3`).
- `SERVES_MODEL`: An inference engine or local runner provides optimized kernels for running the target model (e.g. `vllm` -> `deepseek-r1`, `ollama` -> `llama-3.3-70b-instruct`).
- `ORCHESTRATES`: An agent framework coordinates or routes tasks through the model (e.g. `langgraph` -> `claude-3-5-sonnet-20241022`).
- `OPTIMIZES_FOR`: A prompt compiler optimizes prompts targeting the model (e.g. `dspy` -> `qwen-2.5-72b-instruct`).
- `INTEGRATES_WITH`: Two tools or frameworks interoperate via an official adapter (e.g. `crewai` -> `langgraph`).

---

## 8. Recent Developments (`recent_developments`)

Chronological timeline feed items capturing breakthroughs, major version releases, and announcements.

| Column Name | Type | Nullable | Description | Example / Allowed Values |
|---|---|---|---|---|
| `event_id` | `VARCHAR(64)` | NO | Unique slug: `event:{date}:{slug}`. | `event:2025-01-20:deepseek-r1-release` |
| `title` | `VARCHAR(256)` | NO | Headline title of the event. | `"DeepSeek Unveils DeepSeek-R1 Open Reasoning Model & Weights"` |
| `category` | `VARCHAR(32)` | NO | Event category classification. | `model_release`, `framework_release`, `benchmark_record`, `research_paper` |
| `event_date` | `TIMESTAMP` | NO | ISO 8601 UTC timestamp of occurrence. | `2025-01-20T08:00:00Z` |
| `summary` | `TEXT` | NO | Concise executive briefing. | `"DeepSeek releases 671B MoE weights and 6 distilled dense models..."` |
| `primary_entity_id` | `VARCHAR(64)` | NO | Primary entity involved. | `deepseek-r1`, `vllm`, `o3-mini` |
| `tags` | `ARRAY<VARCHAR>` | NO | Faceting and topic keywords. | `["deepseek", "reasoning", "open-weights", "mit-license"]` |
| `evidence` | `STRUCT` | NO | Provenance attribution record. | (See Section 9) |

---

## 9. Evidence Record Struct (`EvidenceRecord`)

Mandatory nested provenance structure embedded in every record. Conforms strictly to `coordination/contracts/provider.py::EvidenceRecord`.

```json
{
  "source_id": "arxiv:2501.12948",
  "source_type": "arxiv | huggingface | github | announcement | benchmark_evaluation | ecosystem_graph",
  "source_url": "https://arxiv.org/abs/2501.12948",
  "publication_date": "2025-01-22T08:00:00Z",
  "ingestion_timestamp": "2026-11-08T09:47:52.896876+00:00",
  "extraction_timestamp": "2026-11-08T09:47:52.896876+00:00",
  "evidence_confidence": "PEER_REVIEWED | EVALUATED | ANNOUNCEMENT | EMPIRICAL | HYPOTHESIS",
  "raw_payload_ref": "2501.12948"
}
```

### Evidence Confidence Tiers:
- `PEER_REVIEWED`: Verified publication with formal peer review or official archival preprint (e.g. arXiv).
- `EVALUATED`: Standardized evaluation score verified against executable unit tests or public leaderboard (e.g. SWE-bench Verified, LMSYS).
- `ANNOUNCEMENT`: Official technical release announcement or lab blog post directly from the developing organization.
- `EMPIRICAL`: Observed benchmark or performance log from external independent replication.
- `HYPOTHESIS`: Theoretical projection or unverified community claim.

---

## 10. Normalization & Units Conventions

1. **Benchmark Scores:** Expressed as a percentage scale from `0.0` to `100.0`. A score of 49.2% is represented as `49.2` (never `0.492`).
2. **Parameters:** Counted in Billions ($10^9$) as a floating point number. For instance, 671 Billion is stored as `671.0`; 70 Billion as `70.0`; 14 Billion as `14.0`.
3. **Context Window:** Counted in integer number of tokens (e.g. `128000`, `200000`, `1048576`).
4. **Token Pricing:** Denominated in USD per 1 Million tokens ($/M tokens). Input and output prices are stored separately.
5. **Timestamps & Dates:** All dates adhere to ISO 8601: `YYYY-MM-DD` for date-only fields and `YYYY-MM-DDTHH:MM:SSZ` in UTC for timestamps.
