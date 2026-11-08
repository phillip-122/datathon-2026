# SIGNAL Data Sources & Acquisition Architecture

## 1. Overview & Principles

SIGNAL (AI Landscape Intelligence) tracks the fast-moving artificial intelligence ecosystem across research literature, model releases, benchmark evaluations, open developer tools, and inference runtimes.

To provide credible intelligence, every data point ingested into SIGNAL must satisfy four core principles:
1. **Verifiable Provenance:** Every model attribute, benchmark score, and relationship links directly to an authoritative origin (academic paper, official lab technical report, release tag, or verifiable test suite).
2. **Deterministic Dual-Mode Operation:** Collectors support both live HTTP API acquisition and deterministic offline mock replay, ensuring test reproducibility and zero setup friction.
3. **Lossless Bronze Storage:** Raw API responses are preserved in full with metadata envelopes before downstream normalization, ensuring zero data loss.
4. **Strict Evidence Attribution:** All records conform to the shared `EvidenceRecord` schema contract defined in [`coordination/contracts/provider.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/coordination/contracts/provider.py).

---

## 2. Ingestion Sources Catalog

### 2.1 arXiv API (Scientific Research & Landmark Papers)

- **Collector Implementation:** [`SIGNAL/ingestion/sources/arxiv_collector.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/SIGNAL/ingestion/sources/arxiv_collector.py)
- **Primary Scope:** Landmark foundational architectures, reinforcement learning recipes (e.g. DeepSeek-R1), agent methodologies (e.g. ReAct, Toolformer), and hardware kernel breakthroughs (e.g. FlashAttention-3).
- **Target Categories:** `cs.AI` (Artificial Intelligence), `cs.CL` (Computation and Language), `cs.LG` (Machine Learning), `cs.SE` (Software Engineering).
- **Endpoint:** `http://export.arxiv.org/api/query`
- **Protocol:** HTTP GET, Atom 1.0 XML response.
- **Key Query Parameters:**
  - `search_query`: Category or title query (e.g., `cat:cs.AI OR cat:cs.CL OR cat:cs.LG`, `ti:"DeepSeek-R1"`).
  - `start`: 0-indexed pagination offset.
  - `max_results`: Page size (default: 10-25).
  - `sortBy`: `submittedDate` or `relevance`.
  - `sortOrder`: `descending`.
- **Authentication:** Public; no API key required.
- **Rate Limits & Politeness:**
  - Recommended delay: At least 3 seconds between requests.
  - Header: Identifies client with custom `User-Agent`.
  - Concurrency: Serial requests only.
- **Raw Storage Location:** `SIGNAL/data/raw/arxiv/arxiv_papers_{mode}_{timestamp}.json`

### 2.2 Hugging Face Model Hub API (Weights & Metadata)

- **Collector Implementation:** [`SIGNAL/ingestion/sources/huggingface_collector.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/SIGNAL/ingestion/sources/huggingface_collector.py)
- **Primary Scope:** Open foundation model weights, community popularity metrics (likes, downloads), architecture descriptors, model card YAML frontmatter, and base model lineage.
- **Target Authors / Hub Repos:** `deepseek-ai`, `meta-llama`, `Qwen`, `mistralai`, `microsoft`, `google`.
- **Endpoints:**
  - List / Search: `https://huggingface.co/api/models`
  - Model Detail: `https://huggingface.co/api/models/{author}/{model_name}`
- **Protocol:** HTTP GET, JSON response.
- **Key Query Parameters:**
  - `search`: Keyword string.
  - `author`: Organization filter.
  - `pipeline_tag`: Task tag (e.g., `text-generation`).
  - `sort`: `downloads`, `likes`, `createdAt`.
  - `direction`: `-1` (descending).
  - `full`: `true` (requests comprehensive model card data).
- **Authentication:** Optional `HF_TOKEN` via Bearer header. Public endpoints support unauthenticated access up to standard rate thresholds.
- **Rate Limits & Resilience:**
  - Upstream limit: ~1,000 requests/hour unauthenticated; 10,000+ authenticated.
  - Fallback: Deterministic mock fixtures representing state-of-the-art models on 403/429 status codes.
- **Raw Storage Location:** `SIGNAL/data/raw/huggingface/hf_models_{mode}_{timestamp}.json`

### 2.3 GitHub REST API (Agent Frameworks & Developer Tools)

- **Collector Implementation:** [`SIGNAL/ingestion/sources/github_collector.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/SIGNAL/ingestion/sources/github_collector.py)
- **Primary Scope:** Ecosystem repositories, developer adoption (stars, forks), open issue velocity, releases, and changelogs for agent frameworks and runtime engines.
- **Core Tracked Repositories:**
  - `langchain-ai/langgraph` (State-machine agent graph runtime)
  - `crewAIInc/crewAI` (Role-playing collaborative multi-agent teams)
  - `microsoft/autogen` (Actor-based conversational agent architecture)
  - `vllm-project/vllm` (High-throughput inference serving engine)
  - `ollama/ollama` (Local model package manager & runner)
  - `run-llama/llama_index` (RAG & context augmentation framework)
  - `huggingface/smolagents` (Code-first lightweight agent framework)
  - `stanfordnlp/dspy` (Algorithmic prompt compiler & optimizer)
  - `jxnl/instructor` (Structured Pydantic validation for LLMs)
- **Endpoints:**
  - Repository Info: `https://api.github.com/repos/{owner}/{repo}`
  - Releases: `https://api.github.com/repos/{owner}/{repo}/releases`
- **Protocol:** HTTP GET, JSON response (`Accept: application/vnd.github+json`).
- **Authentication:** Optional `GITHUB_TOKEN` from environment.
- **Rate Limits & Resilience:**
  - Unauthenticated limit: 60 requests/hour per IP.
  - Authenticated limit: 5,000 requests/hour.
  - Auto-fallback: Transparent fallback to verified mock fixtures upon receiving HTTP 403 Rate Limit Exceeded.
- **Raw Storage Location:** `SIGNAL/data/raw/github/github_frameworks_{mode}_{timestamp}.json`

### 2.4 Official Technical Reports, LMSYS & Benchmark Suites (Curated Evaluations)

- **Collector / Generation Implementation:** [`SIGNAL/ingestion/generate_fixtures.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/SIGNAL/ingestion/generate_fixtures.py)
- **Primary Scope:** Factual evaluation scores from peer-reviewed publications, arXiv preprints, LMSYS Chatbot Arena, SWE-bench Verified leaderboard, and official AI lab release reports (OpenAI, DeepSeek, Anthropic, Meta, Google DeepMind, Alibaba Cloud).
- **Core Standardized Benchmarks:**
  - `swe-bench-verified`: Human-verified software engineering problem resolution (Pass@1 %).
  - `mmlu-pro`: 10-choice rigorous multidisciplinary reasoning (Accuracy %).
  - `math-500`: Advanced Olympiad/competition mathematics (Accuracy %).
  - `humaneval`: Standard Python code generation (Pass@1 %).
  - `gsm8k`: Grade-school multi-step arithmetic reasoning (Accuracy %).
  - `arena-hard`: Chatbot Arena Elo win rate against GPT-4-0314 baseline (Win Rate %).
  - `livecodebench`: Contamination-free contest coding evaluation (Pass@1 %).
  - `gpqa-diamond`: PhD-level graduate science reasoning (Accuracy %).
- **Provenance:** Every single score records `model_id`, `benchmark_id`, `score`, `evaluation_date`, `evaluated_by`, and an `EvidenceRecord` link.

---

## 3. Raw Bronze Storage Layout & Envelopes

All collectors write their output to `SIGNAL/data/raw/` in the following directory hierarchy:

```
SIGNAL/data/raw/
├── arxiv/
│   ├── arxiv_papers_live_20261108_094752.json
│   └── arxiv_papers_mock_20261108_094750.json
├── huggingface/
│   ├── hf_models_live_20261108_094753.json
│   └── hf_models_mock_20261108_094750.json
├── github/
│   ├── github_frameworks_live_20261108_094759.json
│   └── github_frameworks_mock_20261108_094750.json
└── manifest.json
```

### Standardized Raw Payload Envelope
Each raw JSON document is wrapped in a standardized bronze envelope:

```json
{
  "metadata": {
    "source_type": "arxiv | huggingface | github",
    "collection_timestamp": "2026-11-08T09:47:52.896876+00:00",
    "mode": "live | mock",
    "record_count": 10,
    "query_parameters": {
      "query": "cat:cs.AI OR cat:cs.CL OR cat:cs.LG",
      "max_results": 10
    },
    "collector_version": "1.0.0"
  },
  "records": [
    { ... }
  ]
}
```

### Ingestion Manifest
The unified orchestrator ([`SIGNAL/ingestion/sources/collect_all.py`](file:///Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/SIGNAL/ingestion/sources/collect_all.py)) writes `SIGNAL/data/raw/manifest.json` after every run, recording:
- Pipeline run start and finish UTC timestamps.
- Overall execution duration.
- Per-source execution status (`SUCCESS` / `ERROR`).
- Extracted record counts and target raw file paths.

---

## 4. Evidence Record Attribution Contract

In accordance with Section 3 of `coordination/INTEGRATION_CONTRACT.md`, every individual entity and factual observation attaches an `evidence` record:

| Field | Type | Description | Example |
|---|---|---|---|
| `source_id` | string | Unique primary key of the source origin | `arxiv:2501.12948`, `hf:deepseek-ai/DeepSeek-R1` |
| `source_type` | string | Origin platform name | `arxiv`, `huggingface`, `github`, `announcement` |
| `source_url` | string | Canonical resolvable web URL | `https://arxiv.org/abs/2501.12948` |
| `publication_date`| string | Official publication or release ISO timestamp | `2025-01-20T08:00:00Z` |
| `ingestion_timestamp`| string | UTC timestamp when raw payload was ingested | `2026-11-08T09:47:52.896876+00:00` |
| `extraction_timestamp`| string | UTC timestamp when entities were normalized | `2026-11-08T09:47:52.896876+00:00` |
| `evidence_confidence`| string | Confidence level tier | `PEER_REVIEWED`, `EVALUATED`, `ANNOUNCEMENT`, `HYPOTHESIS` |
| `raw_payload_ref` | string | Raw identifier or bronze payload hash | `2501.12948`, `deepseek-ai/DeepSeek-R1` |

---

## 5. Collector CLI Usage Guide

Run all collectors in dry-run/mock mode:
```bash
.venv/bin/python -m SIGNAL.ingestion.sources.collect_all --mode mock
```

Run all collectors in live mode:
```bash
.venv/bin/python -m SIGNAL.ingestion.sources.collect_all --mode live
```

Run individual collectors:
```bash
# arXiv Collector
.venv/bin/python -m SIGNAL.ingestion.sources.arxiv_collector --mode live --query "ti:DeepSeek-R1" --max-results 5

# Hugging Face Collector
.venv/bin/python -m SIGNAL.ingestion.sources.huggingface_collector --mode live --author meta-llama --limit 10

# GitHub Collector
.venv/bin/python -m SIGNAL.ingestion.sources.github_collector --mode live --repos langchain-ai/langgraph crewAIInc/crewAI
```

Regenerate the comprehensive offline fixture dataset:
```bash
.venv/bin/python SIGNAL/ingestion/generate_fixtures.py
```
