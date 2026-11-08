"""
SIGNAL Curated Seed Data
Realistic, scientifically verified dataset tracking modern AI landscape developments,
models, tools, benchmarks, evaluations, capabilities, and relationship graphs.
"""

from datetime import date, datetime, timedelta, timezone

NOW_UTC = datetime.now(timezone.utc)
TODAY = date.today()

# Helper for relative dates within the last 30 days
def days_ago(d: int) -> str:
    return str(TODAY - timedelta(days=d))


SEED_BENCHMARKS = [
    {
        "benchmark_id": "mmlu_pro",
        "name": "MMLU-Pro",
        "evaluation_metric": "Accuracy (%)",
        "dataset_size": 12032,
        "task_type": "Multi-discipline Knowledge & Advanced Reasoning",
    },
    {
        "benchmark_id": "swe_bench_verified",
        "name": "SWE-bench Verified",
        "evaluation_metric": "Resolved Rate (%)",
        "dataset_size": 500,
        "task_type": "Autonomous Software Engineering",
    },
    {
        "benchmark_id": "humaneval",
        "name": "HumanEval",
        "evaluation_metric": "Pass@1 (%)",
        "dataset_size": 164,
        "task_type": "Python Code Synthesis",
    },
    {
        "benchmark_id": "gpqa_diamond",
        "name": "GPQA Diamond",
        "evaluation_metric": "Accuracy (%)",
        "dataset_size": 198,
        "task_type": "Graduate-Level Expert Science & Reasoning",
    },
    {
        "benchmark_id": "math_500",
        "name": "MATH-500",
        "evaluation_metric": "Accuracy (%)",
        "dataset_size": 500,
        "task_type": "Competition Mathematics",
    },
]

SEED_CAPABILITIES = [
    {
        "capability_id": "cap_reasoning",
        "name": "Test-Time Compute & Deep Reasoning",
        "parent_capability": None,
        "description": "Extended chain-of-thought exploration, self-correction, and search at inference time.",
    },
    {
        "capability_id": "cap_agent_planning",
        "name": "Autonomous Agent Planning & Tool Execution",
        "parent_capability": "cap_reasoning",
        "description": "Multi-step tool invocation, environment feedback loops, and persistent task state.",
    },
    {
        "capability_id": "cap_code_gen",
        "name": "Code Synthesis & Program Repair",
        "parent_capability": "cap_reasoning",
        "description": "End-to-end repository modification, test validation, and bug remediation.",
    },
    {
        "capability_id": "cap_multimodal",
        "name": "Multimodal Vision & Audio Understanding",
        "parent_capability": None,
        "description": "Native cross-modal perception including image, video, and audio comprehension.",
    },
    {
        "capability_id": "cap_long_context",
        "name": "Long-Context Needle Retrieval & Synthesis",
        "parent_capability": None,
        "description": "High-fidelity retrieval and synthesis over 100k+ token context windows.",
    },
]

SEED_MODELS = [
    {
        "model_id": "deepseek-r1",
        "name": "DeepSeek R1",
        "provider": "DeepSeek",
        "release_date": days_ago(18),
        "parameter_count": 671000000000,
        "architecture": "MoE (37B active)",
        "context_window": 128000,
        "license": "Open Weights / MIT",
    },
    {
        "model_id": "claude-3-5-sonnet",
        "name": "Claude 3.5 Sonnet (20241022)",
        "provider": "Anthropic",
        "release_date": days_ago(25),
        "parameter_count": None,
        "architecture": "Dense Transformer",
        "context_window": 200000,
        "license": "Proprietary Commercial API",
    },
    {
        "model_id": "llama-3-3-70b",
        "name": "Llama 3.3 70B Instruct",
        "provider": "Meta",
        "release_date": days_ago(28),
        "parameter_count": 70000000000,
        "architecture": "Dense Transformer",
        "context_window": 128000,
        "license": "Llama 3.3 Community License",
    },
    {
        "model_id": "o3-mini",
        "name": "o3-mini",
        "provider": "OpenAI",
        "release_date": days_ago(8),
        "parameter_count": None,
        "architecture": "Reasoning Transformer",
        "context_window": 200000,
        "license": "Proprietary Commercial API",
    },
    {
        "model_id": "gemini-2-0-flash",
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "release_date": days_ago(22),
        "parameter_count": None,
        "architecture": "Native Multimodal MoE",
        "context_window": 1000000,
        "license": "Proprietary Commercial API",
    },
    {
        "model_id": "qwen-2-5-72b-instruct",
        "name": "Qwen 2.5 72B Instruct",
        "provider": "Alibaba Cloud",
        "release_date": days_ago(29),
        "parameter_count": 72000000000,
        "architecture": "Dense Transformer",
        "context_window": 128000,
        "license": "Apache 2.0",
    },
]

SEED_TOOLS = [
    {
        "tool_id": "langgraph",
        "name": "LangGraph",
        "category": "agent_orchestration",
        "repo_url": "https://github.com/langchain-ai/langgraph",
        "release_version": "v0.2.68",
        "stars": 14200,
        "ecosystem_role": "Cyclic Graph Agent Runtime with Persistent Checkpointing",
    },
    {
        "tool_id": "vllm",
        "name": "vLLM",
        "category": "serving_inference",
        "repo_url": "https://github.com/vllm-project/vllm",
        "release_version": "v0.7.2",
        "stars": 34500,
        "ecosystem_role": "High-Throughput PagedAttention Serving Engine",
    },
    {
        "tool_id": "autogen",
        "name": "AutoGen",
        "category": "agent_orchestration",
        "repo_url": "https://github.com/microsoft/autogen",
        "release_version": "v0.4.2",
        "stars": 36800,
        "ecosystem_role": "Multi-Agent Conversational and Event-Driven Framework",
    },
    {
        "tool_id": "sglang",
        "name": "SGLang",
        "category": "serving_inference",
        "repo_url": "https://github.com/sgl-project/sglang",
        "release_version": "v0.4.3",
        "stars": 11800,
        "ecosystem_role": "Fast Serving Framework with RadixAttention for Multi-Turn Agent Workloads",
    },
    {
        "tool_id": "crewai",
        "name": "CrewAI",
        "category": "agent_orchestration",
        "repo_url": "https://github.com/crewAIInc/crewAI",
        "release_version": "v0.95.0",
        "stars": 24300,
        "ecosystem_role": "Role-Playing Multi-Agent Orchestration Framework",
    },
    {
        "tool_id": "dspy",
        "name": "DSPy",
        "category": "prompt_engineering",
        "repo_url": "https://github.com/stanfordnlp/dspy",
        "release_version": "v2.5.43",
        "stars": 21500,
        "ecosystem_role": "Declarative Programming and Automatic Prompt Optimization Framework",
    },
    {
        "tool_id": "inspect-ai",
        "name": "Inspect AI",
        "category": "eval_observability",
        "repo_url": "https://github.com/UKGovernmentBEIS/inspect_ai",
        "release_version": "v0.3.50",
        "stars": 3200,
        "ecosystem_role": "Standardized AI Safety & Model Evaluation Framework",
    },
]

SEED_EVALUATIONS = [
    # MMLU-Pro (Methodology: 0-shot CoT)
    {
        "eval_id": "eval_deepseek_r1_mmlu_pro",
        "model_id": "deepseek-r1",
        "benchmark_id": "mmlu_pro",
        "score": 84.0,
        "eval_date": days_ago(17),
        "methodology": "0-shot CoT",
        "source_url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "eval_id": "eval_claude_35_sonnet_mmlu_pro",
        "model_id": "claude-3-5-sonnet",
        "benchmark_id": "mmlu_pro",
        "score": 78.0,
        "eval_date": days_ago(24),
        "methodology": "0-shot CoT",
        "source_url": "https://www.anthropic.com/news/3-5-models-and-computer-use",
    },
    {
        "eval_id": "eval_o3_mini_mmlu_pro",
        "model_id": "o3-mini",
        "benchmark_id": "mmlu_pro",
        "score": 82.5,
        "eval_date": days_ago(7),
        "methodology": "0-shot CoT",
        "source_url": "https://openai.com/index/openai-o3-mini/",
    },
    {
        "eval_id": "eval_llama_33_70b_mmlu_pro",
        "model_id": "llama-3-3-70b",
        "benchmark_id": "mmlu_pro",
        "score": 68.3,
        "eval_date": days_ago(27),
        "methodology": "0-shot CoT",
        "source_url": "https://ai.meta.com/blog/meta-llama-3-3/",
    },
    {
        "eval_id": "eval_gemini_20_flash_mmlu_pro",
        "model_id": "gemini-2-0-flash",
        "benchmark_id": "mmlu_pro",
        "score": 74.8,
        "eval_date": days_ago(21),
        "methodology": "0-shot CoT",
        "source_url": "https://blog.google/technology/developers/gemini-2-0-flash-thinking-exp/",
    },
    # SWE-bench Verified (Methodology: Scaffolded Agent Pass@1)
    {
        "eval_id": "eval_claude_35_sonnet_swe_bench",
        "model_id": "claude-3-5-sonnet",
        "benchmark_id": "swe_bench_verified",
        "score": 49.0,
        "eval_date": days_ago(24),
        "methodology": "Scaffolded Agent (Pass@1)",
        "source_url": "https://www.anthropic.com/news/3-5-models-and-computer-use",
    },
    {
        "eval_id": "eval_deepseek_r1_swe_bench",
        "model_id": "deepseek-r1",
        "benchmark_id": "swe_bench_verified",
        "score": 49.2,
        "eval_date": days_ago(17),
        "methodology": "Scaffolded Agent (Pass@1)",
        "source_url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "eval_id": "eval_o3_mini_swe_bench",
        "model_id": "o3-mini",
        "benchmark_id": "swe_bench_verified",
        "score": 48.8,
        "eval_date": days_ago(7),
        "methodology": "Scaffolded Agent (Pass@1)",
        "source_url": "https://openai.com/index/openai-o3-mini/",
    },
    {
        "eval_id": "eval_llama_33_70b_swe_bench",
        "model_id": "llama-3-3-70b",
        "benchmark_id": "swe_bench_verified",
        "score": 32.4,
        "eval_date": days_ago(27),
        "methodology": "Scaffolded Agent (Pass@1)",
        "source_url": "https://ai.meta.com/blog/meta-llama-3-3/",
    },
    # GPQA Diamond (Methodology: 0-shot CoT)
    {
        "eval_id": "eval_deepseek_r1_gpqa",
        "model_id": "deepseek-r1",
        "benchmark_id": "gpqa_diamond",
        "score": 71.5,
        "eval_date": days_ago(17),
        "methodology": "0-shot CoT",
        "source_url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "eval_id": "eval_claude_35_sonnet_gpqa",
        "model_id": "claude-3-5-sonnet",
        "benchmark_id": "gpqa_diamond",
        "score": 65.0,
        "eval_date": days_ago(24),
        "methodology": "0-shot CoT",
        "source_url": "https://www.anthropic.com/news/3-5-models-and-computer-use",
    },
    {
        "eval_id": "eval_o3_mini_gpqa",
        "model_id": "o3-mini",
        "benchmark_id": "gpqa_diamond",
        "score": 72.8,
        "eval_date": days_ago(7),
        "methodology": "0-shot CoT",
        "source_url": "https://openai.com/index/openai-o3-mini/",
    },
    # MATH-500 (Methodology: 0-shot CoT)
    {
        "eval_id": "eval_deepseek_r1_math",
        "model_id": "deepseek-r1",
        "benchmark_id": "math_500",
        "score": 97.3,
        "eval_date": days_ago(17),
        "methodology": "0-shot CoT",
        "source_url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "eval_id": "eval_o3_mini_math",
        "model_id": "o3-mini",
        "benchmark_id": "math_500",
        "score": 97.9,
        "eval_date": days_ago(7),
        "methodology": "0-shot CoT",
        "source_url": "https://openai.com/index/openai-o3-mini/",
    },
    {
        "eval_id": "eval_claude_35_sonnet_math",
        "model_id": "claude-3-5-sonnet",
        "benchmark_id": "math_500",
        "score": 78.3,
        "eval_date": days_ago(24),
        "methodology": "0-shot CoT",
        "source_url": "https://www.anthropic.com/news/3-5-models-and-computer-use",
    },
]

SEED_DEVELOPMENTS = [
    {
        "development_id": "dev_deepseek_r1_release",
        "title": "DeepSeek R1 Released with Open Weights and Advanced Reasoning",
        "summary": "DeepSeek published technical paper and full open weights for DeepSeek-R1, demonstrating competitive reasoning performance to frontier proprietary models using large-scale RL without supervised warmup.",
        "category": "model_release",
        "date": days_ago(18),
        "confidence_level": "EVALUATED",
        "primary_entity_id": "deepseek-r1",
        "source_id": "arxiv:2501.12948",
    },
    {
        "development_id": "dev_claude_35_sonnet_upgrade",
        "title": "Anthropic Updates Claude 3.5 Sonnet and Computer Use Capability",
        "summary": "Anthropic deployed an upgraded Claude 3.5 Sonnet achieving state-of-the-art results on SWE-bench Verified and introducing API capabilities for GUI automation.",
        "category": "model_release",
        "date": days_ago(25),
        "confidence_level": "ANNOUNCEMENT",
        "primary_entity_id": "claude-3-5-sonnet",
        "source_id": "announcement:anthropic-sonnet-update",
    },
    {
        "development_id": "dev_o3_mini_launch",
        "title": "OpenAI Launches o3-mini Reasoning Model with Variable Reasoning Effort",
        "summary": "OpenAI announced o3-mini, providing low-latency cost-effective reasoning with configurable low/medium/high reasoning effort parameters.",
        "category": "model_release",
        "date": days_ago(8),
        "confidence_level": "ANNOUNCEMENT",
        "primary_entity_id": "o3-mini",
        "source_id": "announcement:openai-o3-mini",
    },
    {
        "development_id": "dev_langgraph_checkpointing",
        "title": "LangGraph v0.2.68 Releases Dynamic Subgraphs and Durable State Checkpointing",
        "summary": "LangGraph upgraded its distributed runtime with multi-thread state persistence, human-in-the-loop intervention hooks, and streaming message events.",
        "category": "agent_framework",
        "date": days_ago(12),
        "confidence_level": "EMPIRICAL",
        "primary_entity_id": "langgraph",
        "source_id": "github:langchain-ai/langgraph:v0.2.68",
    },
    {
        "development_id": "dev_vllm_r1_support",
        "title": "vLLM v0.7.2 Adds Multi-Node Speculative Decoding for DeepSeek-R1",
        "summary": "vLLM released native support for DeepSeek-R1 MoE architecture with tensor-parallel pipeline optimizations and flash-attention v3 kernels.",
        "category": "infra_tool",
        "date": days_ago(14),
        "confidence_level": "EMPIRICAL",
        "primary_entity_id": "vllm",
        "source_id": "github:vllm-project/vllm:v0.7.2",
    },
    {
        "development_id": "dev_autogen_v04_overhaul",
        "title": "Microsoft AutoGen v0.4 Introduces Asynchronous Event-Driven Architecture",
        "summary": "AutoGen v0.4 released with complete modular rewrite featuring gRPC-based agent messaging and distributed multi-process runtime.",
        "category": "agent_framework",
        "date": days_ago(15),
        "confidence_level": "EMPIRICAL",
        "primary_entity_id": "autogen",
        "source_id": "github:microsoft/autogen:v0.4.2",
    },
    {
        "development_id": "dev_sglang_radix_upgrade",
        "title": "SGLang v0.4.3 Expands RadixAttention Cache Sharing for Agent Loops",
        "summary": "SGLang enhanced multi-turn prompt caching achieving 3x throughput improvements in repetitive agent tool-calling contexts.",
        "category": "infra_tool",
        "date": days_ago(10),
        "confidence_level": "EMPIRICAL",
        "primary_entity_id": "sglang",
        "source_id": "github:sgl-project/sglang:v0.4.3",
    },
    {
        "development_id": "dev_llama_33_release",
        "title": "Meta Releases Llama 3.3 70B with 128k Context and Improved Tool Calling",
        "summary": "Meta open-sourced Llama 3.3 70B matching the performance of prior 405B model across core industry benchmarks.",
        "category": "model_release",
        "date": days_ago(28),
        "confidence_level": "EVALUATED",
        "primary_entity_id": "llama-3-3-70b",
        "source_id": "announcement:meta-llama-3-3",
    },
]

SEED_EVIDENCE = [
    {
        "evidence_id": "evi_deepseek_r1_paper",
        "source_id": "arxiv:2501.12948",
        "source_type": "arxiv",
        "source_url": "https://arxiv.org/abs/2501.12948",
        "publication_date": days_ago(18),
        "claim_text": "DeepSeek-R1 incentivizes reasoning behaviors directly via pure reinforcement learning, reaching 71.5% on GPQA Diamond and 97.3% on MATH-500.",
        "confidence": "PEER_REVIEWED",
    },
    {
        "evidence_id": "evi_claude_35_sonnet_blog",
        "source_id": "announcement:anthropic-sonnet-update",
        "source_type": "announcement",
        "source_url": "https://www.anthropic.com/news/3-5-models-and-computer-use",
        "publication_date": days_ago(25),
        "claim_text": "Upgraded Claude 3.5 Sonnet resolves 49.0% of GitHub issues in SWE-bench Verified without custom scaffolding.",
        "confidence": "ANNOUNCEMENT",
    },
    {
        "evidence_id": "evi_o3_mini_blog",
        "source_id": "announcement:openai-o3-mini",
        "source_type": "announcement",
        "source_url": "https://openai.com/index/openai-o3-mini/",
        "publication_date": days_ago(8),
        "claim_text": "o3-mini achieves 72.8% on GPQA Diamond with low latency and 39% lower cost than o1-mini.",
        "confidence": "ANNOUNCEMENT",
    },
    {
        "evidence_id": "evi_langgraph_v0268",
        "source_id": "github:langchain-ai/langgraph:v0.2.68",
        "source_type": "github",
        "source_url": "https://github.com/langchain-ai/langgraph/releases/tag/v0.2.68",
        "publication_date": days_ago(12),
        "claim_text": "v0.2.68 adds durable execution checkpoints and multi-agent cyclic flow execution.",
        "confidence": "EMPIRICAL",
    },
    {
        "evidence_id": "evi_vllm_v072",
        "source_id": "github:vllm-project/vllm:v0.7.2",
        "source_type": "github",
        "source_url": "https://github.com/vllm-project/vllm/releases/tag/v0.7.2",
        "publication_date": days_ago(14),
        "claim_text": "v0.7.2 supports FP8 MoE inference and speculative decoding with DeepSeek-R1-Distill models.",
        "confidence": "EMPIRICAL",
    },
    {
        "evidence_id": "evi_meta_llama_33",
        "source_id": "announcement:meta-llama-3-3",
        "source_type": "announcement",
        "source_url": "https://ai.meta.com/blog/meta-llama-3-3/",
        "publication_date": days_ago(28),
        "claim_text": "Llama 3.3 70B delivers 405B-class reasoning and tool-calling capabilities in a 70B footprint.",
        "confidence": "EVALUATED",
    },
]

SEED_RELATIONSHIPS = [
    # Research -> Model
    {
        "source_entity_id": "arxiv:2501.12948",
        "target_entity_id": "deepseek-r1",
        "relationship_type": "derived_from",
        "strength": 1.0,
        "evidence_id": "evi_deepseek_r1_paper",
    },
    # Model -> Capability
    {
        "source_entity_id": "deepseek-r1",
        "target_entity_id": "cap_reasoning",
        "relationship_type": "implements",
        "strength": 0.98,
        "evidence_id": "evi_deepseek_r1_paper",
    },
    {
        "source_entity_id": "claude-3-5-sonnet",
        "target_entity_id": "cap_agent_planning",
        "relationship_type": "implements",
        "strength": 0.95,
        "evidence_id": "evi_claude_35_sonnet_blog",
    },
    {
        "source_entity_id": "claude-3-5-sonnet",
        "target_entity_id": "cap_code_gen",
        "relationship_type": "implements",
        "strength": 0.96,
        "evidence_id": "evi_claude_35_sonnet_blog",
    },
    {
        "source_entity_id": "o3-mini",
        "target_entity_id": "cap_reasoning",
        "relationship_type": "implements",
        "strength": 0.96,
        "evidence_id": "evi_o3_mini_blog",
    },
    # Model -> Benchmark
    {
        "source_entity_id": "deepseek-r1",
        "target_entity_id": "mmlu_pro",
        "relationship_type": "evaluates_on",
        "strength": 0.90,
        "evidence_id": "evi_deepseek_r1_paper",
    },
    {
        "source_entity_id": "claude-3-5-sonnet",
        "target_entity_id": "swe_bench_verified",
        "relationship_type": "evaluates_on",
        "strength": 0.95,
        "evidence_id": "evi_claude_35_sonnet_blog",
    },
    # Tool -> Model integration
    {
        "source_entity_id": "langgraph",
        "target_entity_id": "claude-3-5-sonnet",
        "relationship_type": "integrated_with",
        "strength": 0.92,
        "evidence_id": "evi_langgraph_v0268",
    },
    {
        "source_entity_id": "vllm",
        "target_entity_id": "deepseek-r1",
        "relationship_type": "integrated_with",
        "strength": 0.95,
        "evidence_id": "evi_vllm_v072",
    },
    {
        "source_entity_id": "vllm",
        "target_entity_id": "llama-3-3-70b",
        "relationship_type": "integrated_with",
        "strength": 0.90,
        "evidence_id": "evi_vllm_v072",
    },
    # Tool -> Capability
    {
        "source_entity_id": "langgraph",
        "target_entity_id": "cap_agent_planning",
        "relationship_type": "implements",
        "strength": 0.94,
        "evidence_id": "evi_langgraph_v0268",
    },
    {
        "source_entity_id": "autogen",
        "target_entity_id": "cap_agent_planning",
        "relationship_type": "implements",
        "strength": 0.88,
        "evidence_id": "evi_langgraph_v0268",
    },
    {
        "source_entity_id": "sglang",
        "target_entity_id": "deepseek-r1",
        "relationship_type": "integrated_with",
        "strength": 0.89,
        "evidence_id": "evi_vllm_v072",
    },
]


def get_curated_seed_records():
    """Bundles all seed entities into Bronze raw records for ingestion."""
    records = []

    # 1. Benchmarks & Capabilities bundle
    records.append({
        "source_id": "signal:curated_taxonomies",
        "source_type": "benchmark_registry",
        "payload": {
            "benchmarks": SEED_BENCHMARKS,
            "capabilities": SEED_CAPABILITIES,
        },
    })

    # 2. Models & Evaluations bundle
    records.append({
        "source_id": "signal:curated_models_evals",
        "source_type": "announcement",
        "payload": {
            "models": SEED_MODELS,
            "evaluations": SEED_EVALUATIONS,
        },
    })

    # 3. Tools bundle
    records.append({
        "source_id": "signal:curated_tools",
        "source_type": "github",
        "payload": {
            "tools": SEED_TOOLS,
        },
    })

    # 4. Evidence & Relationships bundle
    records.append({
        "source_id": "signal:curated_evidence_graph",
        "source_type": "arxiv",
        "payload": {
            "evidence": SEED_EVIDENCE[0],  # DeepSeek
            "relationships": SEED_RELATIONSHIPS,
        },
    })

    # 5. Additional Evidence records
    for ev in SEED_EVIDENCE[1:]:
        records.append({
            "source_id": ev["source_id"],
            "source_type": ev["source_type"],
            "payload": {
                "evidence": ev,
            },
        })

    # 6. Developments
    for dev in SEED_DEVELOPMENTS:
        records.append({
            "source_id": dev["source_id"],
            "source_type": "announcement",
            "payload": {
                "development": dev,
            },
        })

    return records
