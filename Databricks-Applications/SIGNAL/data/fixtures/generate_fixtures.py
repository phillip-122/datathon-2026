"""
Generate comprehensive offline fixtures for SIGNAL AI Landscape Observatory.
Ensures strict compliance with EvidenceRecord and DataProvider contracts.
"""

import json
from pathlib import Path

fixtures = {
    "metadata": {
        "version": "1.0.0",
        "generated_at": "2025-02-01T12:00:00Z",
        "description": "Curated realistic AI landscape dataset tracking developments, models, tools, research, and benchmarks across 30 days."
    },
    "metrics": {
        "active_models": 16,
        "tracked_tools": 24,
        "benchmark_evaluations": 112,
        "developments_30d": 21,
        "velocity_change_pct": 38.5,
        "confidence_distribution": {
            "EVALUATED": 42,
            "EMPIRICAL": 36,
            "PEER_REVIEWED": 22,
            "ANNOUNCEMENT": 12
        },
        "category_distribution": {
            "Models": 7,
            "Tools": 8,
            "Research": 6,
            "Benchmarks": 4,
            "Infrastructure": 3
        }
    },
    "recent_developments": [
        {
            "id": "dev-deepseek-r1",
            "title": "DeepSeek-R1 Released: Open Large-Scale Reasoning Model via Pure RL",
            "summary": "DeepSeek introduces R1 and R1-Zero, demonstrating that reasoning behaviors like reflection and verification can emerge through pure large-scale reinforcement learning without supervised fine-tuning (SFT). Weights published under MIT license with distilled 1.5B to 70B variants.",
            "category": "Research",
            "date": "2025-01-20",
            "provider_or_author": "DeepSeek AI",
            "evidence_confidence": "EVALUATED",
            "impact_score": 9.8,
            "tags": ["Reasoning", "Reinforcement Learning", "Open Weights", "Distillation"],
            "related_entities": ["model-deepseek-r1", "paper-deepseek-r1", "tool-vllm"],
            "evidence_id": "ev-arxiv-2501-12948"
        },
        {
            "id": "dev-claude-3-5-sonnet-upgrade",
            "title": "Anthropic Upgrades Claude 3.5 Sonnet with Computer Use & SOTA Coding",
            "summary": "Updated Claude 3.5 Sonnet shows significant gains across coding benchmarks, reaching 49.0% on SWE-bench Verified. Introduces revolutionary 'Computer Use' API enabling direct OS interaction, keystrokes, and cursor movement in a sandboxed environment.",
            "category": "Models",
            "date": "2024-10-22",
            "provider_or_author": "Anthropic",
            "evidence_confidence": "EVALUATED",
            "impact_score": 9.5,
            "tags": ["Computer Use", "Coding", "Autonomous Agents", "API"],
            "related_entities": ["model-claude-3-5-sonnet", "tool-mcp", "bench-swe-bench-verified"],
            "evidence_id": "ev-anthropic-claude-3-5"
        },
        {
            "id": "dev-mcp-anthropic",
            "title": "Anthropic Open-Sources Model Context Protocol (MCP) Standard",
            "summary": "Anthropic releases MCP, an open architecture standard connecting AI models to external data sources and development tools securely. Swiftly adopted by Cursor, Claude Desktop, and developer tooling communities.",
            "category": "Tools",
            "date": "2024-11-25",
            "provider_or_author": "Anthropic Open Source",
            "evidence_confidence": "EMPIRICAL",
            "impact_score": 9.2,
            "tags": ["Protocol", "Standardization", "Developer Tools", "Agent Architecture"],
            "related_entities": ["tool-mcp", "model-claude-3-5-sonnet", "tool-langgraph"],
            "evidence_id": "ev-github-mcp-spec"
        },
        {
            "id": "dev-o3-mini-reasoning",
            "title": "OpenAI Releases o3-mini Reasoning Model with Effort Toggles",
            "summary": "OpenAI rolls out o3-mini, delivering STEM and competitive coding performance comparable to o1 at 63% lower latency. Introduces low, medium, and high reasoning effort parameters to balance cost and deliberate compute.",
            "category": "Models",
            "date": "2025-01-31",
            "provider_or_author": "OpenAI",
            "evidence_confidence": "ANNOUNCEMENT",
            "impact_score": 9.1,
            "tags": ["Reasoning", "STEM", "Low Latency", "Test-time Compute"],
            "related_entities": ["model-o3-mini", "bench-math-500", "bench-gpqa-diamond"],
            "evidence_id": "ev-openai-o3-mini"
        },
        {
            "id": "dev-smolagents",
            "title": "Hugging Face Launches Smolagents: Lightweight Code-Centric Agents",
            "summary": "Hugging Face unveils smolagents, a compact library advocating CodeAgents (where agents write Python snippets rather than verbose JSON tool schemas). Demonstrates 30% fewer tool invocation tokens on GAIA benchmark.",
            "category": "Tools",
            "date": "2025-01-08",
            "provider_or_author": "Hugging Face",
            "evidence_confidence": "EVALUATED",
            "impact_score": 8.7,
            "tags": ["CodeAgent", "Python", "Tool Calling", "Hugging Face"],
            "related_entities": ["tool-smolagents", "model-deepseek-r1", "bench-gaia"],
            "evidence_id": "ev-github-smolagents"
        },
        {
            "id": "dev-langgraph-0-2",
            "title": "LangGraph 0.2: Durable Multi-Agent State Machine & Human-in-the-Loop",
            "summary": "LangChain releases LangGraph 0.2 with checkpointing backends (PostgreSQL, SQLite), cyclical graph architectures, and fine-grained state time-travel for enterprise production workflows.",
            "category": "Tools",
            "date": "2024-12-12",
            "provider_or_author": "LangChain",
            "evidence_confidence": "EMPIRICAL",
            "impact_score": 8.6,
            "tags": ["Multi-Agent", "Durable Execution", "State Machine", "Workflows"],
            "related_entities": ["tool-langgraph", "tool-crewai", "tool-mcp"],
            "evidence_id": "ev-langchain-langgraph-02"
        },
        {
            "id": "dev-swe-bench-verified",
            "title": "SWE-bench Verified Released: 500 Human-Validated Software Fixes",
            "summary": "Princeton NLP and SWE-bench team release 'SWE-bench Verified', filtering down 2,294 problems to 500 rigorously validated issues to eliminate unresolvable unit tests and metric noise.",
            "category": "Benchmarks",
            "date": "2024-08-15",
            "provider_or_author": "Princeton NLP & OpenAI",
            "evidence_confidence": "PEER_REVIEWED",
            "impact_score": 9.4,
            "tags": ["Benchmark", "Software Engineering", "Evaluation Rigor"],
            "related_entities": ["bench-swe-bench-verified", "model-claude-3-5-sonnet", "model-deepseek-r1"],
            "evidence_id": "ev-arxiv-swe-bench-verified"
        },
        {
            "id": "dev-autogen-0-4",
            "title": "Microsoft AutoGen 0.4 Rebuilt on Asynchronous Actor Framework",
            "summary": "Microsoft announces comprehensive architectural overhaul of AutoGen in 0.4. Rebuilt as an asynchronous, event-driven actor framework supporting distributed agents across networked services.",
            "category": "Tools",
            "date": "2025-01-15",
            "provider_or_author": "Microsoft Research",
            "evidence_confidence": "EMPIRICAL",
            "impact_score": 8.4,
            "tags": ["Multi-Agent", "Actor Model", "Event-Driven", "Distributed"],
            "related_entities": ["tool-autogen", "tool-langgraph"],
            "evidence_id": "ev-github-autogen-04"
        },
        {
            "id": "dev-gemini-2-flash",
            "title": "Google Releases Gemini 2.0 Flash with Native Multimodal Streaming",
            "summary": "Google announces Gemini 2.0 Flash featuring sub-500ms bidirectional audio and video streaming, built-in tool execution, and an expanded 1-million-token context window.",
            "category": "Models",
            "date": "2024-12-11",
            "provider_or_author": "Google DeepMind",
            "evidence_confidence": "ANNOUNCEMENT",
            "impact_score": 8.9,
            "tags": ["Multimodal", "Streaming", "Real-Time", "Long Context"],
            "related_entities": ["model-gemini-2-flash", "bench-human-eval"],
            "evidence_id": "ev-google-gemini-2"
        },
        {
            "id": "dev-browser-use",
            "title": "Browser-Use Library Emerges as Standard for Autonomous Web Agents",
            "summary": "Open-source Browser-Use framework hits 20k GitHub stars. Utilizes vision-language models and Playwright to navigate complex dynamic DOMs, handle authentication flows, and extract structured web data.",
            "category": "Tools",
            "date": "2024-12-28",
            "provider_or_author": "Browser-Use Team",
            "evidence_confidence": "EMPIRICAL",
            "impact_score": 8.3,
            "tags": ["Web Automation", "Playwright", "Vision Agents"],
            "related_entities": ["tool-browser-use", "model-claude-3-5-sonnet"],
            "evidence_id": "ev-github-browser-use"
        },
        {
            "id": "dev-llama-3-3-70b",
            "title": "Meta Releases Llama 3.3 70B: 405B-Class Capabilities in 70B Footprint",
            "summary": "Meta releases Llama 3.3 70B Instruct. Through improved synthetic data generation and knowledge distillation from 405B, achieves competitive benchmark parity with the flagship model at 1/5th serving cost.",
            "category": "Models",
            "date": "2024-12-06",
            "provider_or_author": "Meta AI",
            "evidence_confidence": "EVALUATED",
            "impact_score": 8.8,
            "tags": ["Open Weights", "Distillation", "Efficiency", "Self-Hostable"],
            "related_entities": ["model-llama-3-3-70b", "tool-vllm", "bench-mmlu-pro"],
            "evidence_id": "ev-meta-llama-3-3"
        },
        {
            "id": "dev-vllm-speculative",
            "title": "vLLM Adds Speculative Decoding & MLA Support for DeepSeek-R1",
            "summary": "vLLM v0.7.0 adds native Multi-head Latent Attention (MLA) and speculative drafting for DeepSeek-R1 and Qwen reasoning models, tripling token generation throughput on dual H100s.",
            "category": "Infrastructure",
            "date": "2025-01-25",
            "provider_or_author": "vLLM Team",
            "evidence_confidence": "EMPIRICAL",
            "impact_score": 8.7,
            "tags": ["Inference", "MLA", "High Throughput", "Hardware Optimization"],
            "related_entities": ["tool-vllm", "model-deepseek-r1", "model-qwen-2-5-coder-32b"],
            "evidence_id": "ev-github-vllm-070"
        },
        {
            "id": "dev-dspy-2-5",
            "title": "Stanford DSPy 2.5 Introduces MIPROv2 Autonomous Prompt Optimization",
            "summary": "Stanford NLP group introduces MIPROv2 within DSPy 2.5, replacing manual prompt engineering with Bayesian program optimization across multi-stage LM pipelines and agent signatures.",
            "category": "Research",
            "date": "2024-11-18",
            "provider_or_author": "Stanford NLP",
            "evidence_confidence": "PEER_REVIEWED",
            "impact_score": 8.5,
            "tags": ["Prompt Optimization", "Bayesian", "Agent Pipelines", "Compilers"],
            "related_entities": ["tool-dspy", "paper-dspy-mipro", "model-gpt-4o"],
            "evidence_id": "ev-arxiv-dspy-mipro"
        },
        {
            "id": "dev-qwen-2-5-coder",
            "title": "Alibaba Releases Qwen 2.5-Coder: Open Source SOTA Coding Models",
            "summary": "Alibaba Cloud open-sources Qwen 2.5-Coder suite up to 32B parameters. Outperforms previous open coding models on HumanEval (92.7%) and achieves 36.8% on SWE-bench Verified.",
            "category": "Models",
            "date": "2024-11-12",
            "provider_or_author": "Alibaba Cloud",
            "evidence_confidence": "EVALUATED",
            "impact_score": 8.6,
            "tags": ["Coding", "Open Weights", "Apache 2.0", "Repository Reasoning"],
            "related_entities": ["model-qwen-2-5-coder-32b", "bench-human-eval", "bench-swe-bench-verified"],
            "evidence_id": "ev-hf-qwen-2-5-coder"
        },
        {
            "id": "dev-gaia-benchmark-study",
            "title": "GAIA Benchmark Study: Complex Web Assistants Plateau at Level 3",
            "summary": "Independent evaluation of leading LLM agents on GAIA demonstrates stark contrast: Level 1 simple tasks exceed 80% success, while Level 3 multi-modal long-horizon web tasks remain below 35% across all providers.",
            "category": "Benchmarks",
            "date": "2024-11-05",
            "provider_or_author": "Meta AI & Hugging Face",
            "evidence_confidence": "PEER_REVIEWED",
            "impact_score": 8.9,
            "tags": ["General Assistant", "Web Agents", "Multi-modal Evaluation"],
            "related_entities": ["bench-gaia", "model-claude-3-5-sonnet", "tool-smolagents"],
            "evidence_id": "ev-arxiv-gaia-eval"
        }
    ],
    "models": [
        {
            "id": "model-deepseek-r1",
            "name": "DeepSeek-R1",
            "provider": "DeepSeek AI",
            "architecture": "MoE Transformer (671B total, 37B active)",
            "release_date": "2025-01-20",
            "context_window": 131072,
            "license": "MIT (Open Weights & Commercial)",
            "modalities": ["Text", "Code"],
            "pricing": {
                "input_per_million": 0.55,
                "output_per_million": 2.19
            },
            "description": "State-of-the-art open reasoning model trained via large-scale reinforcement learning without supervised warm-up. Demonstrates chain-of-thought self-correction and reflection.",
            "strengths": ["Mathematics", "Algorithmic Code", "Logical Deduction", "Transparent Inference"],
            "benchmarks": {
                "SWE-bench Verified": 49.2,
                "GPQA Diamond": 71.5,
                "HumanEval": 92.4,
                "MATH-500": 97.3,
                "MMLU-Pro": 84.0,
                "GAIA": 41.2
            },
            "evidence_id": "ev-arxiv-2501-12948"
        },
        {
            "id": "model-claude-3-5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "provider": "Anthropic",
            "architecture": "Proprietary Dense / MoE",
            "release_date": "2024-10-22",
            "context_window": 200000,
            "license": "Proprietary Commercial API",
            "modalities": ["Text", "Vision", "Code"],
            "pricing": {
                "input_per_million": 3.00,
                "output_per_million": 15.00
            },
            "description": "Leading frontier model for software development, agentic workflows, and GUI computer use. Known for nuanced prompt adherence and multi-file reasoning.",
            "strengths": ["Agentic Coding", "GUI Navigation", "Complex Tool Calling", "Technical Writing"],
            "benchmarks": {
                "SWE-bench Verified": 49.0,
                "GPQA Diamond": 65.0,
                "HumanEval": 93.7,
                "MATH-500": 78.3,
                "MMLU-Pro": 78.0,
                "GAIA": 46.5
            },
            "evidence_id": "ev-anthropic-claude-3-5"
        },
        {
            "id": "model-o3-mini",
            "name": "o3-mini (High Effort)",
            "provider": "OpenAI",
            "architecture": "Reasoning Model with Deliberate Search",
            "release_date": "2025-01-31",
            "context_window": 200000,
            "license": "Proprietary Commercial API",
            "modalities": ["Text", "Code"],
            "pricing": {
                "input_per_million": 1.10,
                "output_per_million": 4.40
            },
            "description": "Cost-effective reasoning model offering selectable reasoning effort (low/med/high). Designed for STEM, competitive coding, and math problem decomposition.",
            "strengths": ["Competitive Programming", "High-School Olympiad Math", "Structured JSON", "Low Latency"],
            "benchmarks": {
                "SWE-bench Verified": 48.8,
                "GPQA Diamond": 77.2,
                "HumanEval": 91.8,
                "MATH-500": 97.9,
                "MMLU-Pro": 82.5,
                "GAIA": 44.0
            },
            "evidence_id": "ev-openai-o3-mini"
        },
        {
            "id": "model-gpt-4o",
            "name": "GPT-4o (2024-11-20)",
            "provider": "OpenAI",
            "architecture": "Omni Multimodal Transformer",
            "release_date": "2024-11-20",
            "context_window": 128000,
            "license": "Proprietary Commercial API",
            "modalities": ["Text", "Audio", "Vision", "Code"],
            "pricing": {
                "input_per_million": 2.50,
                "output_per_million": 10.00
            },
            "description": "High-throughput omni model natively handling text, image, and voice inputs/outputs. Standard workhorse for broad multi-modal assistant applications.",
            "strengths": ["Native Audio & Vision", "Broad Knowledge", "Fast Response", "Ecosystem Integrations"],
            "benchmarks": {
                "SWE-bench Verified": 38.8,
                "GPQA Diamond": 53.6,
                "HumanEval": 90.2,
                "MATH-500": 74.6,
                "MMLU-Pro": 72.6,
                "GAIA": 38.2
            },
            "evidence_id": "ev-openai-gpt4o"
        },
        {
            "id": "model-gemini-2-flash",
            "name": "Gemini 2.0 Flash",
            "provider": "Google DeepMind",
            "architecture": "Multimodal Transformer with Native Streaming",
            "release_date": "2024-12-11",
            "context_window": 1048576,
            "license": "Proprietary Commercial API",
            "modalities": ["Text", "Vision", "Audio", "Video", "Code"],
            "pricing": {
                "input_per_million": 0.10,
                "output_per_million": 0.40
            },
            "description": "Next-generation ultra-fast multimodal model. Native tool execution, live audio/video streaming, and massive 1M token context window at ultra-low price point.",
            "strengths": ["Sub-second Latency", "1M Context Ingestion", "Multimodal Video Reasoning", "Low Cost"],
            "benchmarks": {
                "SWE-bench Verified": 35.2,
                "GPQA Diamond": 56.4,
                "HumanEval": 88.0,
                "MATH-500": 72.0,
                "MMLU-Pro": 70.4,
                "GAIA": 37.0
            },
            "evidence_id": "ev-google-gemini-2"
        },
        {
            "id": "model-llama-3-3-70b",
            "name": "Llama 3.3 70B Instruct",
            "provider": "Meta AI",
            "architecture": "Dense Autoregressive Transformer 70B",
            "release_date": "2024-12-06",
            "context_window": 131072,
            "license": "Llama 3.3 Community License",
            "modalities": ["Text", "Code"],
            "pricing": {
                "input_per_million": 0.20,
                "output_per_million": 0.20
            },
            "description": "Open weights powerhouse matching the performance of the earlier 405B model across core benchmarks. Highly optimized for on-premises deployment.",
            "strengths": ["Enterprise Privacy", "Self-Hosting", "High Throughput", "Generalist Tasks"],
            "benchmarks": {
                "SWE-bench Verified": 34.0,
                "GPQA Diamond": 52.8,
                "HumanEval": 88.4,
                "MATH-500": 71.0,
                "MMLU-Pro": 69.2,
                "GAIA": 32.5
            },
            "evidence_id": "ev-meta-llama-3-3"
        },
        {
            "id": "model-qwen-2-5-coder-32b",
            "name": "Qwen 2.5 Coder 32B Instruct",
            "provider": "Alibaba Cloud",
            "architecture": "Dense Transformer 32.5B",
            "release_date": "2024-11-12",
            "context_window": 131072,
            "license": "Apache 2.0",
            "modalities": ["Text", "Code"],
            "pricing": {
                "input_per_million": 0.15,
                "output_per_million": 0.15
            },
            "description": "Per-parameter leader in software engineering. Fits comfortably on a single RTX 4090 with 4-bit quantization while delivering frontier-tier coding results.",
            "strengths": ["Local Code Assistants", "Repo-level Context", "Git Diffs", "Permissive License"],
            "benchmarks": {
                "SWE-bench Verified": 36.8,
                "GPQA Diamond": 48.2,
                "HumanEval": 92.7,
                "MATH-500": 78.4,
                "MMLU-Pro": 65.8,
                "GAIA": 33.1
            },
            "evidence_id": "ev-hf-qwen-2-5-coder"
        }
    ],
    "benchmarks": [
        {
            "id": "bench-swe-bench-verified",
            "name": "SWE-bench Verified",
            "category": "Software Engineering & Autonomous Agents",
            "metric_name": "% Solved (Unit Test Passing)",
            "description": "Curated subset of 500 human-verified GitHub issue resolution tasks across 12 major Python repositories. Represents true agentic programming capabilities.",
            "methodology": "Agent receives issue description + repository snapshot; must produce git patch that passes secret unit test suite.",
            "comparability_conditions": "Scores must use the verified 500-test split under identical bash execution harness without human interventions.",
            "evaluations": [
                {"model_id": "model-deepseek-r1", "score": 49.2, "evaluated_by": "DeepSeek Research", "date": "2025-01-20", "confidence": "EVALUATED"},
                {"model_id": "model-claude-3-5-sonnet", "score": 49.0, "evaluated_by": "Anthropic & SWE-bench Leaderboard", "date": "2024-10-22", "confidence": "EVALUATED"},
                {"model_id": "model-o3-mini", "score": 48.8, "evaluated_by": "OpenAI Evaluation Suite", "date": "2025-01-31", "confidence": "EVALUATED"},
                {"model_id": "model-gpt-4o", "score": 38.8, "evaluated_by": "SWE-bench Official", "date": "2024-11-20", "confidence": "PEER_REVIEWED"},
                {"model_id": "model-qwen-2-5-coder-32b", "score": 36.8, "evaluated_by": "Qwen Team", "date": "2024-11-12", "confidence": "EVALUATED"},
                {"model_id": "model-gemini-2-flash", "score": 35.2, "evaluated_by": "Google DeepMind", "date": "2024-12-11", "confidence": "ANNOUNCEMENT"},
                {"model_id": "model-llama-3-3-70b", "score": 34.0, "evaluated_by": "Meta AI", "date": "2024-12-06", "confidence": "EVALUATED"}
            ]
        },
        {
            "id": "bench-gpqa-diamond",
            "name": "GPQA Diamond",
            "category": "Graduate-Level Reasoning & STEM",
            "metric_name": "Accuracy (%) 0-shot CoT",
            "description": "448 multiple-choice questions written by PhD experts in biology, physics, and chemistry designed to resist web search lookups.",
            "methodology": "0-shot chain-of-thought prompt with temperature 0.0 or majority voting.",
            "comparability_conditions": "Must specify whether test-time compute (consensus / n-samples) or single pass was utilized.",
            "evaluations": [
                {"model_id": "model-o3-mini", "score": 77.2, "evaluated_by": "OpenAI Evaluation Suite", "date": "2025-01-31", "confidence": "EVALUATED"},
                {"model_id": "model-deepseek-r1", "score": 71.5, "evaluated_by": "DeepSeek Research", "date": "2025-01-20", "confidence": "EVALUATED"},
                {"model_id": "model-claude-3-5-sonnet", "score": 65.0, "evaluated_by": "Anthropic", "date": "2024-10-22", "confidence": "EVALUATED"},
                {"model_id": "model-gemini-2-flash", "score": 56.4, "evaluated_by": "Google DeepMind", "date": "2024-12-11", "confidence": "ANNOUNCEMENT"},
                {"model_id": "model-gpt-4o", "score": 53.6, "evaluated_by": "NYU & Google Research", "date": "2024-11-20", "confidence": "PEER_REVIEWED"},
                {"model_id": "model-llama-3-3-70b", "score": 52.8, "evaluated_by": "Meta AI", "date": "2024-12-06", "confidence": "EVALUATED"},
                {"model_id": "model-qwen-2-5-coder-32b", "score": 48.2, "evaluated_by": "Alibaba Cloud", "date": "2024-11-12", "confidence": "EVALUATED"}
            ]
        },
        {
            "id": "bench-math-500",
            "name": "MATH-500",
            "category": "Mathematical Reasoning",
            "metric_name": "Accuracy (%)",
            "description": "500 challenging competition math problems from the Hendrycks MATH dataset across algebra, calculus, geometry, and number theory.",
            "methodology": "Exact match with LaTeX normalization.",
            "comparability_conditions": "Must be evaluated on the standard 500-question subset without external Python REPL access unless flagged.",
            "evaluations": [
                {"model_id": "model-o3-mini", "score": 97.9, "evaluated_by": "OpenAI Evaluation Suite", "date": "2025-01-31", "confidence": "EVALUATED"},
                {"model_id": "model-deepseek-r1", "score": 97.3, "evaluated_by": "DeepSeek Research", "date": "2025-01-20", "confidence": "EVALUATED"},
                {"model_id": "model-qwen-2-5-coder-32b", "score": 78.4, "evaluated_by": "Alibaba Cloud", "date": "2024-11-12", "confidence": "EVALUATED"},
                {"model_id": "model-claude-3-5-sonnet", "score": 78.3, "evaluated_by": "Anthropic", "date": "2024-10-22", "confidence": "EVALUATED"},
                {"model_id": "model-gpt-4o", "score": 74.6, "evaluated_by": "OpenAI", "date": "2024-11-20", "confidence": "PEER_REVIEWED"},
                {"model_id": "model-gemini-2-flash", "score": 72.0, "evaluated_by": "Google DeepMind", "date": "2024-12-11", "confidence": "ANNOUNCEMENT"},
                {"model_id": "model-llama-3-3-70b", "score": 71.0, "evaluated_by": "Meta AI", "date": "2024-12-06", "confidence": "EVALUATED"}
            ]
        },
        {
            "id": "bench-gaia",
            "name": "GAIA (General AI Assistant)",
            "category": "Multimodal Agentic Tasks",
            "metric_name": "Success Rate (%) Across Levels 1-3",
            "description": "466 questions testing multimodal real-world assistant abilities: web search, PDF inspection, audio listening, spreadsheet manipulation, multi-hop reasoning.",
            "methodology": "Exact answer match with tool-calling environment access.",
            "comparability_conditions": "Levels 1, 2, and 3 must be reported disaggregated; tool set (browsing, code execution) must be declared.",
            "evaluations": [
                {"model_id": "model-claude-3-5-sonnet", "score": 46.5, "evaluated_by": "GAIA Leaderboard", "date": "2024-10-22", "confidence": "EVALUATED"},
                {"model_id": "model-o3-mini", "score": 44.0, "evaluated_by": "Independent Benchmark", "date": "2025-01-31", "confidence": "EMPIRICAL"},
                {"model_id": "model-deepseek-r1", "score": 41.2, "evaluated_by": "Community Eval", "date": "2025-01-20", "confidence": "EMPIRICAL"},
                {"model_id": "model-gpt-4o", "score": 38.2, "evaluated_by": "Meta / HF Benchmark", "date": "2024-11-20", "confidence": "PEER_REVIEWED"},
                {"model_id": "model-gemini-2-flash", "score": 37.0, "evaluated_by": "Google DeepMind", "date": "2024-12-11", "confidence": "ANNOUNCEMENT"},
                {"model_id": "model-qwen-2-5-coder-32b", "score": 33.1, "evaluated_by": "Community Eval", "date": "2024-11-12", "confidence": "EMPIRICAL"},
                {"model_id": "model-llama-3-3-70b", "score": 32.5, "evaluated_by": "Meta AI", "date": "2024-12-06", "confidence": "EVALUATED"}
            ]
        }
    ],
    "tools": [
        {
            "id": "tool-langgraph",
            "name": "LangGraph",
            "category": "Agent Framework",
            "provider_or_org": "LangChain",
            "repo_url": "https://github.com/langchain-ai/langgraph",
            "stars": 12400,
            "description": "Stateful, cyclical multi-agent orchestration framework with first-class human-in-the-loop, time-travel debugging, and distributed persistence.",
            "release_date": "2024-12-12",
            "integrations": ["MCP", "OpenAI", "Anthropic", "vLLM", "DuckDB", "Postgres"],
            "evidence_id": "ev-langchain-langgraph-02"
        },
        {
            "id": "tool-mcp",
            "name": "Model Context Protocol (MCP)",
            "category": "Tool Protocol Standard",
            "provider_or_org": "Anthropic Open Source",
            "repo_url": "https://github.com/modelcontextprotocol/specification",
            "stars": 9800,
            "description": "Open standard that standardizes how applications provide context and tools to LLMs via JSON-RPC servers.",
            "release_date": "2024-11-25",
            "integrations": ["Claude Desktop", "Cursor", "LangGraph", "FastMCP", "Zed"],
            "evidence_id": "ev-github-mcp-spec"
        },
        {
            "id": "tool-smolagents",
            "name": "smolagents",
            "category": "Agent Framework",
            "provider_or_org": "Hugging Face",
            "repo_url": "https://github.com/huggingface/smolagents",
            "stars": 8200,
            "description": "Minimalist library where agents write executable Python code blocks directly to invoke tools, significantly reducing JSON parsing overhead.",
            "release_date": "2025-01-08",
            "integrations": ["Hugging Face Hub", "E2B", "LiteLLM", "Transformers"],
            "evidence_id": "ev-github-smolagents"
        },
        {
            "id": "tool-autogen",
            "name": "AutoGen 0.4",
            "category": "Agent Framework",
            "provider_or_org": "Microsoft Research",
            "repo_url": "https://github.com/microsoft/autogen",
            "stars": 34500,
            "description": "Asynchronous actor-based multi-agent framework facilitating conversational problem-solving across network boundaries.",
            "release_date": "2025-01-15",
            "integrations": ["Azure OpenAI", "Docker", "Ollama", "gRPC"],
            "evidence_id": "ev-github-autogen-04"
        },
        {
            "id": "tool-crewai",
            "name": "CrewAI",
            "category": "Agent Framework",
            "provider_or_org": "CrewAI Inc.",
            "repo_url": "https://github.com/crewAIInc/crewAI",
            "stars": 24200,
            "description": "Role-playing autonomous agent framework enabling team collaboration, task delegation, and structured process execution.",
            "release_date": "2024-10-30",
            "integrations": ["LangChain", "Serper", "OpenAI", "Anthropic"],
            "evidence_id": "ev-crewai-flows"
        },
        {
            "id": "tool-browser-use",
            "name": "Browser-Use",
            "category": "Autonomous Web Agent",
            "provider_or_org": "Browser-Use Team",
            "repo_url": "https://github.com/browser-use/browser-use",
            "stars": 21800,
            "description": "Open-source web automation library connecting LLMs to browser sessions for human-like navigation and interaction.",
            "release_date": "2024-12-28",
            "integrations": ["Playwright", "Claude 3.5 Sonnet", "GPT-4o"],
            "evidence_id": "ev-github-browser-use"
        },
        {
            "id": "tool-vllm",
            "name": "vLLM",
            "category": "Inference Infrastructure",
            "provider_or_org": "UC Berkeley / vLLM Project",
            "repo_url": "https://github.com/vllm-project/vllm",
            "stars": 31200,
            "description": "High-throughput and memory-efficient LLM serving engine powered by PagedAttention, native MLA, and continuous batching.",
            "release_date": "2025-01-25",
            "integrations": ["DeepSeek-R1", "Llama 3.3", "Ray", "Triton"],
            "evidence_id": "ev-github-vllm-070"
        },
        {
            "id": "tool-dspy",
            "name": "DSPy",
            "category": "Prompt Compiler",
            "provider_or_org": "Stanford NLP",
            "repo_url": "https://github.com/stanfordnlp/dspy",
            "stars": 19400,
            "description": "Framework for programming—rather than prompting—foundation models with automatic parameter and prompt optimization.",
            "release_date": "2024-11-18",
            "integrations": ["OpenAI", "Anthropic", "Cohere", "Hugging Face"],
            "evidence_id": "ev-arxiv-dspy-mipro"
        }
    ],
    "ecosystem_graph": {
        "nodes": [
            {"id": "paper-deepseek-r1", "label": "DeepSeek-R1 Paper", "category": "Research", "sub_type": "Paper", "description": "RL without SFT reasoning discovery", "metrics": {"citations": 1420}},
            {"id": "paper-dspy-mipro", "label": "DSPy MIPROv2", "category": "Research", "sub_type": "Paper", "description": "Bayesian prompt optimization", "metrics": {"citations": 380}},
            {"id": "paper-swe-bench", "label": "SWE-bench Verified", "category": "Research", "sub_type": "Paper", "description": "Human-validated software benchmarks", "metrics": {"citations": 540}},
            {"id": "model-deepseek-r1", "label": "DeepSeek-R1", "category": "Model", "sub_type": "Open Weights", "description": "671B MoE frontier reasoning model", "metrics": {"benchmark_avg": 76.1}},
            {"id": "model-claude-3-5-sonnet", "label": "Claude 3.5 Sonnet", "category": "Model", "sub_type": "Proprietary", "description": "Frontier agentic coding & computer use", "metrics": {"benchmark_avg": 73.4}},
            {"id": "model-o3-mini", "label": "o3-mini", "category": "Model", "sub_type": "Proprietary", "description": "Configurable test-time compute model", "metrics": {"benchmark_avg": 74.6}},
            {"id": "model-llama-3-3-70b", "label": "Llama 3.3 70B", "category": "Model", "sub_type": "Open Weights", "description": "Meta 405B-class distilled weights", "metrics": {"benchmark_avg": 59.8}},
            {"id": "model-qwen-2-5-coder-32b", "label": "Qwen 2.5 Coder 32B", "category": "Model", "sub_type": "Open Weights", "description": "Open state-of-the-art code generation", "metrics": {"benchmark_avg": 60.1}},
            {"id": "model-gemini-2-flash", "label": "Gemini 2.0 Flash", "category": "Model", "sub_type": "Proprietary", "description": "Fast multimodal 1M context engine", "metrics": {"benchmark_avg": 59.8}},
            {"id": "model-gpt-4o", "label": "GPT-4o", "category": "Model", "sub_type": "Proprietary", "description": "Omni multimodal workhorse", "metrics": {"benchmark_avg": 61.3}},
            {"id": "tool-mcp", "label": "MCP Protocol", "category": "Tool", "sub_type": "Protocol", "description": "Standardized LLM tool calling interface", "metrics": {"stars": 9800}},
            {"id": "tool-langgraph", "label": "LangGraph", "category": "Tool", "sub_type": "Framework", "description": "Stateful cyclical multi-agent graph", "metrics": {"stars": 12400}},
            {"id": "tool-smolagents", "label": "smolagents", "category": "Tool", "sub_type": "Framework", "description": "CodeAgent lightweight framework", "metrics": {"stars": 8200}},
            {"id": "tool-autogen", "label": "AutoGen 0.4", "category": "Tool", "sub_type": "Framework", "description": "Async actor-based multi-agent system", "metrics": {"stars": 34500}},
            {"id": "tool-crewai", "label": "CrewAI", "category": "Tool", "sub_type": "Framework", "description": "Role-playing team agent orchestration", "metrics": {"stars": 24200}},
            {"id": "tool-browser-use", "label": "Browser-Use", "category": "Tool", "sub_type": "Agent", "description": "Autonomous browser navigation", "metrics": {"stars": 21800}},
            {"id": "tool-vllm", "label": "vLLM", "category": "Infrastructure", "sub_type": "Serving", "description": "High throughput MLA / PagedAttention engine", "metrics": {"stars": 31200}},
            {"id": "tool-dspy", "label": "DSPy", "category": "Tool", "sub_type": "Compiler", "description": "Self-optimizing prompt compilation", "metrics": {"stars": 19400}},
            {"id": "bench-swe-bench-verified", "label": "SWE-bench Verified", "category": "Benchmark", "sub_type": "Benchmark", "description": "500 software engineering benchmark", "metrics": {"evals": 7}}
        ],
        "edges": [
            {"id": "edge-1", "source": "paper-deepseek-r1", "target": "model-deepseek-r1", "relationship": "derives_from", "strength": 1.0, "confidence": "PEER_REVIEWED"},
            {"id": "edge-2", "source": "model-deepseek-r1", "target": "tool-vllm", "relationship": "powers", "strength": 0.95, "confidence": "EMPIRICAL"},
            {"id": "edge-3", "source": "model-deepseek-r1", "target": "tool-smolagents", "relationship": "integrates_with", "strength": 0.85, "confidence": "EMPIRICAL"},
            {"id": "edge-4", "source": "model-claude-3-5-sonnet", "target": "tool-mcp", "relationship": "implements", "strength": 1.0, "confidence": "ANNOUNCEMENT"},
            {"id": "edge-5", "source": "model-claude-3-5-sonnet", "target": "tool-browser-use", "relationship": "powers", "strength": 0.9, "confidence": "EMPIRICAL"},
            {"id": "edge-6", "source": "model-claude-3-5-sonnet", "target": "bench-swe-bench-verified", "relationship": "evaluates_on", "strength": 0.95, "confidence": "EVALUATED"},
            {"id": "edge-7", "source": "model-deepseek-r1", "target": "bench-swe-bench-verified", "relationship": "evaluates_on", "strength": 0.95, "confidence": "EVALUATED"},
            {"id": "edge-8", "source": "model-o3-mini", "target": "bench-swe-bench-verified", "relationship": "evaluates_on", "strength": 0.95, "confidence": "EVALUATED"},
            {"id": "edge-9", "source": "tool-mcp", "target": "tool-langgraph", "relationship": "integrates_with", "strength": 0.9, "confidence": "EMPIRICAL"},
            {"id": "edge-10", "source": "tool-langgraph", "target": "tool-crewai", "relationship": "competes_with", "strength": 0.8, "confidence": "EMPIRICAL"},
            {"id": "edge-11", "source": "tool-autogen", "target": "tool-langgraph", "relationship": "competes_with", "strength": 0.8, "confidence": "EMPIRICAL"},
            {"id": "edge-12", "source": "paper-dspy-mipro", "target": "tool-dspy", "relationship": "implements", "strength": 1.0, "confidence": "PEER_REVIEWED"},
            {"id": "edge-13", "source": "tool-dspy", "target": "model-gpt-4o", "relationship": "optimizes_for", "strength": 0.85, "confidence": "EVALUATED"},
            {"id": "edge-14", "source": "model-llama-3-3-70b", "target": "tool-vllm", "relationship": "powers", "strength": 0.9, "confidence": "EMPIRICAL"},
            {"id": "edge-15", "source": "model-qwen-2-5-coder-32b", "target": "tool-vllm", "relationship": "powers", "strength": 0.9, "confidence": "EMPIRICAL"},
            {"id": "edge-16", "source": "paper-swe-bench", "target": "bench-swe-bench-verified", "relationship": "defines", "strength": 1.0, "confidence": "PEER_REVIEWED"}
        ]
    },
    "evidence": {
        "ev-arxiv-2501-12948": {
            "source_id": "arxiv:2501.12948",
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2501.12948",
            "publication_date": "2025-01-22T04:21:00Z",
            "ingestion_timestamp": "2025-01-22T06:00:00Z",
            "extraction_timestamp": "2025-01-22T06:15:00Z",
            "evidence_confidence": "EVALUATED",
            "raw_payload_ref": "bronze/arxiv/2501.12948.json",
            "title": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
            "authors": ["DeepSeek-AI", "Daya Guo", "Dejian Yang", "Haowei Zhang", "et al."],
            "summary_claim": "RL without supervised fine-tuning produces emergent reflection, multi-step search, and verification capabilities on MATH-500 (97.3%) and SWE-bench (49.2%).",
            "verification_status": "Independently verified by open community replication and Hugging Face leaderboard."
        },
        "ev-anthropic-claude-3-5": {
            "source_id": "anthropic:blog:2024-10-claude-3-5",
            "source_type": "official_announcement",
            "source_url": "https://www.anthropic.com/news/3-5-models-and-computer-use",
            "publication_date": "2024-10-22T14:00:00Z",
            "ingestion_timestamp": "2024-10-22T14:30:00Z",
            "extraction_timestamp": "2024-10-22T14:45:00Z",
            "evidence_confidence": "EVALUATED",
            "raw_payload_ref": "bronze/anthropic/claude_3_5_release.json",
            "title": "Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku",
            "authors": ["Anthropic Product & Research Teams"],
            "summary_claim": "Upgraded Claude 3.5 Sonnet scores 49.0% on SWE-bench Verified and introduces OS-level API for mouse, keyboard, and screen automation.",
            "verification_status": "SWE-bench Verified score confirmed by Princeton SWE-bench official leaderboard."
        },
        "ev-github-mcp-spec": {
            "source_id": "github:modelcontextprotocol/specification",
            "source_type": "github",
            "source_url": "https://github.com/modelcontextprotocol/specification",
            "publication_date": "2024-11-25T17:00:00Z",
            "ingestion_timestamp": "2024-11-25T18:00:00Z",
            "extraction_timestamp": "2024-11-25T18:10:00Z",
            "evidence_confidence": "EMPIRICAL",
            "raw_payload_ref": "bronze/github/mcp_repo.json",
            "title": "Model Context Protocol Specification v1.0",
            "authors": ["Model Context Protocol Core Contributors"],
            "summary_claim": "Specification defining standardized client-server protocol over stdio and SSE for tools, prompts, and resource access.",
            "verification_status": "Open source repository active with thousands of stars and community implementations."
        },
        "ev-openai-o3-mini": {
            "source_id": "openai:announcement:o3-mini",
            "source_type": "official_announcement",
            "source_url": "https://openai.com/index/openai-o3-mini/",
            "publication_date": "2025-01-31T18:00:00Z",
            "ingestion_timestamp": "2025-01-31T18:15:00Z",
            "extraction_timestamp": "2025-01-31T18:30:00Z",
            "evidence_confidence": "ANNOUNCEMENT",
            "raw_payload_ref": "bronze/openai/o3_mini_launch.json",
            "title": "OpenAI o3-mini: A small reasoning model for STEM and coding",
            "authors": ["OpenAI Team"],
            "summary_claim": "Provides high reasoning effort configuration scoring 77.2% on GPQA Diamond and 97.9% on MATH-500 with major cost reductions.",
            "verification_status": "Self-reported benchmark claims pending exhaustive third-party leaderboard indexing."
        },
        "ev-github-smolagents": {
            "source_id": "github:huggingface/smolagents",
            "source_type": "github",
            "source_url": "https://github.com/huggingface/smolagents",
            "publication_date": "2025-01-08T11:00:00Z",
            "ingestion_timestamp": "2025-01-08T12:00:00Z",
            "extraction_timestamp": "2025-01-08T12:15:00Z",
            "evidence_confidence": "EVALUATED",
            "raw_payload_ref": "bronze/github/hf_smolagents.json",
            "title": "smolagents: a barebones library for agents that write Python code",
            "authors": ["Aymeric Roucher", "Hugging Face Team"],
            "summary_claim": "Demonstrates that CodeAgents writing raw Python function calls execute faster and with fewer tokens than JSON-schema based tool callers.",
            "verification_status": "Evaluated on GAIA benchmark with reproducible execution logs."
        },
        "ev-langchain-langgraph-02": {
            "source_id": "github:langchain-ai/langgraph/releases/tag/0.2.0",
            "source_type": "github",
            "source_url": "https://github.com/langchain-ai/langgraph/releases/tag/0.2.0",
            "publication_date": "2024-12-12T16:00:00Z",
            "ingestion_timestamp": "2024-12-12T16:30:00Z",
            "extraction_timestamp": "2024-12-12T16:45:00Z",
            "evidence_confidence": "EMPIRICAL",
            "raw_payload_ref": "bronze/github/langgraph_release.json",
            "title": "LangGraph v0.2 Release Notes",
            "authors": ["LangChain Core Engineering"],
            "summary_claim": "First-class support for durable execution with asynchronous checkpoints, branch replay, and interrupt commands.",
            "verification_status": "Production package on PyPI with automated test suites."
        },
        "ev-arxiv-swe-bench-verified": {
            "source_id": "arxiv:2408.06292",
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2408.06292",
            "publication_date": "2024-08-15T15:00:00Z",
            "ingestion_timestamp": "2024-08-15T16:00:00Z",
            "extraction_timestamp": "2024-08-15T16:30:00Z",
            "evidence_confidence": "PEER_REVIEWED",
            "raw_payload_ref": "bronze/arxiv/swe_bench_verified.json",
            "title": "SWE-bench Verified: Resolving Problems in Software Engineering Benchmarking",
            "authors": ["Carlos E. Jimenez", "John Yang", "Alexander Wettig", "Ofir Press", "et al."],
            "summary_claim": "Human-in-the-loop audit removing over-constrained unit tests, reducing benchmark noise and providing clean ground truth.",
            "verification_status": "Peer reviewed and adopted as standard evaluation harness by Anthropic, OpenAI, Meta, and DeepSeek."
        },
        "ev-google-gemini-2": {
            "source_id": "google:blog:gemini-2",
            "source_type": "official_announcement",
            "source_url": "https://blog.google/technology/ai/google-gemini-next-generation-model-december-2024/",
            "publication_date": "2024-12-11T17:00:00Z",
            "ingestion_timestamp": "2024-12-11T17:30:00Z",
            "extraction_timestamp": "2024-12-11T17:45:00Z",
            "evidence_confidence": "ANNOUNCEMENT",
            "raw_payload_ref": "bronze/google/gemini2_announcement.json",
            "title": "Our next-generation model: Gemini 2.0 Flash",
            "authors": ["Demis Hassabis", "Google DeepMind Team"],
            "summary_claim": "Real-time multimodal streaming API with native tool calling, search grounding, and Python code execution.",
            "verification_status": "Developer preview live in Google AI Studio."
        },
        "ev-meta-llama-3-3": {
            "source_id": "meta:blog:llama-3-3",
            "source_type": "official_announcement",
            "source_url": "https://ai.meta.com/blog/llama-3-3-70b-open-source-foundation-model/",
            "publication_date": "2024-12-06T15:00:00Z",
            "ingestion_timestamp": "2024-12-06T15:30:00Z",
            "extraction_timestamp": "2024-12-06T15:45:00Z",
            "evidence_confidence": "EVALUATED",
            "raw_payload_ref": "bronze/meta/llama33_release.json",
            "title": "Llama 3.3: 70B Model with 405B Performance",
            "authors": ["Meta AI Research Team"],
            "summary_claim": "Achieves 71.0% on MATH-500 and 88.4% on HumanEval, matching Llama 3.1 405B through advanced post-training.",
            "verification_status": "Open weights evaluated independently across Hugging Face Open LLM Leaderboard."
        },
        "ev-github-browser-use": {
            "source_id": "github:browser-use/browser-use",
            "source_type": "github",
            "source_url": "https://github.com/browser-use/browser-use",
            "publication_date": "2024-12-28T12:00:00Z",
            "ingestion_timestamp": "2024-12-28T13:00:00Z",
            "extraction_timestamp": "2024-12-28T13:30:00Z",
            "evidence_confidence": "EMPIRICAL",
            "raw_payload_ref": "bronze/github/browser_use.json",
            "title": "browser-use: Make websites accessible for AI agents",
            "authors": ["Gregor Zunic", "Magnus Müller"],
            "summary_claim": "Combines Playwright DOM extraction with multimodal vision feedback to reliably click, type, and complete workflows.",
            "verification_status": "Empirical adoption across thousands of open-source developers."
        },
        "ev-github-vllm-070": {
            "source_id": "github:vllm-project/vllm/releases/tag/v0.7.0",
            "source_type": "github",
            "source_url": "https://github.com/vllm-project/vllm/releases/tag/v0.7.0",
            "publication_date": "2025-01-25T20:00:00Z",
            "ingestion_timestamp": "2025-01-25T20:30:00Z",
            "extraction_timestamp": "2025-01-25T20:45:00Z",
            "evidence_confidence": "EMPIRICAL",
            "raw_payload_ref": "bronze/github/vllm_070.json",
            "title": "vLLM v0.7.0 Release Notes",
            "authors": ["vLLM Core Contributors"],
            "summary_claim": "Full support for Multi-head Latent Attention (MLA) and FP8 kernels enabling high-throughput DeepSeek-R1 serving.",
            "verification_status": "Validated on 8xH100 clusters with published benchmark throughput curves."
        },
        "ev-arxiv-dspy-mipro": {
            "source_id": "arxiv:2406.11695",
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2406.11695",
            "publication_date": "2024-11-18T10:00:00Z",
            "ingestion_timestamp": "2024-11-18T11:00:00Z",
            "extraction_timestamp": "2024-11-18T11:30:00Z",
            "evidence_confidence": "PEER_REVIEWED",
            "raw_payload_ref": "bronze/arxiv/dspy_mipro.json",
            "title": "MIPROv2: Multi-Prompt Optimization for Complex Agentic Pipelines",
            "authors": ["Krista Opsahl-Ong", "Michael Ryan", "Omar Khattab", "Christopher Potts"],
            "summary_claim": "Joint optimization of instructions and few-shot demonstrations delivers +12-18% gains across reasoning tasks over human prompts.",
            "verification_status": "Peer reviewed at EMNLP 2024."
        },
        "ev-hf-qwen-2-5-coder": {
            "source_id": "huggingface:Qwen/Qwen2.5-Coder-32B-Instruct",
            "source_type": "huggingface",
            "source_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
            "publication_date": "2024-11-12T08:00:00Z",
            "ingestion_timestamp": "2024-11-12T08:30:00Z",
            "extraction_timestamp": "2024-11-12T09:00:00Z",
            "evidence_confidence": "EVALUATED",
            "raw_payload_ref": "bronze/hf/qwen_2_5_coder.json",
            "title": "Qwen2.5-Coder: Powerful Code-Specific LLMs",
            "authors": ["Qwen Team", "Alibaba Cloud"],
            "summary_claim": "Surpasses earlier code models with 92.7% on HumanEval and 36.8% on SWE-bench Verified under Apache 2.0 license.",
            "verification_status": "Verified on official BigCode and SWE-bench leaderboards."
        },
        "ev-arxiv-gaia-eval": {
            "source_id": "arxiv:2311.12983",
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2311.12983",
            "publication_date": "2024-11-05T14:00:00Z",
            "ingestion_timestamp": "2024-11-05T14:30:00Z",
            "extraction_timestamp": "2024-11-05T15:00:00Z",
            "evidence_confidence": "PEER_REVIEWED",
            "raw_payload_ref": "bronze/arxiv/gaia_benchmark.json",
            "title": "GAIA: A Benchmark for General AI Assistants",
            "authors": ["Gregoire Mialon", "Clémentine Fourrier", "Craig Swift", "Thomas Wolf", "Yann LeCun"],
            "summary_claim": "Demonstrates gap between synthetic benchmark scores and real-world multi-tool assistant execution.",
            "verification_status": "ICLR 2024 Oral Presentation."
        },
        "ev-github-autogen-04": {
            "source_id": "github:microsoft/autogen/releases/tag/v0.4.0",
            "source_type": "github",
            "source_url": "https://github.com/microsoft/autogen/releases/tag/v0.4.0",
            "publication_date": "2025-01-15T19:00:00Z",
            "ingestion_timestamp": "2025-01-15T19:30:00Z",
            "extraction_timestamp": "2025-01-15T20:00:00Z",
            "evidence_confidence": "EMPIRICAL",
            "raw_payload_ref": "bronze/github/autogen_04.json",
            "title": "AutoGen 0.4 Architecture Release",
            "authors": ["Chi Wang", "Microsoft Research"],
            "summary_claim": "Decoupled actor-model architecture replacing legacy conversational loops with typed message passing.",
            "verification_status": "Verified in open source release tests."
        },
        "ev-crewai-flows": {
            "source_id": "crewai:blog:flows-launch",
            "source_type": "official_announcement",
            "source_url": "https://blog.crewai.com/introducing-crewai-flows/",
            "publication_date": "2024-10-30T13:00:00Z",
            "ingestion_timestamp": "2024-10-30T13:30:00Z",
            "extraction_timestamp": "2024-10-30T14:00:00Z",
            "evidence_confidence": "ANNOUNCEMENT",
            "raw_payload_ref": "bronze/crewai/flows_launch.json",
            "title": "Introducing CrewAI Flows: Structured Multi-Agent Control",
            "authors": ["João Moura", "CrewAI Team"],
            "summary_claim": "Combines deterministic workflows with autonomous agent flexibility.",
            "verification_status": "Available in CrewAI >= 0.70.0."
        },
        "ev-openai-gpt4o": {
            "source_id": "openai:announcement:gpt-4o-update",
            "source_type": "official_announcement",
            "source_url": "https://openai.com/index/hello-gpt-4o/",
            "publication_date": "2024-11-20T16:00:00Z",
            "ingestion_timestamp": "2024-11-20T16:30:00Z",
            "extraction_timestamp": "2024-11-20T17:00:00Z",
            "evidence_confidence": "PEER_REVIEWED",
            "raw_payload_ref": "bronze/openai/gpt4o_update.json",
            "title": "GPT-4o Evaluation and System Card",
            "authors": ["OpenAI Safety & Evaluation Teams"],
            "summary_claim": "Full evaluation metrics on GPQA, HumanEval, and MMLU-Pro with red-teaming report.",
            "verification_status": "Independent evaluations documented in LMSYS Chatbot Arena."
        }
    }
}

target_path = Path("/Users/jonathankalsky/Developer/Datathon26/datathon-2026/Databricks-Applications/SIGNAL/data/fixtures/signal_fixtures.json")
target_path.parent.mkdir(parents=True, exist_ok=True)
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(fixtures, f, indent=2)

print(f"Wrote {len(fixtures['recent_developments'])} developments, {len(fixtures['models'])} models, {len(fixtures['benchmarks'])} benchmarks, {len(fixtures['tools'])} tools, and {len(fixtures['evidence'])} evidence records to {target_path}")
