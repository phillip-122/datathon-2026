"""
SIGNAL GitHub Ingestion Collector
Acquires repository metadata, release versions, release notes, star counts,
and licenses for key AI agent frameworks and LLM serving runtimes.
Supports live GitHub REST API calls (with optional GITHUB_TOKEN) and deterministic mock mode for offline replay.
"""

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from .base_collector import BaseCollector


class GitHubCollector(BaseCollector):
    """
    Collector for AI Developer Tools and Agent Frameworks on GitHub.
    """

    DEFAULT_REPOS = [
        "langchain-ai/langgraph",
        "crewAIInc/crewAI",
        "microsoft/autogen",
        "vllm-project/vllm",
        "ollama/ollama",
        "run-llama/llama_index",
        "huggingface/smolagents",
        "stanfordnlp/dspy",
        "jxnl/instructor",
    ]

    @property
    def source_name(self) -> str:
        return "github"

    def collect(
        self,
        repos: Optional[List[str]] = None,
        fetch_releases: bool = True,
        limit: Optional[int] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Collects repository and release metadata for AI frameworks.
        """
        target_repos = repos or self.DEFAULT_REPOS
        if limit:
            target_repos = target_repos[:limit]

        query_params = {
            "repos": target_repos,
            "fetch_releases": fetch_releases,
            "limit": len(target_repos),
        }

        if self.mode == "live":
            self.logger.info(f"Querying live GitHub API for {len(target_repos)} repositories...")
            try:
                records = self._fetch_live(target_repos, fetch_releases=fetch_releases)
            except Exception as e:
                self.logger.warning(
                    f"Live GitHub fetch encountered error ({e}). Falling back to curated mock dataset."
                )
                records = self._get_mock_records(limit=len(target_repos))
        else:
            self.logger.info(f"Generating mock GitHub records (count={len(target_repos)})")
            records = self._get_mock_records(limit=len(target_repos))

        envelope = self.build_envelope(records, query_params=query_params)
        if save:
            self.save_raw(envelope, filename_prefix="github_frameworks")

        return envelope

    def _fetch_live(
        self,
        repos: List[str],
        fetch_releases: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Queries api.github.com for repository info and latest releases.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "SIGNAL-AI-Landscape-Intelligence/1.0",
        }
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        records: List[Dict[str, Any]] = []

        for repo_slug in repos:
            repo_url = f"https://api.github.com/repos/{repo_slug}"
            resp = requests.get(repo_url, headers=headers, timeout=self.timeout)

            if resp.status_code == 403:
                self.logger.warning(
                    f"GitHub API rate limit exceeded for {repo_slug}. Using mock data for remaining."
                )
                # Fallback to mock for this repo
                mock_match = [m for m in self._get_mock_records() if m["repo_slug"].lower() == repo_slug.lower()]
                if mock_match:
                    records.append(mock_match[0])
                continue

            resp.raise_for_status()
            repo_data = resp.json()

            releases: List[Dict[str, Any]] = []
            if fetch_releases:
                rel_url = f"https://api.github.com/repos/{repo_slug}/releases"
                rel_resp = requests.get(rel_url, headers=headers, params={"per_page": 3}, timeout=self.timeout)
                if rel_resp.status_code == 200:
                    for rel in rel_resp.json():
                        releases.append({
                            "tag_name": rel.get("tag_name"),
                            "name": rel.get("name"),
                            "published_at": rel.get("published_at"),
                            "html_url": rel.get("html_url"),
                            "body_snippet": (rel.get("body") or "")[:400],
                        })

            latest_rel = releases[0] if releases else None
            pub_date = (latest_rel.get("published_at") if latest_rel else None) or repo_data.get("updated_at")

            record = {
                "source_id": f"github:{repo_slug}",
                "repo_slug": repo_slug,
                "name": repo_data.get("name"),
                "full_name": repo_data.get("full_name"),
                "owner": repo_data.get("owner", {}).get("login"),
                "description": repo_data.get("description"),
                "html_url": repo_data.get("html_url"),
                "stars_count": repo_data.get("stargazers_count", 0),
                "forks_count": repo_data.get("forks_count", 0),
                "open_issues_count": repo_data.get("open_issues_count", 0),
                "license": repo_data.get("license", {}).get("spdx_id") if repo_data.get("license") else None,
                "topics": repo_data.get("topics", []),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
                "latest_release": latest_rel,
                "recent_releases": releases,
                "evidence": self.build_evidence(
                    source_id=f"github:{repo_slug}",
                    source_type="github",
                    source_url=repo_data.get("html_url", f"https://github.com/{repo_slug}"),
                    publication_date=pub_date,
                    confidence="EVALUATED",
                    raw_payload_ref=repo_slug,
                ),
            }
            records.append(record)

        return records

    def _get_mock_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        High-fidelity fixture repositories representing leading AI agent frameworks and tools.
        """
        fixtures = [
            {
                "source_id": "github:langchain-ai/langgraph",
                "repo_slug": "langchain-ai/langgraph",
                "name": "langgraph",
                "full_name": "langchain-ai/langgraph",
                "owner": "langchain-ai",
                "framework_category": "Agent Orchestration & State Graphs",
                "description": "Build resilient language agents as graphs. Native persistence, human-in-the-loop, streaming, and multi-agent workflows.",
                "html_url": "https://github.com/langchain-ai/langgraph",
                "stars_count": 18950,
                "forks_count": 2580,
                "open_issues_count": 310,
                "license": "MIT",
                "topics": ["agents", "langchain", "state-machine", "multi-agent", "human-in-the-loop", "llm"],
                "created_at": "2024-01-16T15:00:00Z",
                "updated_at": "2025-01-24T18:40:00Z",
                "latest_release": {
                    "tag_name": "v0.2.74",
                    "name": "v0.2.74 - Enhanced Subgraph Streaming & Dynamic Interrupts",
                    "published_at": "2025-01-20T19:22:00Z",
                    "html_url": "https://github.com/langchain-ai/langgraph/releases/tag/v0.2.74",
                    "body_snippet": "Adds support for structured streaming across nested subgraphs and simplifies dynamic breakpoints.",
                },
                "supported_protocols": ["Model Context Protocol (MCP)", "OpenAI Tool Calling", "LangSmith Tracing"],
                "primary_language": "Python",
            },
            {
                "source_id": "github:crewAIInc/crewAI",
                "repo_slug": "crewAIInc/crewAI",
                "name": "crewAI",
                "full_name": "crewAIInc/crewAI",
                "owner": "crewAIInc",
                "framework_category": "Role-Playing Autonomous Multi-Agent Teams",
                "description": "Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly to tackle complex tasks.",
                "html_url": "https://github.com/crewAIInc/crewAI",
                "stars_count": 31400,
                "forks_count": 4200,
                "open_issues_count": 285,
                "license": "MIT",
                "topics": ["agents", "ai-agents", "crewai", "collaboration", "automation", "multi-agent-systems"],
                "created_at": "2023-11-23T20:00:00Z",
                "updated_at": "2025-01-24T14:10:00Z",
                "latest_release": {
                    "tag_name": "v0.105.0",
                    "name": "v0.105.0 - Hierarchical Process Telemetry & Memory Embeddings",
                    "published_at": "2025-01-18T16:45:00Z",
                    "html_url": "https://github.com/crewAIInc/crewAI/releases/tag/v0.105.0",
                    "body_snippet": "Introduces persistent contextual memory vectors and distributed task delegation across agent crews.",
                },
                "supported_protocols": ["OpenAI API", "Anthropic Claude", "Ollama", "CrewAI Flow"],
                "primary_language": "Python",
            },
            {
                "source_id": "github:microsoft/autogen",
                "repo_slug": "microsoft/autogen",
                "name": "autogen",
                "full_name": "microsoft/autogen",
                "owner": "microsoft",
                "framework_category": "Multi-Agent Conversation & Agentic AI",
                "description": "A programming framework for agentic AI. Next-generation AutoGen 0.4 architecture providing event-driven, scalable multi-agent applications.",
                "html_url": "https://github.com/microsoft/autogen",
                "stars_count": 38700,
                "forks_count": 5580,
                "open_issues_count": 420,
                "license": "CC-BY-4.0",
                "topics": ["agents", "autogen", "multi-agent", "conversational-ai", "microsoft-research"],
                "created_at": "2023-08-18T10:00:00Z",
                "updated_at": "2025-01-24T20:15:00Z",
                "latest_release": {
                    "tag_name": "v0.4.7",
                    "name": "AutoGen 0.4.7 Core Architecture Overhaul",
                    "published_at": "2025-01-22T21:00:00Z",
                    "html_url": "https://github.com/microsoft/autogen/releases/tag/v0.4.7",
                    "body_snippet": "Major release introducing asynchronous actor model, clean event bus, and declarative studio configuration.",
                },
                "supported_protocols": ["OpenAI Assistants API", "Semantic Kernel", "Docker Code Execution Sandbox"],
                "primary_language": "Python",
            },
            {
                "source_id": "github:vllm-project/vllm",
                "repo_slug": "vllm-project/vllm",
                "name": "vllm",
                "full_name": "vllm-project/vllm",
                "owner": "vllm-project",
                "framework_category": "High-Throughput LLM Serving & Inference",
                "description": "A high-throughput and memory-efficient inference and serving engine for LLMs. Features PagedAttention, continuous batching, chunked prefill, and FP8/AWQ quantization.",
                "html_url": "https://github.com/vllm-project/vllm",
                "stars_count": 42500,
                "forks_count": 6800,
                "open_issues_count": 890,
                "license": "Apache-2.0",
                "topics": ["llm-serving", "inference", "paged-attention", "cuda", "vllm", "deepseek-r1-serving"],
                "created_at": "2023-04-20T12:00:00Z",
                "updated_at": "2025-01-25T01:30:00Z",
                "latest_release": {
                    "tag_name": "v0.7.3",
                    "name": "v0.7.3 - Native DeepSeek-R1 / V3 Multi-Head Latent Attention & FlashAttention-3",
                    "published_at": "2025-01-23T12:00:00Z",
                    "html_url": "https://github.com/vllm-project/vllm/releases/tag/v0.7.3",
                    "body_snippet": "Optimized MLA kernels for DeepSeek 671B MoE serving, Hopper FP8 support, and speculative decoding.",
                },
                "supported_protocols": ["OpenAI Compatible REST API", "gRPC", "Triton Server Backend"],
                "primary_language": "Python/C++",
            },
            {
                "source_id": "github:ollama/ollama",
                "repo_slug": "ollama/ollama",
                "name": "ollama",
                "full_name": "ollama/ollama",
                "owner": "ollama",
                "framework_category": "Local LLM Runtime & Package Manager",
                "description": "Get up and running with Llama 3.3, DeepSeek-R1, Mistral, and other large language models locally on macOS, Linux, and Windows.",
                "html_url": "https://github.com/ollama/ollama",
                "stars_count": 118000,
                "forks_count": 10500,
                "open_issues_count": 520,
                "license": "MIT",
                "topics": ["local-ai", "llama", "deepseek-r1", "ollama", "macos", "cuda", "rocm"],
                "created_at": "2023-07-07T08:00:00Z",
                "updated_at": "2025-01-24T23:50:00Z",
                "latest_release": {
                    "tag_name": "v0.5.7",
                    "name": "v0.5.7 - DeepSeek-R1 Distill Models & Extended Context",
                    "published_at": "2025-01-22T18:00:00Z",
                    "html_url": "https://github.com/ollama/ollama/releases/tag/v0.5.7",
                    "body_snippet": "Adds one-click run for deepseek-r1:1.5b, 7b, 8b, 14b, 32b, 70b with automated GGUF quantization.",
                },
                "supported_protocols": ["Ollama REST API", "OpenAI Compatible Endpoint"],
                "primary_language": "Go",
            },
            {
                "source_id": "github:run-llama/llama_index",
                "repo_slug": "run-llama/llama_index",
                "name": "llama_index",
                "full_name": "run-llama/llama_index",
                "owner": "run-llama",
                "framework_category": "Context Augmentation & RAG Framework",
                "description": "LlamaIndex is a data framework for your LLM applications. Provides data connectors, document ingestion, query engines, and agent workflows.",
                "html_url": "https://github.com/run-llama/llama_index",
                "stars_count": 39200,
                "forks_count": 5400,
                "open_issues_count": 480,
                "license": "MIT",
                "topics": ["rag", "llm", "knowledge-graphs", "agents", "vector-search", "llamaindex"],
                "created_at": "2022-11-04T16:00:00Z",
                "updated_at": "2025-01-24T17:15:00Z",
                "latest_release": {
                    "tag_name": "v0.12.12",
                    "name": "v0.12.12 - Workflows Engine & Event-Driven Multimodal RAG",
                    "published_at": "2025-01-19T20:30:00Z",
                    "html_url": "https://github.com/run-llama/llama_index/releases/tag/v0.12.12",
                    "body_snippet": "Adds native async Workflows engine replacing legacy DAG pipelines, with integrated OpenTelemetry spans.",
                },
                "supported_protocols": ["LlamaHub Connectors", "OpenAI API", "HuggingFace Datasets"],
                "primary_language": "Python",
            },
            {
                "source_id": "github:huggingface/smolagents",
                "repo_slug": "huggingface/smolagents",
                "name": "smolagents",
                "full_name": "huggingface/smolagents",
                "owner": "huggingface",
                "framework_category": "Code-First Lightweight Agent Framework",
                "description": "A bare-bones library for building agents in plain Python code actions rather than JSON tool-calling envelopes.",
                "html_url": "https://github.com/huggingface/smolagents",
                "stars_count": 14200,
                "forks_count": 1150,
                "open_issues_count": 92,
                "license": "Apache-2.0",
                "topics": ["code-agents", "smolagents", "python", "huggingface", "llm-agents"],
                "created_at": "2024-12-01T12:00:00Z",
                "updated_at": "2025-01-24T19:00:00Z",
                "latest_release": {
                    "tag_name": "v1.5.0",
                    "name": "v1.5.0 - Multi-Step Code Execution Sandbox & DuckDuckGo Tool",
                    "published_at": "2025-01-16T15:00:00Z",
                    "html_url": "https://github.com/huggingface/smolagents/releases/tag/v1.5.0",
                    "body_snippet": "Executes complex tool calling as executable Python programs, drastically reducing token overhead.",
                },
                "supported_protocols": ["Hugging Face Hub", "E2B Code Sandbox"],
                "primary_language": "Python",
            },
            {
                "source_id": "github:stanfordnlp/dspy",
                "repo_slug": "stanfordnlp/dspy",
                "name": "dspy",
                "full_name": "stanfordnlp/dspy",
                "owner": "stanfordnlp",
                "framework_category": "Prompt Optimization & Algorithmic Compilation",
                "description": "DSPy: Programming—not prompting—Foundation Models. Replaces brittle hand-written prompts with composable modules and automated optimizers.",
                "html_url": "https://github.com/stanfordnlp/dspy",
                "stars_count": 21800,
                "forks_count": 1950,
                "open_issues_count": 170,
                "license": "MIT",
                "topics": ["dspy", "prompt-optimization", "nlp", "llm-programming", "stanford"],
                "created_at": "2023-01-15T09:00:00Z",
                "updated_at": "2025-01-23T22:00:00Z",
                "latest_release": {
                    "tag_name": "v2.5.43",
                    "name": "v2.5.43 - MIPROv2 Optimizer and Multi-Task Signatures",
                    "published_at": "2025-01-14T11:00:00Z",
                    "html_url": "https://github.com/stanfordnlp/dspy/releases/tag/v2.5.43",
                    "body_snippet": "Introduces MIPROv2 Bayesian prompt and few-shot instruction optimization with 3x faster convergence.",
                },
                "supported_protocols": ["LiteLLM", "OpenAI", "Anthropic", "vLLM"],
                "primary_language": "Python",
            },
            {
                "source_id": "github:jxnl/instructor",
                "repo_slug": "jxnl/instructor",
                "name": "instructor",
                "full_name": "jxnl/instructor",
                "owner": "jxnl",
                "framework_category": "Structured LLM Outputs & Pydantic Validation",
                "description": "Structured outputs powered by Pydantic for LLMs. Simple, transparent, and user-friendly validation and retries.",
                "html_url": "https://github.com/jxnl/instructor",
                "stars_count": 8900,
                "forks_count": 780,
                "open_issues_count": 65,
                "license": "MIT",
                "topics": ["structured-outputs", "pydantic", "instructor", "function-calling", "json-schema"],
                "created_at": "2023-06-25T14:00:00Z",
                "updated_at": "2025-01-22T16:00:00Z",
                "latest_release": {
                    "tag_name": "v1.7.2",
                    "name": "v1.7.2 - Gemini 2.0 & DeepSeek-R1 Schema Adapters",
                    "published_at": "2025-01-21T18:00:00Z",
                    "html_url": "https://github.com/jxnl/instructor/releases/tag/v1.7.2",
                    "body_snippet": "Streamlined parsing of thinking blocks from reasoning models before validating target Pydantic schemas.",
                },
                "supported_protocols": ["Pydantic v2", "OpenAI", "Anthropic", "Gemini", "Cohere", "Groq"],
                "primary_language": "Python",
            },
        ]

        target_fixtures = fixtures[:limit] if limit else fixtures
        records = []
        for item in target_fixtures:
            record = dict(item)
            pub_date = (
                item["latest_release"]["published_at"]
                if item.get("latest_release")
                else item["updated_at"]
            )
            record["evidence"] = self.build_evidence(
                source_id=item["source_id"],
                source_type="github",
                source_url=item["html_url"],
                publication_date=pub_date,
                confidence="EVALUATED",
                raw_payload_ref=item["repo_slug"],
            )
            records.append(record)
        return records


def main():
    parser = argparse.ArgumentParser(description="Collect repository and release metadata from GitHub API.")
    parser.add_argument("--mode", choices=["live", "mock"], default="mock", help="Execution mode (default: mock)")
    parser.add_argument("--repos", nargs="+", default=None, help="Specific owner/repo slugs to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of repositories")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    collector = GitHubCollector(output_dir=args.output_dir, mode=args.mode)
    result = collector.collect(repos=args.repos, limit=args.limit)
    print(f"Collected {result['metadata']['record_count']} records in mode={args.mode}")


if __name__ == "__main__":
    main()
