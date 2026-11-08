"""
SIGNAL Hugging Face Ingestion Collector
Acquires model metadata, download metrics, community likes, pipeline tags,
and model card parameters from the Hugging Face Model Hub API.
Supports live HTTP API calls with rate-limiting and deterministic mock mode for offline replay.
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from .base_collector import BaseCollector


class HuggingFaceCollector(BaseCollector):
    """
    Collector for Hugging Face model metadata, hub statistics, and model cards.
    """

    BASE_URL = "https://huggingface.co/api/models"

    @property
    def source_name(self) -> str:
        return "huggingface"

    def collect(
        self,
        search: Optional[str] = None,
        author: Optional[str] = None,
        pipeline_tag: str = "text-generation",
        sort: str = "downloads",
        limit: int = 15,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Collects models from Hugging Face Hub API or mock store.
        """
        query_params = {
            "search": search,
            "author": author,
            "pipeline_tag": pipeline_tag,
            "sort": sort,
            "limit": limit,
        }

        if self.mode == "live":
            self.logger.info(
                f"Querying live Hugging Face API: pipeline_tag='{pipeline_tag}', author='{author}', limit={limit}"
            )
            try:
                records = self._fetch_live(
                    search=search, author=author, pipeline_tag=pipeline_tag, sort=sort, limit=limit
                )
            except Exception as e:
                self.logger.warning(
                    f"Live Hugging Face fetch failed ({e}). Falling back to curated mock dataset."
                )
                records = self._get_mock_records(limit)
        else:
            self.logger.info(f"Generating mock Hugging Face records (count={limit})")
            records = self._get_mock_records(limit)

        envelope = self.build_envelope(records, query_params=query_params)
        if save:
            self.save_raw(envelope, filename_prefix="hf_models")

        return envelope

    def _fetch_live(
        self,
        search: Optional[str],
        author: Optional[str],
        pipeline_tag: Optional[str],
        sort: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Executes HTTP GET against huggingface.co/api/models and standardizes the output.
        """
        params: Dict[str, Any] = {
            "sort": sort,
            "direction": -1,
            "limit": limit,
            "full": "true",
            "config": "true",
        }
        if search:
            params["search"] = search
        if author:
            params["author"] = author
        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag

        headers = {
            "User-Agent": "SIGNAL-AI-Landscape-Intelligence/1.0 (academic research and model tracking)"
        }

        response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        raw_items = response.json()

        records: List[Dict[str, Any]] = []
        for item in raw_items:
            repo_id = item.get("id") or item.get("modelId")
            if not repo_id:
                continue

            author_name = repo_id.split("/")[0] if "/" in repo_id else "unknown"
            model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
            created_at = item.get("createdAt")
            likes = item.get("likes", 0)
            downloads = item.get("downloads", 0)
            tag_list = item.get("tags", [])
            tag_str = item.get("pipeline_tag") or "text-generation"
            card_data = item.get("cardData") or {}

            canonical_url = f"https://huggingface.co/{repo_id}"

            record = {
                "source_id": f"hf:{repo_id}",
                "repo_id": repo_id,
                "model_name": model_name,
                "author": author_name,
                "pipeline_tag": tag_str,
                "downloads": downloads,
                "likes": likes,
                "tags": tag_list,
                "created_at": created_at,
                "last_modified": item.get("lastModified"),
                "license": card_data.get("license") or self._extract_license_from_tags(tag_list),
                "base_model": card_data.get("base_model"),
                "canonical_url": canonical_url,
                "evidence": self.build_evidence(
                    source_id=f"hf:{repo_id}",
                    source_type="huggingface",
                    source_url=canonical_url,
                    publication_date=created_at,
                    confidence="EVALUATED",
                    raw_payload_ref=repo_id,
                ),
            }
            records.append(record)

        return records

    @staticmethod
    def _extract_license_from_tags(tags: List[str]) -> Optional[str]:
        for tag in tags:
            if tag.startswith("license:"):
                return tag.split("license:", 1)[1]
        return None

    def _get_mock_records(self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        High-fidelity fixture models representing landmark recent open and accessible models.
        """
        fixtures = [
            {
                "source_id": "hf:deepseek-ai/DeepSeek-R1",
                "repo_id": "deepseek-ai/DeepSeek-R1",
                "model_name": "DeepSeek-R1",
                "author": "deepseek-ai",
                "pipeline_tag": "text-generation",
                "downloads": 4825900,
                "likes": 22940,
                "tags": [
                    "text-generation",
                    "reasoning",
                    "reinforcement-learning",
                    "deepseek",
                    "moe",
                    "license:mit",
                    "arxiv:2501.12948",
                ],
                "created_at": "2025-01-20T09:12:00Z",
                "last_modified": "2025-01-24T18:30:00Z",
                "license": "mit",
                "architecture": "DeepseekV3ForCausalLM (MoE)",
                "parameter_count_total_b": 671,
                "parameter_count_active_b": 37,
                "context_window_tokens": 128000,
                "canonical_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1",
                "card_data": {
                    "base_model": "deepseek-ai/DeepSeek-V3-Base",
                    "license": "mit",
                    "language": ["en", "zh"],
                },
            },
            {
                "source_id": "hf:deepseek-ai/DeepSeek-V3",
                "repo_id": "deepseek-ai/DeepSeek-V3",
                "model_name": "DeepSeek-V3",
                "author": "deepseek-ai",
                "pipeline_tag": "text-generation",
                "downloads": 3140200,
                "likes": 11500,
                "tags": [
                    "text-generation",
                    "deepseek",
                    "moe",
                    "multi-token-prediction",
                    "license:mit",
                    "arxiv:2412.19437",
                ],
                "created_at": "2024-12-26T04:00:00Z",
                "last_modified": "2024-12-28T12:00:00Z",
                "license": "mit",
                "architecture": "DeepseekV3ForCausalLM (MoE)",
                "parameter_count_total_b": 671,
                "parameter_count_active_b": 37,
                "context_window_tokens": 128000,
                "canonical_url": "https://huggingface.co/deepseek-ai/DeepSeek-V3",
                "card_data": {"license": "mit"},
            },
            {
                "source_id": "hf:meta-llama/Llama-3.3-70B-Instruct",
                "repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "model_name": "Llama-3.3-70B-Instruct",
                "author": "meta-llama",
                "pipeline_tag": "text-generation",
                "downloads": 5821000,
                "likes": 8450,
                "tags": [
                    "text-generation",
                    "llama",
                    "llama-3.3",
                    "meta",
                    "license:llama3.3",
                ],
                "created_at": "2024-12-06T17:00:00Z",
                "last_modified": "2024-12-10T11:20:00Z",
                "license": "llama3.3",
                "architecture": "LlamaForCausalLM (Dense)",
                "parameter_count_total_b": 70,
                "parameter_count_active_b": 70,
                "context_window_tokens": 128000,
                "canonical_url": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
                "card_data": {"license": "llama3.3"},
            },
            {
                "source_id": "hf:Qwen/Qwen2.5-72B-Instruct",
                "repo_id": "Qwen/Qwen2.5-72B-Instruct",
                "model_name": "Qwen2.5-72B-Instruct",
                "author": "Qwen",
                "pipeline_tag": "text-generation",
                "downloads": 4120300,
                "likes": 9820,
                "tags": [
                    "text-generation",
                    "qwen",
                    "qwen2.5",
                    "alibaba",
                    "license:apache-2.0",
                ],
                "created_at": "2024-09-18T12:00:00Z",
                "last_modified": "2024-10-15T09:00:00Z",
                "license": "apache-2.0",
                "architecture": "Qwen2ForCausalLM (Dense)",
                "parameter_count_total_b": 72.7,
                "parameter_count_active_b": 72.7,
                "context_window_tokens": 131072,
                "canonical_url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
                "card_data": {"license": "apache-2.0"},
            },
            {
                "source_id": "hf:Qwen/Qwen2.5-Coder-32B-Instruct",
                "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "model_name": "Qwen2.5-Coder-32B-Instruct",
                "author": "Qwen",
                "pipeline_tag": "text-generation",
                "downloads": 3890100,
                "likes": 7650,
                "tags": [
                    "text-generation",
                    "code",
                    "qwen",
                    "qwen2.5-coder",
                    "license:apache-2.0",
                ],
                "created_at": "2024-11-12T08:00:00Z",
                "last_modified": "2024-11-20T14:10:00Z",
                "license": "apache-2.0",
                "architecture": "Qwen2ForCausalLM (Dense)",
                "parameter_count_total_b": 32.5,
                "parameter_count_active_b": 32.5,
                "context_window_tokens": 131072,
                "canonical_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
                "card_data": {"license": "apache-2.0"},
            },
            {
                "source_id": "hf:mistralai/Mistral-Large-Instruct-2411",
                "repo_id": "mistralai/Mistral-Large-Instruct-2411",
                "model_name": "Mistral-Large-Instruct-2411",
                "author": "mistralai",
                "pipeline_tag": "text-generation",
                "downloads": 1285000,
                "likes": 4210,
                "tags": [
                    "text-generation",
                    "mistral",
                    "mistral-large",
                    "function-calling",
                    "license:mistral-research",
                ],
                "created_at": "2024-11-18T10:00:00Z",
                "last_modified": "2024-11-22T16:00:00Z",
                "license": "mistral-research",
                "architecture": "MistralForCausalLM (Dense)",
                "parameter_count_total_b": 123,
                "parameter_count_active_b": 123,
                "context_window_tokens": 128000,
                "canonical_url": "https://huggingface.co/mistralai/Mistral-Large-Instruct-2411",
                "card_data": {"license": "mistral-research"},
            },
            {
                "source_id": "hf:microsoft/phi-4",
                "repo_id": "microsoft/phi-4",
                "model_name": "phi-4",
                "author": "microsoft",
                "pipeline_tag": "text-generation",
                "downloads": 1920400,
                "likes": 5600,
                "tags": [
                    "text-generation",
                    "phi-4",
                    "microsoft",
                    "slm",
                    "math-reasoning",
                    "license:mit",
                ],
                "created_at": "2024-12-12T16:00:00Z",
                "last_modified": "2024-12-18T09:30:00Z",
                "license": "mit",
                "architecture": "Phi3ForCausalLM (Dense)",
                "parameter_count_total_b": 14,
                "parameter_count_active_b": 14,
                "context_window_tokens": 16384,
                "canonical_url": "https://huggingface.co/microsoft/phi-4",
                "card_data": {"license": "mit"},
            },
            {
                "source_id": "hf:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "model_name": "DeepSeek-R1-Distill-Qwen-32B",
                "author": "deepseek-ai",
                "pipeline_tag": "text-generation",
                "downloads": 2740000,
                "likes": 8920,
                "tags": [
                    "text-generation",
                    "distill",
                    "qwen",
                    "deepseek-r1",
                    "reasoning",
                    "license:mit",
                ],
                "created_at": "2025-01-20T10:00:00Z",
                "last_modified": "2025-01-24T12:00:00Z",
                "license": "mit",
                "architecture": "Qwen2ForCausalLM (Dense)",
                "parameter_count_total_b": 32.5,
                "parameter_count_active_b": 32.5,
                "context_window_tokens": 128000,
                "canonical_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "card_data": {"base_model": "Qwen/Qwen2.5-32B", "license": "mit"},
            },
            {
                "source_id": "hf:google/gemma-2-27b-it",
                "repo_id": "google/gemma-2-27b-it",
                "model_name": "gemma-2-27b-it",
                "author": "google",
                "pipeline_tag": "text-generation",
                "downloads": 2450000,
                "likes": 6100,
                "tags": [
                    "text-generation",
                    "gemma",
                    "google",
                    "gemma-2",
                    "license:gemma",
                ],
                "created_at": "2024-06-27T14:00:00Z",
                "last_modified": "2024-07-15T11:00:00Z",
                "license": "gemma",
                "architecture": "Gemma2ForCausalLM (Dense)",
                "parameter_count_total_b": 27.2,
                "parameter_count_active_b": 27.2,
                "context_window_tokens": 8192,
                "canonical_url": "https://huggingface.co/google/gemma-2-27b-it",
                "card_data": {"license": "gemma"},
            },
            {
                "source_id": "hf:meta-llama/Llama-3.1-405B-Instruct",
                "repo_id": "meta-llama/Llama-3.1-405B-Instruct",
                "model_name": "Llama-3.1-405B-Instruct",
                "author": "meta-llama",
                "pipeline_tag": "text-generation",
                "downloads": 980000,
                "likes": 5300,
                "tags": [
                    "text-generation",
                    "llama",
                    "llama-3.1",
                    "flagship",
                    "license:llama3.1",
                ],
                "created_at": "2024-07-23T16:00:00Z",
                "last_modified": "2024-08-01T15:00:00Z",
                "license": "llama3.1",
                "architecture": "LlamaForCausalLM (Dense)",
                "parameter_count_total_b": 405,
                "parameter_count_active_b": 405,
                "context_window_tokens": 128000,
                "canonical_url": "https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct",
                "card_data": {"license": "llama3.1"},
            },
        ]

        records = []
        for item in fixtures[:limit]:
            record = dict(item)
            record["evidence"] = self.build_evidence(
                source_id=item["source_id"],
                source_type="huggingface",
                source_url=item["canonical_url"],
                publication_date=item["created_at"],
                confidence="EVALUATED",
                raw_payload_ref=item["repo_id"],
            )
            records.append(record)
        return records


def main():
    parser = argparse.ArgumentParser(description="Collect model metadata from Hugging Face API.")
    parser.add_argument("--mode", choices=["live", "mock"], default="mock", help="Execution mode (default: mock)")
    parser.add_argument("--search", default=None, help="Search keywords")
    parser.add_argument("--author", default=None, help="Author/Organization filter")
    parser.add_argument("--pipeline-tag", default="text-generation", help="Task tag")
    parser.add_argument("--limit", type=int, default=10, help="Number of models to collect")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    collector = HuggingFaceCollector(output_dir=args.output_dir, mode=args.mode)
    result = collector.collect(
        search=args.search,
        author=args.author,
        pipeline_tag=args.pipeline_tag,
        limit=args.limit,
    )
    print(f"Collected {result['metadata']['record_count']} records in mode={args.mode}")


if __name__ == "__main__":
    main()
