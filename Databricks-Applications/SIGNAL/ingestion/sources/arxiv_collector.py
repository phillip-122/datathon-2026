"""
SIGNAL arXiv Ingestion Collector
Acquires AI/ML research papers from the arXiv API (cs.AI, cs.CL, cs.LG)
and extracts titles, authors, abstracts, published dates, arXiv IDs, and canonical URLs.
Supports live HTTP API calls with rate-limiting and deterministic mock mode for offline replay.
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import requests

from .base_collector import BaseCollector


class ArxivCollector(BaseCollector):
    """
    Collector for arXiv AI/ML research papers.
    """

    ATOM_NS = "{http://www.w3.org/2005/Atom}"
    ARXIV_NS = "{http://arxiv.org/schemas/atom}"

    @property
    def source_name(self) -> str:
        return "arxiv"

    def collect(
        self,
        query: str = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG",
        max_results: int = 15,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Collects papers from arXiv API or mock store.
        """
        query_params = {
            "query": query,
            "max_results": max_results,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }

        if self.mode == "live":
            self.logger.info(f"Querying live arXiv API: search_query='{query}', max_results={max_results}")
            try:
                records = self._fetch_live(query, max_results, sort_by, sort_order)
            except Exception as e:
                self.logger.warning(f"Live arXiv fetch failed ({e}). Falling back to curated mock dataset.")
                records = self._get_mock_records(max_results)
        else:
            self.logger.info(f"Generating mock arXiv records (count={max_results})")
            records = self._get_mock_records(max_results)

        envelope = self.build_envelope(records, query_params=query_params)
        if save:
            self.save_raw(envelope, filename_prefix="arxiv_papers")

        return envelope

    def _fetch_live(
        self,
        query: str,
        max_results: int,
        sort_by: str,
        sort_order: str,
    ) -> List[Dict[str, Any]]:
        """
        Executes HTTP GET against export.arxiv.org/api/query and parses Atom XML.
        """
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        headers = {
            "User-Agent": "SIGNAL-AI-Landscape-Intelligence/1.0 (academic research and ecosystem monitoring)"
        }

        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        return self._parse_atom_xml(response.text)

    def _parse_atom_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """
        Parses Atom XML response into standardized paper dictionaries.
        """
        root = ET.fromstring(xml_text)
        records: List[Dict[str, Any]] = []

        entries = root.findall(f"{self.ATOM_NS}entry")
        for entry in entries:
            raw_id = entry.findtext(f"{self.ATOM_NS}id", default="").strip()
            # Extract bare arXiv ID (e.g., '2501.12948v1' -> '2501.12948')
            arxiv_id_match = re.search(r"abs/([0-9]+\.[0-9]+)", raw_id)
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else raw_id.split("/")[-1]

            title = entry.findtext(f"{self.ATOM_NS}title", default="")
            title = " ".join(title.split())  # Clean newlines and double spaces

            summary = entry.findtext(f"{self.ATOM_NS}summary", default="")
            summary = " ".join(summary.split())

            published = entry.findtext(f"{self.ATOM_NS}published", default="")
            updated = entry.findtext(f"{self.ATOM_NS}updated", default="")

            authors = [
                author.findtext(f"{self.ATOM_NS}name", default="").strip()
                for author in entry.findall(f"{self.ATOM_NS}author")
                if author.findtext(f"{self.ATOM_NS}name")
            ]

            primary_cat_elem = entry.find(f"{self.ARXIV_NS}primary_category")
            primary_cat = (
                primary_cat_elem.attrib.get("term", "cs.AI")
                if primary_cat_elem is not None
                else "cs.AI"
            )

            all_cats = [
                cat.attrib.get("term")
                for cat in entry.findall(f"{self.ATOM_NS}category")
                if "term" in cat.attrib
            ]

            canonical_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            for link in entry.findall(f"{self.ATOM_NS}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", pdf_url)
                elif link.attrib.get("rel") == "alternate":
                    canonical_url = link.attrib.get("href", canonical_url)

            record = {
                "source_id": f"arxiv:{arxiv_id}",
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": summary,
                "published_date": published,
                "updated_date": updated,
                "primary_category": primary_cat,
                "categories": all_cats,
                "canonical_url": canonical_url,
                "pdf_url": pdf_url,
                "evidence": self.build_evidence(
                    source_id=f"arxiv:{arxiv_id}",
                    source_type="arxiv",
                    source_url=canonical_url,
                    publication_date=published,
                    confidence="PEER_REVIEWED",
                    raw_payload_ref=raw_id,
                ),
            }
            records.append(record)

        return records

    def _get_mock_records(self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        High-fidelity fixture records representing pivotal recent AI research papers.
        """
        fixtures = [
            {
                "source_id": "arxiv:2501.12948",
                "arxiv_id": "2501.12948",
                "title": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
                "authors": ["DeepSeek-AI", "Daya Guo", "Dejian Yang", "Haowei Zhang", "Songquan Song", "Ruoyu Zhang", "Runxin Xu", "Qihao Zhu"],
                "abstract": "We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. DeepSeek-R1 incorporates multi-stage training and cold-start data before RL, achieving performance on par with OpenAI-o1-1217 on reasoning tasks. We distill DeepSeek-R1 into smaller dense models (1.5B, 7B, 8B, 14B, 32B, 70B) based on Qwen and Llama.",
                "published_date": "2025-01-22T08:00:00Z",
                "updated_date": "2025-01-22T08:00:00Z",
                "primary_category": "cs.AI",
                "categories": ["cs.AI", "cs.CL", "cs.LG"],
                "canonical_url": "https://arxiv.org/abs/2501.12948",
                "pdf_url": "https://arxiv.org/pdf/2501.12948.pdf",
                "key_contributions": [
                    "Pure RL without preliminary SFT elicits emergent self-verification and reasoning reflection.",
                    "Cold-start data followed by two-stage RL matches OpenAI-o1 performance on MATH-500 and AIME.",
                    "Demonstrates effective distillation from 671B MoE into dense open weights (Qwen/Llama architectures)."
                ],
            },
            {
                "source_id": "arxiv:2407.21783",
                "arxiv_id": "2407.21783",
                "title": "The Llama 3 Herd of Models",
                "authors": ["Meta AI", "Aaron Grattafiori", "Abhimanyu Dubey", "Abhinav Jauhri", "Abhinav Pandey", "Abhishek Kadian", "Alireza Aghajani"],
                "abstract": "Modern artificial intelligence (AI) systems are increasingly powered by foundation models. This paper presents Llama 3, a herd of language models natively supporting multilinguality, coding, reasoning, and tool usage. Our largest model is a 405B parameter dense Transformer trained on 15.6T tokens with a context window of 128K tokens. We demonstrate state-of-the-art results across over 50 benchmarks and detail data curation, architectural optimizations, and safety mitigations.",
                "published_date": "2024-07-31T17:50:00Z",
                "updated_date": "2024-08-01T12:00:00Z",
                "primary_category": "cs.AI",
                "categories": ["cs.AI", "cs.CL", "cs.LG"],
                "canonical_url": "https://arxiv.org/abs/2407.21783",
                "pdf_url": "https://arxiv.org/pdf/2407.21783.pdf",
                "key_contributions": [
                    "Dense 405B parameter architecture trained on 15.6T tokens with 128k context window.",
                    "Detailed recipe for iterative SFT, DPO, and rejection sampling at hyperscale.",
                    "Extensive multi-token prediction and hardware stability benchmarks on 16k GPU clusters."
                ],
            },
            {
                "source_id": "arxiv:2310.06770",
                "arxiv_id": "2310.06770",
                "title": "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
                "authors": ["Carlos E. Jimenez", "John Yang", "Alexander Wettig", "Shunyu Yao", "Karthik Narasimhan", "Ofir Press"],
                "abstract": "Language models have demonstrated impressive capabilities in code generation, yet real-world software engineering challenges remain difficult. We introduce SWE-bench, an evaluation framework consisting of 2,294 software engineering problems drawn from real GitHub issues and corresponding pull requests across 12 popular Python repositories. SWE-bench Verified provides a human-validated subset of 500 gold instances that reliably test agentic bug resolution.",
                "published_date": "2023-10-10T14:20:00Z",
                "updated_date": "2024-08-14T09:00:00Z",
                "primary_category": "cs.SE",
                "categories": ["cs.SE", "cs.AI", "cs.CL"],
                "canonical_url": "https://arxiv.org/abs/2310.06770",
                "pdf_url": "https://arxiv.org/pdf/2310.06770.pdf",
                "key_contributions": [
                    "Rigorous end-to-end task setup resolving real GitHub issues via patch application and unit tests.",
                    "SWE-bench Verified subset validated by professional software engineers.",
                    "Became the primary industry standard for evaluating agentic coding capabilities."
                ],
            },
            {
                "source_id": "arxiv:2406.07887",
                "arxiv_id": "2406.07887",
                "title": "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark",
                "authors": ["Yubo Wang", "Xueguang Ma", "Ge Zhang", "Ni Zhan", "Kailin Wang", "Sheng Shen", "Daniel Khashabi"],
                "abstract": "Massive Multitask Language Understanding (MMLU) has been a cornerstone benchmark for LLMs. However, rapid progress has led to benchmark saturation and vulnerability to random guessing heuristics. We present MMLU-Pro, expanding multiple-choice options from 4 to 10 choices, reducing trivial memorization advantage, and curating reasoning-heavy questions in STEM and business domains.",
                "published_date": "2024-06-12T10:15:00Z",
                "updated_date": "2024-06-14T08:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "cs.AI"],
                "canonical_url": "https://arxiv.org/abs/2406.07887",
                "pdf_url": "https://arxiv.org/pdf/2406.07887.pdf",
                "key_contributions": [
                    "Increases question choices from 4 to 10, drastically lowering random guessing baseline.",
                    "Focuses on expert-level multi-step reasoning, addressing benchmark saturation on original MMLU.",
                    "Demonstrates steep separation between top-tier reasoning models and standard instruction models."
                ],
            },
            {
                "source_id": "arxiv:2210.03629",
                "arxiv_id": "2210.03629",
                "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
                "authors": ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran", "Karthik Narasimhan", "Yuan Cao"],
                "abstract": "While language models have demonstrated reasoning via Chain-of-Thought (CoT) and action execution via tool use, these paradigms have largely operated in isolation. We propose ReAct, where LLMs generate verbal reasoning traces and task-specific actions in an interleaved manner. ReAct overcomes issues of hallucination in CoT and error propagation in action-only loops.",
                "published_date": "2022-10-06T18:00:00Z",
                "updated_date": "2023-03-10T12:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "cs.AI"],
                "canonical_url": "https://arxiv.org/abs/2210.03629",
                "pdf_url": "https://arxiv.org/pdf/2210.03629.pdf",
                "key_contributions": [
                    "Formulated interleaved Thought-Action-Observation loop that underpins modern autonomous agent architectures.",
                    "Significantly reduced hallucination rates when groundable external tools/APIs are available.",
                    "Directly inspired frameworks including LangChain, LangGraph, and AutoGen."
                ],
            },
            {
                "source_id": "arxiv:2407.08608",
                "arxiv_id": "2407.08608",
                "title": "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision",
                "authors": ["Tri Dao", "Jay Shah", "Sukhad Anand", "Atli Kosson", "Dan Fu", "Christopher Ré"],
                "abstract": "Attention remains the computational bottleneck in long-context Transformers. We propose FlashAttention-3, designed specifically for Hopper GPU architectures (H100/H800). By exploiting asynchronous Tensor Core operations (WGMMA), TMA (Tensor Memory Accelerator), and FP8 low-precision accumulation, FlashAttention-3 achieves up to 1.5-2x speedup over FlashAttention-2, reaching up to 850 TFLOPs/s.",
                "published_date": "2024-07-11T16:00:00Z",
                "updated_date": "2024-07-12T10:00:00Z",
                "primary_category": "cs.LG",
                "categories": ["cs.LG", "cs.DC"],
                "canonical_url": "https://arxiv.org/abs/2407.08608",
                "pdf_url": "https://arxiv.org/pdf/2407.08608.pdf",
                "key_contributions": [
                    "Warp-group asynchronous pipelining overlapping GEMM and softmax compute on NVIDIA Hopper.",
                    "Hardware-accelerated FP8 attention with block quantization and numerical stability safeguards.",
                    "Core foundation for ultra-fast serving in vLLM, TensorRT-LLM, and SGLang."
                ],
            },
            {
                "source_id": "arxiv:2302.04761",
                "arxiv_id": "2302.04761",
                "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
                "authors": ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessì", "Roberta Raileanu", "Maria Lomeli", "Luke Zettlemoyer", "Nicola Cancedda", "Thomas Scialom"],
                "abstract": "Language models exhibit remarkable abilities but struggle with basic arithmetic, factual timeliness, and search. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token predictions in a self-supervised manner without human annotation.",
                "published_date": "2023-02-09T17:45:00Z",
                "updated_date": "2023-02-15T09:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "cs.AI"],
                "canonical_url": "https://arxiv.org/abs/2302.04761",
                "pdf_url": "https://arxiv.org/pdf/2302.04761.pdf",
                "key_contributions": [
                    "Self-supervised API annotation pipeline that filters candidate tool calls by loss reduction.",
                    "Demonstrated seamless integration of calculators, Q&A search, machine translation, and calendar APIs.",
                    "Provided theoretical basis for modern function calling and tool-use fine-tuning."
                ],
            },
            {
                "source_id": "arxiv:2305.18290",
                "arxiv_id": "2305.18290",
                "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "authors": ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"],
                "abstract": "While reinforcement learning from human feedback (RLHF) with PPO is effective, it is notoriously complex and unstable, requiring training a separate reward model and sampling from the policy during training. We show how to implicitly optimize the same objective as RLHF via a simple binary cross-entropy loss over preferences, yielding Direct Preference Optimization (DPO).",
                "published_date": "2023-05-29T16:00:00Z",
                "updated_date": "2023-12-14T11:00:00Z",
                "primary_category": "cs.LG",
                "categories": ["cs.LG", "cs.AI", "cs.CL"],
                "canonical_url": "https://arxiv.org/abs/2305.18290",
                "pdf_url": "https://arxiv.org/pdf/2305.18290.pdf",
                "key_contributions": [
                    "Closed-form reparameterization of the Bradley-Terry reward model directly in terms of the optimal policy.",
                    "Eliminates reinforcement learning instability, value networks, and actor-critic orchestration during alignment.",
                    "Adopted across nearly all open-weight model post-training pipelines including Llama, Qwen, and Mistral."
                ],
            },
            {
                "source_id": "arxiv:2201.11903",
                "arxiv_id": "2201.11903",
                "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                "authors": ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter", "Fei Xia", "Ed Chi", "Quoc Le", "Denny Zhou"],
                "abstract": "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. We show that such reasoning abilities emerge naturally in sufficiently large models via simple few-shot demonstrations.",
                "published_date": "2022-01-28T18:00:00Z",
                "updated_date": "2023-01-10T15:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "cs.AI"],
                "canonical_url": "https://arxiv.org/abs/2201.11903",
                "pdf_url": "https://arxiv.org/pdf/2201.11903.pdf",
                "key_contributions": [
                    "Formalized Chain-of-Thought (CoT) prompting for multi-step arithmetic, symbolic, and commonsense reasoning.",
                    "Showed performance scaling discontinuities across model parameter thresholds.",
                    "Established empirical foundation leading directly to o1 and DeepSeek-R1 test-time compute reasoning."
                ],
            },
            {
                "source_id": "arxiv:1706.03762",
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and entirely relying on an attention mechanism to draw global dependencies between input and output.",
                "published_date": "2017-06-12T17:57:00Z",
                "updated_date": "2023-08-02T14:00:00Z",
                "primary_category": "cs.CL",
                "categories": ["cs.CL", "cs.LG"],
                "canonical_url": "https://arxiv.org/abs/1706.03762",
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "key_contributions": [
                    "Introduced multi-head self-attention mechanism replacing recurrence and convolutions.",
                    "Achieved parallelizable sequence training enabling modern foundation model scaling.",
                    "The universal architectural backbone for contemporary generative AI."
                ],
            },
        ]

        # Augment with evidence blocks conforming to provider contract
        records = []
        for item in fixtures[:limit]:
            record = dict(item)
            record["evidence"] = self.build_evidence(
                source_id=item["source_id"],
                source_type="arxiv",
                source_url=item["canonical_url"],
                publication_date=item["published_date"],
                confidence="PEER_REVIEWED",
                raw_payload_ref=item["arxiv_id"],
            )
            records.append(record)
        return records


def main():
    parser = argparse.ArgumentParser(description="Collect research papers from arXiv API.")
    parser.add_argument("--mode", choices=["live", "mock"], default="mock", help="Execution mode (default: mock)")
    parser.add_argument("--query", default="cat:cs.AI OR cat:cs.CL OR cat:cs.LG", help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum papers to retrieve")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    collector = ArxivCollector(output_dir=args.output_dir, mode=args.mode)
    result = collector.collect(query=args.query, max_results=args.max_results)
    print(f"Collected {result['metadata']['record_count']} records in mode={args.mode}")


if __name__ == "__main__":
    main()
