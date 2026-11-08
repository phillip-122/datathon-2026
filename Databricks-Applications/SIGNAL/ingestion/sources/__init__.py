"""
SIGNAL Ingestion Sources Module
Data acquisition collectors for AI landscape intelligence.
"""

from .base_collector import BaseCollector
from .arxiv_collector import ArxivCollector
from .huggingface_collector import HuggingFaceCollector
from .github_collector import GitHubCollector
from .collect_all import run_all_collectors

__all__ = [
    "BaseCollector",
    "ArxivCollector",
    "HuggingFaceCollector",
    "GitHubCollector",
    "run_all_collectors",
]
