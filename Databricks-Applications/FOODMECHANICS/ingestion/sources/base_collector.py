"""
FOODMECHANICS Ingestion - Base Collector
Standardized foundation for all food science data collectors.
Supports live API calls with rate-limiting, SSL fallback, raw payload capture,
and offline mock/fixture replay.
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests
import urllib3

# Suppress insecure request warnings if corporate SSL inspection requires verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class BaseFoodCollector(ABC):
    """
    Abstract base collector enforcing data contracts, raw capture, and dual live/mock execution.
    """

    def __init__(
        self,
        source_name: str,
        base_url: str,
        rate_limit_delay_sec: float = 0.5,
        verify_ssl: bool = False,
        raw_output_dir: Optional[str] = None
    ):
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay_sec = rate_limit_delay_sec
        self.verify_ssl = verify_ssl

        # Resolve paths relative to FOODMECHANICS
        current_dir = os.path.dirname(os.path.abspath(__file__))
        foodmechanics_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        self.raw_output_dir = raw_output_dir or os.path.join(foodmechanics_dir, "data", "raw", self.source_name)
        os.makedirs(self.raw_output_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"FoodMechanics-ResearchBot/1.0 (+https://github.com/datathon2026/foodmechanics)",
            "Accept": "application/json"
        })

    def _get_timestamp_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def fetch_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        mock_fallback_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute an HTTP GET request with error handling, rate limiting, and optional mock fallback.
        """
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}/{endpoint.lstrip('/')}"
        time.sleep(self.rate_limit_delay_sec)

        try:
            resp = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                verify=self.verify_ssl
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "success",
                "http_status": resp.status_code,
                "url": resp.url,
                "payload": data,
                "fetched_at": self._get_timestamp_iso()
            }
        except Exception as exc:
            logger.warning(f"[{self.source_name}] Failed to fetch from {url}: {exc}")
            if mock_fallback_func:
                logger.info(f"[{self.source_name}] Using mock fallback for {url}")
                mock_data = mock_fallback_func(endpoint, params)
                return {
                    "status": "mock_fallback",
                    "http_status": 200,
                    "url": url,
                    "payload": mock_data,
                    "fetched_at": self._get_timestamp_iso(),
                    "warning": str(exc)
                }
            raise

    def save_raw_record(self, item_id: str, raw_data: Dict[str, Any]) -> str:
        """
        Saves the raw JSON response payload preserving full metadata and timestamp.
        Returns the absolute filepath of the saved raw record.
        """
        clean_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in item_id)
        filename = f"{clean_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.raw_output_dir, filename)

        record_wrapper = {
            "source_name": self.source_name,
            "item_id": item_id,
            "captured_at": self._get_timestamp_iso(),
            "data": raw_data
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record_wrapper, f, indent=2, ensure_ascii=False)

        return filepath

    def build_evidence_record(
        self,
        source_id: str,
        source_url: str,
        confidence: str = "PEER_REVIEWED",
        publication_date: Optional[str] = None,
        raw_payload_ref: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enforces coordination/INTEGRATION_CONTRACT.md EvidenceRecord structure.
        Confidence must be one of: ANNOUNCEMENT, EVALUATED, PEER_REVIEWED, HYPOTHESIS, EMPIRICAL.
        """
        valid_confidences = {"ANNOUNCEMENT", "EVALUATED", "PEER_REVIEWED", "HYPOTHESIS", "EMPIRICAL"}
        if confidence not in valid_confidences:
            confidence = "EMPIRICAL"

        return {
            "source_id": source_id,
            "source_type": self.source_name,
            "source_url": source_url,
            "publication_date": publication_date or "2024-01-01",
            "ingestion_timestamp": self._get_timestamp_iso(),
            "extraction_timestamp": self._get_timestamp_iso(),
            "evidence_confidence": confidence,
            "raw_payload_ref": raw_payload_ref
        }

    @abstractmethod
    def collect(self, query_or_ids: List[str], dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Main collection entry point to be implemented by each specific source collector.
        """
        pass
