"""
SIGNAL Base Ingestion Collector
Abstract base class providing standardized raw output formatting, provenance tagging,
and execution modes (live vs. mock).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class BaseCollector(ABC):
    """
    Base collector for all SIGNAL external intelligence sources.
    Standardizes payload envelopes, provenance tracking, and offline/mock operation.
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        mode: str = "mock",
        timeout: int = 15,
    ):
        """
        Args:
            output_dir: Destination directory for raw Bronze payloads.
                        Defaults to SIGNAL/data/raw/{source_name}.
            mode: 'mock' for deterministically generated offline responses,
                  'live' for live HTTP requests to upstream APIs.
            timeout: HTTP timeout in seconds for live calls.
        """
        self.mode = mode.lower()
        if self.mode not in ("live", "mock"):
            raise ValueError(f"Invalid mode '{mode}'. Choose 'live' or 'mock'.")

        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

        default_base = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        self.output_dir = Path(output_dir) if output_dir else default_base / self.source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this collector source (e.g. 'arxiv', 'huggingface', 'github')."""
        pass

    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """
        Execute data acquisition.
        Returns the standardized payload envelope containing metadata and records.
        """
        pass

    def build_envelope(
        self,
        records: List[Dict[str, Any]],
        query_params: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Wraps extracted records into a standardized Raw Bronze envelope.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = {
            "source_type": self.source_name,
            "collection_timestamp": now_iso,
            "mode": self.mode,
            "record_count": len(records),
            "query_parameters": query_params or {},
            "collector_version": "1.0.0",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return {
            "metadata": metadata,
            "records": records,
        }

    def save_raw(
        self,
        envelope: Dict[str, Any],
        filename_prefix: Optional[str] = None,
    ) -> Path:
        """
        Persists the raw envelope to disk as pretty JSON.
        """
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix or f"{self.source_name}_raw"
        filename = f"{prefix}_{self.mode}_{timestamp_slug}.json"
        target_path = self.output_dir / filename

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Saved {envelope['metadata']['record_count']} raw records to {target_path}"
        )
        return target_path

    @staticmethod
    def build_evidence(
        source_id: str,
        source_type: str,
        source_url: str,
        publication_date: Optional[str] = None,
        confidence: str = "ANNOUNCEMENT",
        raw_payload_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Standardizes evidence records conforming to coordination/contracts/provider.py.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "source_id": source_id,
            "source_type": source_type,
            "source_url": source_url,
            "publication_date": publication_date,
            "ingestion_timestamp": now_iso,
            "extraction_timestamp": now_iso,
            "evidence_confidence": confidence,
            "raw_payload_ref": raw_payload_ref,
        }
