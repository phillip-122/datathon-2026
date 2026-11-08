"""
coordination/contracts/provider.py
Shared Abstract Interfaces and Models for Data and Genie Providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    source_id: str
    source_type: str
    source_url: str
    publication_date: Optional[str] = None
    ingestion_timestamp: str
    extraction_timestamp: Optional[str] = None
    evidence_confidence: str = Field(
        default="ANNOUNCEMENT",
        description="One of: ANNOUNCEMENT, EVALUATED, PEER_REVIEWED, HYPOTHESIS, EMPIRICAL"
    )
    raw_payload_ref: Optional[str] = None


class EntityRelationship(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    strength: float = 1.0
    evidence: Optional[EvidenceRecord] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class GenieResponse(BaseModel):
    conversation_id: str
    status: str  # "COMPLETED", "EXECUTING", "ERROR", "UNAVAILABLE"
    answer_text: str
    generated_sql: Optional[str] = None
    query_results: Optional[List[Dict[str, Any]]] = None
    evidence_sources: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None


class DataProvider(ABC):
    @abstractmethod
    def get_recent_records(self, limit: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent developments or ingredient/mechanism records."""
        pass

    @abstractmethod
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search records matching keyword and optional facet filters."""
        pass

    @abstractmethod
    def get_detail(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Fetch complete detail for a single entity/record."""
        pass

    @abstractmethod
    def get_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Fetch adjacent graph nodes and edges."""
        pass

    @abstractmethod
    def get_metrics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch high-level KPIs and summary statistics."""
        pass


class GenieProvider(ABC):
    @abstractmethod
    def ask(self, question: str, conversation_id: Optional[str] = None) -> GenieResponse:
        """Submit question to Genie Agent or fallback if unconfigured."""
        pass
