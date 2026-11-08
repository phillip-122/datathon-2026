# Integration Contracts & Data Interfaces

## 1. Core Data Contract Architecture

Both applications are decoupled from storage engines via the standard `DataProvider` abstraction.
Any application view (UI, API endpoint, CLI) communicates strictly through this interface.

```python
class DataProvider:
    def get_recent_records(self, limit: int = 20, category: str = None) -> list[dict]:
        """Fetch latest records with timestamp, status, and summary."""
        raise NotImplementedError

    def search(self, query: str, filters: dict = None) -> list[dict]:
        """Search records by keyword, entity type, tags, or date range."""
        raise NotImplementedError

    def get_detail(self, record_id: str) -> dict:
        """Fetch full record payload including provenance, attributes, and raw reference."""
        raise NotImplementedError

    def get_relationships(self, entity_id: str) -> list[dict]:
        """Fetch graph edges connecting this entity to models, tools, papers, mechanisms, etc."""
        raise NotImplementedError

    def get_metrics(self, filters: dict = None) -> dict:
        """Fetch aggregate summary KPIs (counts, trends, distributions)."""
        raise NotImplementedError
```

## 2. Genie Natural Language Interface

Applications access Databricks Genie through the `GenieProvider` interface.
If Databricks is offline or credentials are missing, the UI gracefully enters a clear fallback state explaining that live Genie querying requires an active Databricks workspace session.

```python
class GenieProvider:
    def ask(self, question: str, conversation_id: str = None) -> dict:
        """
        Submits a query to the Databricks Genie Space/Agent.
        Returns:
            {
                "conversation_id": str,
                "status": "COMPLETED" | "EXECUTING" | "ERROR",
                "answer_text": str,
                "generated_sql": str | None,
                "query_results": list[dict] | None,
                "evidence_sources": list[dict]
            }
        """
        raise NotImplementedError
```

## 3. Mandatory Evidence & Provenance Schema

Every Silver and Gold entity or relationship record MUST preserve evidence attribution:

| Field | Type | Description |
|---|---|---|
| `source_id` | string | Unique source identifier (e.g. `arxiv:2401.12345`, `usda:171234`) |
| `source_type` | string | Source origin (e.g. `arxiv`, `huggingface`, `github`, `usda_fdc`, `pubchem`) |
| `source_url` | string | Canonical web URL of origin |
| `publication_date` | timestamp/string | Official publication or release date |
| `ingestion_timestamp` | timestamp/string | When this record was ingested into Bronze |
| `extraction_timestamp` | timestamp/string | When entities/attributes were parsed into Silver |
| `evidence_confidence` | string/float | `ANNOUNCEMENT`, `EVALUATED`, `PEER_REVIEWED`, `HYPOTHESIS` |
| `raw_payload_ref` | string | Location or hash of original bronze record |

## 4. Serving Basis Normalization Contract (FOODMECHANICS)
- All nutritional comparisons across ingredients MUST be normalized to a standard `100g` basis.
- Serving sizes and volume-to-mass conversion factors must be documented explicitly.
- Inferred properties must be flagged with `is_inferred: true`.
