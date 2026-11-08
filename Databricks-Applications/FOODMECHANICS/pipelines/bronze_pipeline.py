"""Bronze Pipeline for FOODMECHANICS.

Captures raw incoming records from USDA FoodData Central, PubChem, Crossref,
and culinary experiment sources. Enforces content-hash deduplication and immutable audit trails.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb

logger = logging.getLogger("foodmechanics.bronze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def repair_json_escapes(raw_text: str) -> str:
    """Repair unescaped backslashes in JSON string (e.g. LaTeX formulas like \\Delta, \\gamma)."""
    out = []
    i = 0
    n = len(raw_text)
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    while i < n:
        if raw_text[i] == "\\":
            if i + 1 < n and raw_text[i + 1] in valid_escapes:
                out.append(raw_text[i])
                out.append(raw_text[i + 1])
                i += 2
            else:
                out.append("\\\\")
                i += 1
        else:
            out.append(raw_text[i])
            i += 1
    return "".join(out)


def load_json_safely(file_path: Path) -> Any:
    """Safely parse JSON file with automatic escape sequence repair."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = repair_json_escapes(raw_text)
        return json.loads(repaired)


def compute_content_hash(payload: Union[Dict[str, Any], List[Any], str]) -> str:
    """Compute deterministic SHA-256 hash for raw payload."""
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except Exception:
            canonical = payload
    else:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def init_bronze_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Initialize Bronze layer tables and indexes in DuckDB."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bronze_food_raw (
            source_id VARCHAR PRIMARY KEY,
            source_type VARCHAR NOT NULL,
            payload_json TEXT NOT NULL,
            ingestion_timestamp TIMESTAMP NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            source_url VARCHAR,
            batch_id VARCHAR
        );
        """
    )
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_bronze_hash ON bronze_food_raw(content_hash);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_bronze_type ON bronze_food_raw(source_type);")
    except Exception as e:
        logger.debug("Index notice: %s", e)


def ingest_raw_record(
    con: duckdb.DuckDBPyConnection,
    source_id: str,
    source_type: str,
    payload: Union[Dict[str, Any], List[Any], str],
    source_url: Optional[str] = None,
    batch_id: Optional[str] = None,
    ingestion_timestamp: Optional[datetime.datetime] = None,
) -> Tuple[str, str]:
    """
    Ingest a single raw record into bronze_food_raw with content hash deduplication.
    Returns (source_id, status) where status is 'INSERTED', 'UPDATED', or 'SKIPPED'.
    """
    if ingestion_timestamp is None:
        ingestion_timestamp = datetime.datetime.now(datetime.timezone.utc)

    payload_str = payload if isinstance(payload, str) else json.dumps(payload)
    content_hash = compute_content_hash(payload)

    existing = con.execute(
        "SELECT content_hash FROM bronze_food_raw WHERE source_id = ?",
        [source_id],
    ).fetchone()

    if existing is not None:
        existing_hash = existing[0]
        if existing_hash == content_hash:
            logger.debug("Record %s already exists with identical content_hash. Skipping.", source_id)
            return source_id, "SKIPPED"
        else:
            con.execute(
                """
                UPDATE bronze_food_raw
                SET source_type = ?,
                    payload_json = ?,
                    ingestion_timestamp = ?,
                    content_hash = ?,
                    source_url = ?,
                    batch_id = ?
                WHERE source_id = ?
                """,
                [source_type, payload_str, ingestion_timestamp, content_hash, source_url, batch_id, source_id],
            )
            logger.info("Updated existing record %s with new hash %s", source_id, content_hash[:8])
            return source_id, "UPDATED"

    con.execute(
        """
        INSERT INTO bronze_food_raw (
            source_id, source_type, payload_json, ingestion_timestamp, content_hash, source_url, batch_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [source_id, source_type, payload_str, ingestion_timestamp, content_hash, source_url, batch_id],
    )
    logger.info("Inserted new Bronze record %s (%s)", source_id, source_type)
    return source_id, "INSERTED"


def ingest_fixture_files(
    con: duckdb.DuckDBPyConnection,
    fixtures_dir: Optional[Union[str, Path]] = None,
    batch_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Ingest all fixture JSON files from data/fixtures directory into bronze_food_raw.
    Supports both arrays of records and nested category dictionaries.
    """
    init_bronze_tables(con)

    if fixtures_dir is None:
        fixtures_dir = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    else:
        fixtures_dir = Path(fixtures_dir)

    if not fixtures_dir.exists():
        logger.warning("Fixtures directory %s does not exist.", fixtures_dir)
        return {"inserted": 0, "updated": 0, "skipped": 0}

    if batch_id is None:
        batch_id = f"batch_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    counts = {"inserted": 0, "updated": 0, "skipped": 0}

    for json_file in sorted(fixtures_dir.glob("*.json")):
        try:
            data = load_json_safely(json_file)

            records: List[Dict[str, Any]] = []
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Check for standard multi-entity collections
                known_keys = ["ingredients", "compounds", "mechanisms", "formulation_comparisons", "evidence_records", "formulation_experiments"]
                has_subcollections = any(k in data for k in known_keys)
                if has_subcollections:
                    for k in known_keys:
                        sublist = data.get(k, [])
                        for item in sublist:
                            if isinstance(item, dict):
                                item_copy = dict(item)
                                item_copy["_entity_group"] = k
                                records.append(item_copy)
                else:
                    records = [data]

            logger.info("Processing fixture file %s (%d records)...", json_file.name, len(records))

            for item in records:
                source_id = item.get("source_id") or item.get("id") or item.get("mechanism_id") or item.get("compound_id") or item.get("formulation_id") or item.get("evidence_id")
                if not source_id:
                    source_id = f"file:{json_file.stem}:{compute_content_hash(item)[:12]}"
                source_type = item.get("source_type") or item.get("_entity_group") or f"file_{json_file.stem}"
                source_url = item.get("source_url")

                _, status = ingest_raw_record(
                    con=con,
                    source_id=source_id,
                    source_type=source_type,
                    payload=item,
                    source_url=source_url,
                    batch_id=batch_id,
                )
                counts[status.lower()] += 1
        except Exception as e:
            logger.error("Failed to ingest fixture file %s: %s", json_file, e)
            raise

    logger.info("Bronze ingestion finished: %s", counts)
    return counts


def get_bronze_records(
    con: duckdb.DuckDBPyConnection,
    source_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve raw bronze records as dicts."""
    query = "SELECT source_id, source_type, payload_json, ingestion_timestamp, content_hash, source_url, batch_id FROM bronze_food_raw"
    params = []
    if source_type:
        query += " WHERE source_type = ?"
        params.append(source_type)
    query += " ORDER BY ingestion_timestamp ASC"

    cursor = con.execute(query, params)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    results = []
    for row in rows:
        record = dict(zip(columns, row))
        record["payload"] = json.loads(record["payload_json"])
        results.append(record)
    return results
