"""
SIGNAL Bronze Ingestion Pipeline
Captures raw payloads from external sources (arXiv, Hugging Face, GitHub, Announcements),
computes canonical SHA-256 content hashes, handles idempotent deduplication,
and manages incremental ingestion checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb

logger = logging.getLogger("signal.pipelines.bronze")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def canonical_json_bytes(payload: Union[Dict[str, Any], List[Any], str]) -> bytes:
    """
    Serializes a payload to canonical JSON bytes with sorted keys and compact separators
    to guarantee deterministic hashing across varying key orders and whitespace.
    """
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        except json.JSONDecodeError:
            return payload.strip().encode("utf-8")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_content_hash(payload: Union[Dict[str, Any], List[Any], str]) -> str:
    """Calculates SHA-256 digest of canonical payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class BronzePipeline:
    """
    Manages Bronze raw table storage and incremental ingestion with deduplication.
    Compatible with DuckDB and Databricks SQL.
    """

    def __init__(self, db_path: Optional[str] = None, connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initializes BronzePipeline with an existing DuckDB connection or opens/creates
        a database at db_path. If both are None, uses in-memory DuckDB.
        """
        if connection is not None:
            self.con = connection
        elif db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(db_path)
        else:
            self.con = duckdb.connect(":memory:")

        self.init_tables()

    def init_tables(self) -> None:
        """Creates bronze tables and indexes if they do not exist."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS bronze_raw_feeds (
                source_id VARCHAR NOT NULL,
                source_type VARCHAR NOT NULL,
                payload_json VARCHAR NOT NULL,
                ingestion_timestamp TIMESTAMP NOT NULL,
                content_hash VARCHAR NOT NULL,
                PRIMARY KEY (source_id, content_hash)
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS bronze_checkpoints (
                feed_name VARCHAR PRIMARY KEY,
                last_ingested_at TIMESTAMP NOT NULL,
                record_count BIGINT NOT NULL DEFAULT 0,
                checkpoint_offset VARCHAR
            );
        """)

        # Auxiliary index for fast hash lookups
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_bronze_hash ON bronze_raw_feeds (content_hash);
        """)

    def is_duplicate(self, source_id: str, content_hash: str) -> bool:
        """Checks if a (source_id, content_hash) tuple already exists in bronze_raw_feeds."""
        res = self.con.execute(
            "SELECT 1 FROM bronze_raw_feeds WHERE source_id = ? AND content_hash = ? LIMIT 1",
            [source_id, content_hash],
        ).fetchone()
        return res is not None

    def ingest_record(
        self,
        source_id: str,
        source_type: str,
        payload: Union[Dict[str, Any], str],
        ingestion_timestamp: Optional[datetime] = None,
        feed_name: Optional[str] = None,
        checkpoint_offset: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Ingests a single raw record if not duplicate.
        Returns:
            Tuple[bool, str]: (was_inserted, content_hash)
        """
        if ingestion_timestamp is None:
            ingestion_timestamp = datetime.now(timezone.utc)

        content_hash = compute_content_hash(payload)

        # Idempotency check
        if self.is_duplicate(source_id, content_hash):
            logger.debug(f"Skipping duplicate record for source_id={source_id}, hash={content_hash[:8]}")
            return False, content_hash

        payload_str = (
            json.dumps(payload, ensure_ascii=False)
            if isinstance(payload, (dict, list))
            else str(payload)
        )

        self.con.execute(
            """
            INSERT INTO bronze_raw_feeds (source_id, source_type, payload_json, ingestion_timestamp, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            [source_id, source_type, payload_str, ingestion_timestamp, content_hash],
        )

        # Update checkpoint tracking
        fn = feed_name or source_type
        self.con.execute(
            """
            INSERT INTO bronze_checkpoints (feed_name, last_ingested_at, record_count, checkpoint_offset)
            VALUES (?, ?, 1, ?)
            ON CONFLICT (feed_name) DO UPDATE SET
                last_ingested_at = EXCLUDED.last_ingested_at,
                record_count = bronze_checkpoints.record_count + 1,
                checkpoint_offset = COALESCE(EXCLUDED.checkpoint_offset, bronze_checkpoints.checkpoint_offset)
            """,
            [fn, ingestion_timestamp, checkpoint_offset],
        )

        return True, content_hash

    def ingest_batch(
        self,
        records: List[Dict[str, Any]],
        feed_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a batch of records. Each record dict must contain:
        - source_id: str
        - source_type: str
        - payload: dict | str (or the record itself if no nested 'payload' key)
        Optional:
        - ingestion_timestamp: datetime | str
        - checkpoint_offset: str
        """
        inserted = 0
        duplicates = 0
        hashes = []

        for item in records:
            source_id = item.get("source_id")
            source_type = item.get("source_type", "unknown")
            payload = item.get("payload", item.get("payload_json", item))
            offset = item.get("checkpoint_offset")
            ts = item.get("ingestion_timestamp")

            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    ts = None

            if not source_id:
                # Derive synthetic source_id if missing
                source_id = f"{source_type}:{compute_content_hash(payload)[:16]}"

            was_new, chash = self.ingest_record(
                source_id=source_id,
                source_type=source_type,
                payload=payload,
                ingestion_timestamp=ts,
                feed_name=feed_name or source_type,
                checkpoint_offset=offset,
            )

            if was_new:
                inserted += 1
            else:
                duplicates += 1
            hashes.append(chash)

        logger.info(f"Ingested batch: {inserted} new, {duplicates} skipped duplicates (total: {len(records)})")
        return {
            "total": len(records),
            "inserted": inserted,
            "skipped_duplicates": duplicates,
            "hashes": hashes,
        }

    def ingest_file(self, file_path: Union[str, Path], source_type: Optional[str] = None) -> Dict[str, Any]:
        """Loads and ingests a JSON file (single object, list of objects) or JSONL file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records: List[Dict[str, Any]] = []
        st = source_type or path.stem.split("_")[0]

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        records.append({
                            "source_id": obj.get("source_id", f"{st}:{compute_content_hash(obj)[:16]}"),
                            "source_type": obj.get("source_type", st),
                            "payload": obj,
                        })
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    for item in content:
                        records.append({
                            "source_id": item.get("source_id", f"{st}:{compute_content_hash(item)[:16]}"),
                            "source_type": item.get("source_type", st),
                            "payload": item,
                        })
                elif isinstance(content, dict):
                    records.append({
                        "source_id": content.get("source_id", f"{st}:{compute_content_hash(content)[:16]}"),
                        "source_type": content.get("source_type", st),
                        "payload": content,
                    })

        return self.ingest_batch(records, feed_name=f"file:{path.name}")

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Returns all bronze records as a list of dictionaries."""
        df = self.con.execute(
            "SELECT source_id, source_type, payload_json, ingestion_timestamp, content_hash FROM bronze_raw_feeds ORDER BY ingestion_timestamp ASC"
        ).fetchdf()
        return df.to_dict(orient="records")

    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Returns checkpoint status for all feeds."""
        df = self.con.execute(
            "SELECT feed_name, last_ingested_at, record_count, checkpoint_offset FROM bronze_checkpoints"
        ).fetchdf()
        return df.to_dict(orient="records")

    def get_count(self) -> int:
        """Returns total records in bronze_raw_feeds."""
        return self.con.execute("SELECT COUNT(*) FROM bronze_raw_feeds").fetchone()[0]
