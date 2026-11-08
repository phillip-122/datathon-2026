"""
SIGNAL Silver Normalization & Knowledge Pipeline
Transforms raw Bronze payloads into normalized entities:
- silver_developments
- silver_models
- silver_tools
- silver_benchmarks
- silver_evaluations
- silver_capabilities
- silver_relationships
- silver_evidence
- silver_audit_log

Enforces schema validation, reference integrity, confidence taxonomy,
and audit tracking for version preservation.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb

logger = logging.getLogger("signal.pipelines.silver")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

VALID_CONFIDENCE_LEVELS = {
    "ANNOUNCEMENT",
    "EVALUATED",
    "PEER_REVIEWED",
    "HYPOTHESIS",
    "EMPIRICAL",
}

VALID_RELATIONSHIP_TYPES = {
    "implements",
    "depends_on",
    "evaluates_on",
    "improves_upon",
    "integrated_with",
    "derived_from",
}

VALID_TOOL_CATEGORIES = {
    "agent_orchestration",
    "serving_inference",
    "eval_observability",
    "prompt_engineering",
    "infrastructure",
}

VALID_DEV_CATEGORIES = {
    "model_release",
    "agent_framework",
    "benchmark",
    "research_breakthrough",
    "infra_tool",
}


def parse_iso_date(val: Any) -> Optional[date]:
    """Safely extracts date from string or datetime object."""
    if not val:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    try:
        s = str(val).split("T")[0].split(" ")[0].strip()
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def parse_iso_datetime(val: Any) -> datetime:
    """Safely extracts datetime from string or datetime object, defaulting to UTC now."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now(timezone.utc)


class SilverPipeline:
    """
    Silver normalization engine for SIGNAL.
    Operates on a DuckDB connection or creates/attaches to a database file.
    """

    def __init__(self, db_path: Optional[str] = None, connection: Optional[duckdb.DuckDBPyConnection] = None):
        if connection is not None:
            self.con = connection
        elif db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(db_path)
        else:
            self.con = duckdb.connect(":memory:")

        self.init_tables()

    def init_tables(self) -> None:
        """Initializes all Silver relational tables and integrity audit structures."""

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_developments (
                development_id VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                summary VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                date DATE NOT NULL,
                confidence_level VARCHAR NOT NULL,
                primary_entity_id VARCHAR,
                source_id VARCHAR NOT NULL,
                raw_ref VARCHAR NOT NULL,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_models (
                model_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                release_date DATE NOT NULL,
                parameter_count BIGINT,
                architecture VARCHAR,
                context_window INTEGER,
                license VARCHAR NOT NULL,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_tools (
                tool_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                repo_url VARCHAR,
                release_version VARCHAR,
                stars INTEGER,
                ecosystem_role VARCHAR,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_benchmarks (
                benchmark_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                evaluation_metric VARCHAR NOT NULL,
                dataset_size INTEGER,
                task_type VARCHAR NOT NULL,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_evaluations (
                eval_id VARCHAR PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                benchmark_id VARCHAR NOT NULL,
                score DOUBLE NOT NULL,
                eval_date DATE NOT NULL,
                methodology VARCHAR NOT NULL,
                source_url VARCHAR NOT NULL,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_capabilities (
                capability_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                parent_capability VARCHAR,
                description VARCHAR,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_relationships (
                source_entity_id VARCHAR NOT NULL,
                target_entity_id VARCHAR NOT NULL,
                relationship_type VARCHAR NOT NULL,
                strength DOUBLE NOT NULL DEFAULT 1.0,
                evidence_id VARCHAR NOT NULL,
                extraction_timestamp TIMESTAMP NOT NULL,
                PRIMARY KEY (source_entity_id, target_entity_id, relationship_type)
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_evidence (
                evidence_id VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                source_type VARCHAR NOT NULL,
                source_url VARCHAR NOT NULL,
                publication_date TIMESTAMP,
                claim_text VARCHAR NOT NULL,
                confidence VARCHAR NOT NULL,
                extraction_timestamp TIMESTAMP NOT NULL
            );
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS silver_audit_log (
                audit_id VARCHAR PRIMARY KEY,
                entity_type VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                details VARCHAR,
                timestamp TIMESTAMP NOT NULL
            );
        """)

        # Performance indexes
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_dev_date ON silver_developments(date);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_dev_cat ON silver_developments(category);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_eval_model ON silver_evaluations(model_id);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_eval_bench ON silver_evaluations(benchmark_id);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON silver_relationships(source_entity_id);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON silver_relationships(target_entity_id);")

    def _log_audit(self, entity_type: str, entity_id: str, action: str, details: Optional[str] = None) -> None:
        """Records an action in the audit log for lineage and version preservation."""
        self.con.execute(
            """
            INSERT INTO silver_audit_log (audit_id, entity_type, entity_id, action, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [str(uuid.uuid4()), entity_type, entity_id, action, details, datetime.now(timezone.utc)],
        )

    # --------------------------------------------------------------------------
    # Individual Entity Upsert Helpers
    # --------------------------------------------------------------------------

    def upsert_evidence(self, ev: Dict[str, Any], extraction_time: datetime) -> str:
        ev_id = ev.get("evidence_id") or f"evi_{uuid.uuid4().hex[:12]}"
        confidence = ev.get("confidence", "ANNOUNCEMENT")
        if confidence not in VALID_CONFIDENCE_LEVELS:
            confidence = "ANNOUNCEMENT"

        pub_date = ev.get("publication_date")
        if pub_date:
            pub_date = parse_iso_datetime(pub_date)

        self.con.execute(
            """
            INSERT INTO silver_evidence (evidence_id, source_id, source_type, source_url, publication_date, claim_text, confidence, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (evidence_id) DO UPDATE SET
                source_url = EXCLUDED.source_url,
                claim_text = EXCLUDED.claim_text,
                confidence = EXCLUDED.confidence,
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [
                ev_id,
                ev.get("source_id", "unknown_source"),
                ev.get("source_type", "announcement"),
                ev.get("source_url", ""),
                pub_date,
                ev.get("claim_text", ""),
                confidence,
                extraction_time,
            ],
        )
        return ev_id

    def upsert_capability(self, cap: Dict[str, Any], extraction_time: datetime) -> str:
        cap_id = cap.get("capability_id") or cap.get("id")
        if not cap_id:
            cap_id = f"cap_{cap.get('name', 'unknown').lower().replace(' ', '_')}"

        self.con.execute(
            """
            INSERT INTO silver_capabilities (capability_id, name, parent_capability, description, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (capability_id) DO UPDATE SET
                name = EXCLUDED.name,
                parent_capability = COALESCE(EXCLUDED.parent_capability, silver_capabilities.parent_capability),
                description = COALESCE(EXCLUDED.description, silver_capabilities.description),
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [
                cap_id,
                cap.get("name", cap_id),
                cap.get("parent_capability"),
                cap.get("description"),
                extraction_time,
            ],
        )
        return cap_id

    def upsert_benchmark(self, bm: Dict[str, Any], extraction_time: datetime) -> str:
        bm_id = bm.get("benchmark_id") or bm.get("id")
        if not bm_id:
            bm_id = bm.get("name", "unknown").lower().replace(" ", "_").replace("-", "_")

        self.con.execute(
            """
            INSERT INTO silver_benchmarks (benchmark_id, name, evaluation_metric, dataset_size, task_type, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (benchmark_id) DO UPDATE SET
                name = EXCLUDED.name,
                evaluation_metric = EXCLUDED.evaluation_metric,
                dataset_size = COALESCE(EXCLUDED.dataset_size, silver_benchmarks.dataset_size),
                task_type = EXCLUDED.task_type,
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [
                bm_id,
                bm.get("name", bm_id),
                bm.get("evaluation_metric", "Accuracy (%)"),
                bm.get("dataset_size"),
                bm.get("task_type", "Reasoning & Knowledge"),
                extraction_time,
            ],
        )
        return bm_id

    def upsert_model(self, model: Dict[str, Any], extraction_time: datetime) -> str:
        m_id = model.get("model_id") or model.get("id")
        if not m_id:
            m_id = model.get("name", "unknown_model").lower().replace(" ", "-")

        rel_date = parse_iso_date(model.get("release_date")) or date.today()

        # Check existing to detect version updates
        existing = self.con.execute("SELECT name, context_window FROM silver_models WHERE model_id = ?", [m_id]).fetchone()
        if existing:
            self._log_audit("model", m_id, "UPDATE", f"Updated model specs: {model.get('name')}")
        else:
            self._log_audit("model", m_id, "INSERT", f"Added new model: {model.get('name')}")

        self.con.execute(
            """
            INSERT INTO silver_models (model_id, name, provider, release_date, parameter_count, architecture, context_window, license, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_id) DO UPDATE SET
                name = EXCLUDED.name,
                provider = EXCLUDED.provider,
                release_date = EXCLUDED.release_date,
                parameter_count = COALESCE(EXCLUDED.parameter_count, silver_models.parameter_count),
                architecture = COALESCE(EXCLUDED.architecture, silver_models.architecture),
                context_window = COALESCE(EXCLUDED.context_window, silver_models.context_window),
                license = EXCLUDED.license,
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [
                m_id,
                model.get("name", m_id),
                model.get("provider", "Unknown"),
                rel_date,
                model.get("parameter_count"),
                model.get("architecture"),
                model.get("context_window"),
                model.get("license", "Proprietary"),
                extraction_time,
            ],
        )
        return m_id

    def upsert_tool(self, tool: Dict[str, Any], extraction_time: datetime) -> str:
        t_id = tool.get("tool_id") or tool.get("id")
        if not t_id:
            t_id = tool.get("name", "unknown_tool").lower().replace(" ", "-")

        cat = tool.get("category", "agent_orchestration")
        if cat not in VALID_TOOL_CATEGORIES:
            cat = "agent_orchestration"

        existing = self.con.execute("SELECT release_version, stars FROM silver_tools WHERE tool_id = ?", [t_id]).fetchone()
        if existing:
            self._log_audit("tool", t_id, "UPDATE", f"Updated version/metrics: {tool.get('release_version')}")
        else:
            self._log_audit("tool", t_id, "INSERT", f"Added new tool: {tool.get('name')}")

        self.con.execute(
            """
            INSERT INTO silver_tools (tool_id, name, category, repo_url, release_version, stars, ecosystem_role, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (tool_id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                repo_url = COALESCE(EXCLUDED.repo_url, silver_tools.repo_url),
                release_version = COALESCE(EXCLUDED.release_version, silver_tools.release_version),
                stars = COALESCE(EXCLUDED.stars, silver_tools.stars),
                ecosystem_role = COALESCE(EXCLUDED.ecosystem_role, silver_tools.ecosystem_role),
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [
                t_id,
                tool.get("name", t_id),
                cat,
                tool.get("repo_url"),
                tool.get("release_version"),
                tool.get("stars"),
                tool.get("ecosystem_role"),
                extraction_time,
            ],
        )
        return t_id

    def upsert_evaluation(self, ev: Dict[str, Any], extraction_time: datetime) -> str:
        eval_id = ev.get("eval_id") or f"eval_{uuid.uuid4().hex[:12]}"
        model_id = ev.get("model_id")
        benchmark_id = ev.get("benchmark_id")
        score = float(ev.get("score", 0.0))
        eval_date = parse_iso_date(ev.get("eval_date")) or date.today()
        methodology = ev.get("methodology", "Standard Evaluation")
        source_url = ev.get("source_url", "")

        # Verify foreign keys
        m_check = self.con.execute("SELECT 1 FROM silver_models WHERE model_id = ?", [model_id]).fetchone()
        if not m_check:
            self._log_audit("evaluation", eval_id, "ORPHAN_REFERENCE", f"Model {model_id} not registered yet")

        b_check = self.con.execute("SELECT 1 FROM silver_benchmarks WHERE benchmark_id = ?", [benchmark_id]).fetchone()
        if not b_check:
            self._log_audit("evaluation", eval_id, "ORPHAN_REFERENCE", f"Benchmark {benchmark_id} not registered yet")

        self.con.execute(
            """
            INSERT INTO silver_evaluations (eval_id, model_id, benchmark_id, score, eval_date, methodology, source_url, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (eval_id) DO UPDATE SET
                score = EXCLUDED.score,
                eval_date = EXCLUDED.eval_date,
                methodology = EXCLUDED.methodology,
                source_url = EXCLUDED.source_url,
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [eval_id, model_id, benchmark_id, score, eval_date, methodology, source_url, extraction_time],
        )
        return eval_id

    def upsert_relationship(self, rel: Dict[str, Any], extraction_time: datetime) -> None:
        source_id = rel.get("source_entity_id")
        target_id = rel.get("target_entity_id")
        rel_type = rel.get("relationship_type", "implements")
        if rel_type not in VALID_RELATIONSHIP_TYPES:
            rel_type = "implements"
        strength = float(rel.get("strength", 1.0))
        evidence_id = rel.get("evidence_id", "evi_default")

        self.con.execute(
            """
            INSERT INTO silver_relationships (source_entity_id, target_entity_id, relationship_type, strength, evidence_id, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_entity_id, target_entity_id, relationship_type) DO UPDATE SET
                strength = EXCLUDED.strength,
                evidence_id = EXCLUDED.evidence_id,
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [source_id, target_id, rel_type, strength, evidence_id, extraction_time],
        )

    def upsert_development(self, dev: Dict[str, Any], raw_ref: str, extraction_time: datetime) -> str:
        dev_id = dev.get("development_id") or f"dev_{uuid.uuid4().hex[:12]}"
        title = dev.get("title", "Untitled Development")
        summary = dev.get("summary", "")
        category = dev.get("category", "model_release")
        if category not in VALID_DEV_CATEGORIES:
            category = "model_release"
        dev_date = parse_iso_date(dev.get("date")) or date.today()
        conf = dev.get("confidence_level", "ANNOUNCEMENT")
        if conf not in VALID_CONFIDENCE_LEVELS:
            conf = "ANNOUNCEMENT"
        primary_entity = dev.get("primary_entity_id")
        source_id = dev.get("source_id", "unknown_source")

        self.con.execute(
            """
            INSERT INTO silver_developments (development_id, title, summary, category, date, confidence_level, primary_entity_id, source_id, raw_ref, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (development_id) DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                category = EXCLUDED.category,
                date = EXCLUDED.date,
                confidence_level = EXCLUDED.confidence_level,
                primary_entity_id = EXCLUDED.primary_entity_id,
                source_id = EXCLUDED.source_id,
                raw_ref = EXCLUDED.raw_ref,
                extraction_timestamp = EXCLUDED.extraction_timestamp
            """,
            [dev_id, title, summary, category, dev_date, conf, primary_entity, source_id, raw_ref, extraction_time],
        )
        return dev_id

    # --------------------------------------------------------------------------
    # Main Bronze-to-Silver Transformation
    # --------------------------------------------------------------------------

    def process_bronze_records(self, bronze_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Parses Bronze records and normalizes them into Silver tables.
        If bronze_records is None, reads all records from bronze_raw_feeds in DuckDB.
        """
        if bronze_records is None:
            df = self.con.execute(
                "SELECT source_id, source_type, payload_json, ingestion_timestamp, content_hash FROM bronze_raw_feeds"
            ).fetchdf()
            bronze_records = df.to_dict(orient="records")

        stats = {
            "processed_bronze": 0,
            "developments": 0,
            "models": 0,
            "tools": 0,
            "benchmarks": 0,
            "evaluations": 0,
            "capabilities": 0,
            "relationships": 0,
            "evidence": 0,
            "audit_warnings": 0,
        }

        now_utc = datetime.now(timezone.utc)

        for record in bronze_records:
            stats["processed_bronze"] += 1
            source_id = record.get("source_id", "unknown")
            source_type = record.get("source_type", "unknown")
            content_hash = record.get("content_hash", "")
            raw_payload = record.get("payload_json")

            if isinstance(raw_payload, str):
                try:
                    payload = json.loads(raw_payload)
                except Exception as err:
                    logger.warning(f"Could not parse payload_json for source_id={source_id}: {err}")
                    self._log_audit("payload", source_id, "VALIDATION_WARNING", f"JSON parse error: {err}")
                    stats["audit_warnings"] += 1
                    continue
            elif isinstance(raw_payload, dict):
                payload = raw_payload
            else:
                payload = {}

            # 1. Evidence extraction
            evidence_id = None
            if "evidence" in payload and isinstance(payload["evidence"], dict):
                ev_data = payload["evidence"]
                ev_data.setdefault("source_id", source_id)
                ev_data.setdefault("source_type", source_type)
                evidence_id = self.upsert_evidence(ev_data, now_utc)
                stats["evidence"] += 1
            elif "claim_text" in payload or "source_url" in payload:
                # Standalone evidence in payload root
                ev_data = {
                    "evidence_id": f"evi_{source_id.replace(':', '_')}",
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_url": payload.get("source_url", ""),
                    "publication_date": payload.get("publication_date") or payload.get("date"),
                    "claim_text": payload.get("claim_text") or payload.get("summary") or payload.get("title", ""),
                    "confidence": payload.get("confidence") or payload.get("confidence_level", "ANNOUNCEMENT"),
                }
                evidence_id = self.upsert_evidence(ev_data, now_utc)
                stats["evidence"] += 1

            # 2. Capabilities
            if "capabilities" in payload and isinstance(payload["capabilities"], list):
                for cap in payload["capabilities"]:
                    self.upsert_capability(cap, now_utc)
                    stats["capabilities"] += 1
            elif "capability" in payload and isinstance(payload["capability"], dict):
                self.upsert_capability(payload["capability"], now_utc)
                stats["capabilities"] += 1

            # 3. Benchmarks
            if "benchmarks" in payload and isinstance(payload["benchmarks"], list):
                for bm in payload["benchmarks"]:
                    self.upsert_benchmark(bm, now_utc)
                    stats["benchmarks"] += 1
            elif "benchmark" in payload and isinstance(payload["benchmark"], dict):
                self.upsert_benchmark(payload["benchmark"], now_utc)
                stats["benchmarks"] += 1

            # 4. Models
            primary_entity_id = None
            if "models" in payload and isinstance(payload["models"], list):
                for m in payload["models"]:
                    mid = self.upsert_model(m, now_utc)
                    stats["models"] += 1
                    if not primary_entity_id:
                        primary_entity_id = mid
            elif "model" in payload and isinstance(payload["model"], dict):
                primary_entity_id = self.upsert_model(payload["model"], now_utc)
                stats["models"] += 1

            # 5. Tools
            if "tools" in payload and isinstance(payload["tools"], list):
                for t in payload["tools"]:
                    tid = self.upsert_tool(t, now_utc)
                    stats["tools"] += 1
                    if not primary_entity_id:
                        primary_entity_id = tid
            elif "tool" in payload and isinstance(payload["tool"], dict):
                primary_entity_id = self.upsert_tool(payload["tool"], now_utc)
                stats["tools"] += 1

            # 6. Evaluations
            if "evaluations" in payload and isinstance(payload["evaluations"], list):
                for ev in payload["evaluations"]:
                    self.upsert_evaluation(ev, now_utc)
                    stats["evaluations"] += 1

            # 7. Relationships
            if "relationships" in payload and isinstance(payload["relationships"], list):
                for rel in payload["relationships"]:
                    if evidence_id and not rel.get("evidence_id"):
                        rel["evidence_id"] = evidence_id
                    self.upsert_relationship(rel, now_utc)
                    stats["relationships"] += 1

            # 8. Developments
            if "developments" in payload and isinstance(payload["developments"], list):
                for d in payload["developments"]:
                    d.setdefault("source_id", source_id)
                    if primary_entity_id and not d.get("primary_entity_id"):
                        d["primary_entity_id"] = primary_entity_id
                    self.upsert_development(d, content_hash, now_utc)
                    stats["developments"] += 1
            elif "development" in payload and isinstance(payload["development"], dict):
                d = payload["development"]
                d.setdefault("source_id", source_id)
                if primary_entity_id and not d.get("primary_entity_id"):
                    d["primary_entity_id"] = primary_entity_id
                self.upsert_development(d, content_hash, now_utc)
                stats["developments"] += 1
            elif "title" in payload and "summary" in payload:
                # Payload is itself a development
                d = {
                    "development_id": payload.get("development_id", f"dev_{source_id.replace(':', '_')}"),
                    "title": payload.get("title"),
                    "summary": payload.get("summary"),
                    "category": payload.get("category", "model_release"),
                    "date": payload.get("date", payload.get("publication_date")),
                    "confidence_level": payload.get("confidence_level", payload.get("confidence", "ANNOUNCEMENT")),
                    "primary_entity_id": primary_entity_id or payload.get("primary_entity_id"),
                    "source_id": source_id,
                }
                self.upsert_development(d, content_hash, now_utc)
                stats["developments"] += 1

        logger.info(f"Silver normalization completed: {stats}")
        return stats
