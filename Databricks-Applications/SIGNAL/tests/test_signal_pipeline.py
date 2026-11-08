"""
SIGNAL Data Engineering Pipeline Automated Test Suite
Verifies:
1. Bronze deduplication and canonical content hashing
2. Incremental checkpointing
3. Silver normalization and schema adherence
4. Entity version preservation and audit logging
5. Relationship and referential integrity
6. Methodology matching on benchmark evaluations
7. Gold analytical views calculation (30-day window, momentum score, graph traversal)
8. End-to-end pipeline execution and Parquet export
"""

import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from SIGNAL.pipelines.bronze_pipeline import BronzePipeline, canonical_json_bytes, compute_content_hash
from SIGNAL.pipelines.silver_pipeline import SilverPipeline
from SIGNAL.pipelines.gold_pipeline import GoldPipeline
from SIGNAL.pipelines.runner import SignalPipelineRunner
from SIGNAL.pipelines.seed_data import get_curated_seed_records


@pytest.fixture
def memory_db():
    """Provides a fresh, clean in-memory DuckDB connection for test isolation."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()


# ==============================================================================
# 1. BRONZE LAYER TESTS
# ==============================================================================

class TestBronzePipeline:
    def test_canonical_json_hashing_determinism(self):
        """Ensures key ordering and formatting differences yield identical hashes."""
        payload_a = {"title": "Test Paper", "authors": ["Alice", "Bob"], "version": 1}
        payload_b = {"version": 1, "authors": ["Alice", "Bob"], "title": "Test Paper"}
        payload_json_str = '{"authors": ["Alice", "Bob"], "title": "Test Paper", "version": 1}'

        hash_a = compute_content_hash(payload_a)
        hash_b = compute_content_hash(payload_b)
        hash_str = compute_content_hash(payload_json_str)

        assert hash_a == hash_b
        assert hash_a == hash_str
        assert len(hash_a) == 64  # SHA-256 length

    def test_bronze_ingestion_and_deduplication(self, memory_db):
        """Verifies that duplicate records are detected and skipped without failure."""
        bronze = BronzePipeline(connection=memory_db)

        record = {
            "title": "DeepSeek-R1 Inception",
            "model": "deepseek-r1",
            "score": 97.3,
        }

        # First ingestion -> inserted
        inserted_1, hash_1 = bronze.ingest_record(
            source_id="arxiv:2501.12948",
            source_type="arxiv",
            payload=record,
        )
        assert inserted_1 is True
        assert bronze.get_count() == 1

        # Second ingestion with exact same payload -> skipped
        inserted_2, hash_2 = bronze.ingest_record(
            source_id="arxiv:2501.12948",
            source_type="arxiv",
            payload=record,
        )
        assert inserted_2 is False
        assert hash_2 == hash_1
        assert bronze.get_count() == 1  # Record count remained 1

    def test_version_change_creates_new_bronze_entry(self, memory_db):
        """When payload content changes, a new record is created with distinct hash."""
        bronze = BronzePipeline(connection=memory_db)

        v1 = {"tool": "langgraph", "version": "v0.2.60"}
        v2 = {"tool": "langgraph", "version": "v0.2.68"}

        ok_1, h1 = bronze.ingest_record("gh:langgraph", "github", v1)
        ok_2, h2 = bronze.ingest_record("gh:langgraph", "github", v2)

        assert ok_1 is True
        assert ok_2 is True
        assert h1 != h2
        assert bronze.get_count() == 2

    def test_checkpointing_tracking(self, memory_db):
        """Verifies bronze_checkpoints stores feed watermarks accurately."""
        bronze = BronzePipeline(connection=memory_db)

        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        bronze.ingest_record("hf:model-1", "huggingface", {"m": 1}, ingestion_timestamp=ts, checkpoint_offset="cursor_1")
        bronze.ingest_record("hf:model-2", "huggingface", {"m": 2}, ingestion_timestamp=ts, checkpoint_offset="cursor_2")

        checkpoints = bronze.get_checkpoints()
        assert len(checkpoints) == 1
        cp = checkpoints[0]
        assert cp["feed_name"] == "huggingface"
        assert cp["record_count"] == 2
        assert cp["checkpoint_offset"] == "cursor_2"


# ==============================================================================
# 2. SILVER LAYER TESTS
# ==============================================================================

class TestSilverPipeline:
    def test_silver_entity_extraction_and_provenance(self, memory_db):
        """Verifies normalized entities and evidence provenance are correctly extracted."""
        bronze = BronzePipeline(connection=memory_db)
        silver = SilverPipeline(connection=memory_db)

        bundle = {
            "model": {
                "model_id": "test-model-42",
                "name": "Test Model 42",
                "provider": "Acme AI",
                "release_date": "2026-01-10",
                "context_window": 128000,
                "license": "Apache 2.0",
            },
            "benchmarks": [
                {
                    "benchmark_id": "test_bench",
                    "name": "Test Benchmark",
                    "evaluation_metric": "Accuracy (%)",
                    "dataset_size": 100,
                    "task_type": "Code Synthesis",
                }
            ],
            "evaluations": [
                {
                    "eval_id": "eval_test_42",
                    "model_id": "test-model-42",
                    "benchmark_id": "test_bench",
                    "score": 91.5,
                    "eval_date": "2026-01-11",
                    "methodology": "0-shot CoT",
                    "source_url": "https://example.com/eval",
                }
            ],
            "evidence": {
                "evidence_id": "evi_test_42",
                "source_id": "test:eval_42",
                "source_type": "independent_eval",
                "source_url": "https://example.com/eval",
                "claim_text": "Test Model 42 achieves 91.5% accuracy.",
                "confidence": "EVALUATED",
            },
            "relationships": [
                {
                    "source_entity_id": "test-model-42",
                    "target_entity_id": "test_bench",
                    "relationship_type": "evaluates_on",
                    "strength": 1.0,
                    "evidence_id": "evi_test_42",
                }
            ],
            "development": {
                "development_id": "dev_test_42",
                "title": "Acme Releases Test Model 42",
                "summary": "State-of-the-art 91.5% code benchmark score.",
                "category": "model_release",
                "date": "2026-01-10",
                "confidence_level": "EVALUATED",
            },
        }

        bronze.ingest_record("test:eval_42", "independent_eval", bundle)
        stats = silver.process_bronze_records()

        assert stats["models"] == 1
        assert stats["benchmarks"] == 1
        assert stats["evaluations"] == 1
        assert stats["evidence"] == 1
        assert stats["relationships"] == 1
        assert stats["developments"] == 1

        # Verify silver_models contents
        m = memory_db.execute("SELECT name, provider, context_window FROM silver_models WHERE model_id = 'test-model-42'").fetchone()
        assert m == ("Test Model 42", "Acme AI", 128000)

        # Verify evidence provenance
        e = memory_db.execute("SELECT confidence, claim_text FROM silver_evidence WHERE evidence_id = 'evi_test_42'").fetchone()
        assert e[0] == "EVALUATED"

    def test_version_preservation_and_audit_logging(self, memory_db):
        """Verifies that entity updates maintain audit log history."""
        silver = SilverPipeline(connection=memory_db)
        now = datetime.now(timezone.utc)

        # Insert tool v1
        silver.upsert_tool({
            "tool_id": "vllm",
            "name": "vLLM",
            "category": "serving_inference",
            "release_version": "v0.7.0",
            "stars": 30000,
        }, now)

        # Update tool to v2
        silver.upsert_tool({
            "tool_id": "vllm",
            "name": "vLLM",
            "category": "serving_inference",
            "release_version": "v0.7.2",
            "stars": 34500,
        }, now)

        # Check tool record
        tool = memory_db.execute("SELECT release_version, stars FROM silver_tools WHERE tool_id = 'vllm'").fetchone()
        assert tool == ("v0.7.2", 34500)

        # Check audit log contains both INSERT and UPDATE
        logs = memory_db.execute("SELECT action, details FROM silver_audit_log WHERE entity_id = 'vllm' ORDER BY timestamp ASC").fetchall()
        assert len(logs) == 2
        assert logs[0][0] == "INSERT"
        assert logs[1][0] == "UPDATE"

    def test_referential_integrity_warning(self, memory_db):
        """Orphaned evaluations (missing model or benchmark) generate audit log warnings."""
        silver = SilverPipeline(connection=memory_db)
        now = datetime.now(timezone.utc)

        # Upsert evaluation without registering model first
        silver.upsert_evaluation({
            "eval_id": "eval_orphan",
            "model_id": "nonexistent_model",
            "benchmark_id": "nonexistent_bench",
            "score": 50.0,
            "methodology": "0-shot",
            "source_url": "https://example.com",
        }, now)

        orphans = memory_db.execute("SELECT action, details FROM silver_audit_log WHERE action = 'ORPHAN_REFERENCE'").fetchall()
        assert len(orphans) >= 1
        assert "nonexistent_model" in orphans[0][1]


# ==============================================================================
# 3. GOLD LAYER & ANALYTICAL VIEWS TESTS
# ==============================================================================

class TestGoldViews:
    @pytest.fixture(autouse=True)
    def setup_seed_pipeline(self, memory_db):
        """Initializes full seed data in the shared memory DB."""
        runner = SignalPipelineRunner(connection=memory_db)
        seed_records = get_curated_seed_records()
        runner.run_full_pipeline(raw_records=seed_records)
        self.runner = runner
        self.gold = runner.gold

    def test_v_recent_developments_window_and_provenance(self, memory_db):
        """Ensures v_recent_developments filters for last 30 days and links evidence."""
        res = self.gold.get_recent_developments(limit=20)
        assert len(res) > 0

        for dev in res:
            assert "development_id" in dev
            assert "title" in dev
            assert "confidence_level" in dev
            assert "primary_entity_name" in dev
            assert "source_url" in dev
            # Ensure date is within 30 days
            dev_date = datetime.strptime(dev["date"], "%Y-%m-%d").date()
            assert dev_date >= (date.today() - timedelta(days=30))

    def test_v_model_comparisons_methodology_matching(self, memory_db):
        """Verifies benchmark scores are correctly partitioned by methodology."""
        # Query MMLU-Pro comparisons
        evals_mmlu = self.gold.get_model_comparisons(benchmark_id="mmlu_pro")
        assert len(evals_mmlu) >= 4

        # All MMLU-Pro in seed data have 0-shot CoT
        for ev in evals_mmlu:
            assert ev["methodology"] == "0-shot CoT"
            assert "score" in ev
            assert "delta_vs_methodology_avg" in ev
            assert "rank_in_benchmark_methodology" in ev

        # Leader should have rank 1
        assert evals_mmlu[0]["rank_in_benchmark_methodology"] == 1
        assert evals_mmlu[0]["model_id"] == "deepseek-r1"
        assert evals_mmlu[0]["score"] == 84.0

    def test_v_ecosystem_relationships_graph(self, memory_db):
        """Ensures graph view connects models, tools, and capabilities with evidence."""
        edges = self.gold.get_ecosystem_graph()
        assert len(edges) >= 10

        # Test filtering by specific entity
        deepseek_edges = self.gold.get_ecosystem_graph(entity_id="deepseek-r1")
        assert len(deepseek_edges) >= 3

        types = {e["relationship_type"] for e in deepseek_edges}
        assert "implements" in types or "derived_from" in types or "integrated_with" in types

    def test_v_agent_trends_momentum_and_velocity(self, memory_db):
        """Verifies agent framework trends calculate momentum score and velocity categories."""
        trends = self.gold.get_agent_trends()
        assert len(trends) >= 3

        # Check fields and order
        for t in trends:
            assert "tool_id" in t
            assert "momentum_score" in t
            assert "velocity_category" in t
            assert t["velocity_category"] in {"ACCELERATING", "STEADY", "MATURE", "EMERGING"}

        # Verify sorted in descending momentum score order
        scores = [t["momentum_score"] for t in trends]
        assert scores == sorted(scores, reverse=True)


# ==============================================================================
# 4. END-TO-END PIPELINE & EXPORT TESTS
# ==============================================================================

class TestEndToEndPipeline:
    def test_runner_and_parquet_export(self):
        """Runs the entire pipeline against temporary directory and verifies Parquet exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_signal.duckdb"
            parquet_dir = Path(tmpdir) / "gold_parquet"

            runner = SignalPipelineRunner(db_path=db_path)
            seed_records = get_curated_seed_records()

            result = runner.run_full_pipeline(
                raw_records=seed_records,
                export_parquet_dir=parquet_dir,
            )

            assert result["status"] == "SUCCESS"
            assert result["bronze"]["inserted"] > 0
            assert result["silver"]["models"] > 0
            assert result["metrics"]["total_models"] >= 5

            # Verify Parquet files were created
            expected_views = [
                "v_recent_developments",
                "v_model_comparisons",
                "v_ecosystem_relationships",
                "v_agent_trends",
            ]
            for v in expected_views:
                p_file = parquet_dir / f"{v}.parquet"
                assert p_file.exists()
                assert p_file.stat().st_size > 0

    def test_databricks_ddl_generation(self):
        """Verifies Databricks Unity Catalog DDL contains all required schema references."""
        ddl = GoldPipeline.generate_unity_catalog_ddl(catalog="workspace", schema_prefix="signal")
        assert "workspace.signal_bronze.bronze_raw_feeds" in ddl
        assert "workspace.signal_silver.silver_developments" in ddl
        assert "workspace.signal_silver.silver_models" in ddl
        assert "workspace.signal_silver.silver_tools" in ddl
        assert "workspace.signal_silver.silver_benchmarks" in ddl
        assert "workspace.signal_silver.silver_evaluations" in ddl
        assert "workspace.signal_silver.silver_capabilities" in ddl
        assert "workspace.signal_silver.silver_relationships" in ddl
        assert "workspace.signal_silver.silver_evidence" in ddl
        assert "workspace.signal_gold.v_recent_developments" in ddl
        assert "workspace.signal_gold.v_model_comparisons" in ddl
        assert "workspace.signal_gold.v_ecosystem_relationships" in ddl
        assert "workspace.signal_gold.v_agent_trends" in ddl


# ==============================================================================
# 5. DYNAMIC SIMULATION & ADVISOR API TESTS
# ==============================================================================

class TestSignalDynamicAPIs:
    def test_api_ingest_simulate(self):
        """Verify POST /api/ingest/simulate calculates Pareto shift and runtime impacts."""
        from fastapi.testclient import TestClient
        from SIGNAL.app.main import app
        client = TestClient(app)

        res = client.post("/api/ingest/simulate", json={
            "model_name": "Test-Reasoner-Preview",
            "provider": "Test Labs",
            "architecture": "MoE 500B / MLA",
            "parameters_b": 500.0,
            "context_k": 500,
            "pricing_output_per_m": 0.20,
            "swe_bench_verified": 62.0,
            "math_500": 98.0,
            "gpqa_diamond": 72.0
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["displaced_incumbent_count"] >= 0
        assert data["benchmark_frontier_shift"]["swe_bench_delta_vs_gpt4o"] > 0
        assert len(data["ecosystem_ripple"]["runtime_impacts"]) >= 3
        assert data["databricks_telemetry"]["engine"]

    def test_api_advisor_match(self):
        """Verify POST /api/advisor/match recommends dual-tier deployment and ROI."""
        from fastapi.testclient import TestClient
        from SIGNAL.app.main import app
        client = TestClient(app)

        res = client.post("/api/advisor/match", json={
            "monthly_budget_usd": 600.0,
            "monthly_tokens_m": 60.0,
            "min_swe_bench": 40.0,
            "min_math_500": 85.0
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "primary_reasoning_tier" in data["recommended_architecture"]
        assert "financial_roi" in data
        assert data["financial_roi"]["baseline_gpt4o_spend"]

    def test_api_head_to_head(self):
        """Verify GET /api/models/head-to-head compares two models across dimensions."""
        from fastapi.testclient import TestClient
        from SIGNAL.app.main import app
        client = TestClient(app)

        res = client.get("/api/models/head-to-head?model_a=deepseek-r1-distill-qwen-14b&model_b=openai-o1")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["comparison_metrics"]) >= 5
        assert data["cost_ratio"] >= 0
