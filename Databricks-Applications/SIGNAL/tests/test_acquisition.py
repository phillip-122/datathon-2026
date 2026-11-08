"""
SIGNAL Data Acquisition & Ingestion Tests
Validates collectors, raw bronze envelopes, and offline fixtures relational integrity.
"""

import json
from pathlib import Path
import pytest

from SIGNAL.ingestion.sources.arxiv_collector import ArxivCollector
from SIGNAL.ingestion.sources.huggingface_collector import HuggingFaceCollector
from SIGNAL.ingestion.sources.github_collector import GitHubCollector
from SIGNAL.ingestion.sources.collect_all import run_all_collectors


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def fixtures_data(repo_root):
    fixture_path = repo_root / "SIGNAL" / "data" / "fixtures" / "signal_fixtures.json"
    assert fixture_path.exists(), f"Fixtures file missing at {fixture_path}"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_arxiv_collector_mock(tmp_path):
    collector = ArxivCollector(output_dir=tmp_path, mode="mock")
    result = collector.collect(max_results=5, save=True)

    assert result["metadata"]["source_type"] == "arxiv"
    assert result["metadata"]["mode"] == "mock"
    assert result["metadata"]["record_count"] == 5
    assert len(result["records"]) == 5

    record = result["records"][0]
    assert "arxiv_id" in record
    assert "title" in record
    assert "authors" in record
    assert "canonical_url" in record
    assert "evidence" in record
    assert record["evidence"]["source_type"] == "arxiv"

    # Verify persisted raw file
    saved_files = list(tmp_path.glob("arxiv_papers_mock_*.json"))
    assert len(saved_files) == 1


def test_huggingface_collector_mock(tmp_path):
    collector = HuggingFaceCollector(output_dir=tmp_path, mode="mock")
    result = collector.collect(limit=5, save=True)

    assert result["metadata"]["source_type"] == "huggingface"
    assert result["metadata"]["mode"] == "mock"
    assert result["metadata"]["record_count"] == 5

    record = result["records"][0]
    assert "repo_id" in record
    assert "model_name" in record
    assert "downloads" in record
    assert "likes" in record
    assert "evidence" in record
    assert record["evidence"]["source_type"] == "huggingface"


def test_github_collector_mock(tmp_path):
    collector = GitHubCollector(output_dir=tmp_path, mode="mock")
    result = collector.collect(limit=4, save=True)

    assert result["metadata"]["source_type"] == "github"
    assert result["metadata"]["mode"] == "mock"
    assert result["metadata"]["record_count"] == 4

    record = result["records"][0]
    assert "repo_slug" in record
    assert "stars_count" in record
    assert "license" in record
    assert "evidence" in record
    assert record["evidence"]["source_type"] == "github"


def test_run_all_collectors(tmp_path):
    manifest = run_all_collectors(mode="mock", base_raw_dir=tmp_path, save=True)

    assert manifest["mode"] == "mock"
    assert manifest["summary"]["successful_sources"] == 3
    assert manifest["summary"]["failed_sources"] == 0
    assert manifest["summary"]["total_records"] >= 20

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.exists()


def test_fixtures_schema_and_relational_integrity(fixtures_data):
    """
    Validates complete referential integrity across models, benchmarks,
    evaluations, tools, papers, and relationships.
    """
    meta = fixtures_data["metadata"]
    assert meta["fixture_version"] == "1.0.0"

    models = {m["model_id"]: m for m in fixtures_data["models"]}
    benchmarks = {b["benchmark_id"]: b for b in fixtures_data["benchmarks"]}
    tools = {t["tool_id"]: t for t in fixtures_data["tools"]}
    papers = {p["paper_id"]: p for p in fixtures_data["research_papers"]}
    all_entity_ids = set(models.keys()) | set(benchmarks.keys()) | set(tools.keys()) | set(papers.keys())

    assert len(models) >= 15, "Expected at least 15 models"
    assert len(benchmarks) >= 8, "Expected at least 8 benchmarks"
    assert len(fixtures_data["evaluations"]) >= 50, "Expected at least 50 benchmark evaluations"
    assert len(tools) >= 8, "Expected at least 8 tools"
    assert len(papers) >= 8, "Expected at least 8 research papers"
    assert len(fixtures_data["relationships"]) >= 15, "Expected at least 15 graph relationships"
    assert len(fixtures_data["recent_developments"]) >= 8, "Expected at least 8 recent development items"

    # Validate evaluations foreign keys and bounds
    for ev in fixtures_data["evaluations"]:
        assert ev["model_id"] in models, f"Model {ev['model_id']} not found in models table"
        assert ev["benchmark_id"] in benchmarks, f"Benchmark {ev['benchmark_id']} not found in benchmarks table"
        assert 0.0 <= ev["score"] <= 100.0, f"Score out of percentage bounds: {ev['score']}"
        assert ev["metric_unit"] == "%"

    # Validate relationships graph integrity
    for rel in fixtures_data["relationships"]:
        assert rel["source_entity_id"] in all_entity_ids, f"Source entity {rel['source_entity_id']} not found"
        assert rel["target_entity_id"] in all_entity_ids, f"Target entity {rel['target_entity_id']} not found"
        assert 0.0 <= rel["strength"] <= 1.0

    # Validate EvidenceRecord in every entity
    sections = ["models", "benchmarks", "evaluations", "tools", "research_papers", "relationships", "recent_developments"]
    for section in sections:
        for item in fixtures_data[section]:
            ev = item.get("evidence")
            assert ev is not None, f"Item in {section} missing evidence: {item}"
            assert ev.get("source_id"), "Missing source_id"
            assert ev.get("source_type"), "Missing source_type"
            assert ev.get("source_url"), "Missing source_url"
            assert ev.get("ingestion_timestamp"), "Missing ingestion_timestamp"
            assert ev.get("evidence_confidence") in [
                "PEER_REVIEWED",
                "EVALUATED",
                "ANNOUNCEMENT",
                "EMPIRICAL",
                "HYPOTHESIS",
            ]
