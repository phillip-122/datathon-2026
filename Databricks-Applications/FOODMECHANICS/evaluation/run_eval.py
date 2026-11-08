#!/usr/bin/env python3
"""
FOODMECHANICS Genie & Food Science SQL Evaluation Runner
=========================================================
Validates and benchmarks SQL queries for Databricks Genie Agent: FOODMECHANICS Intelligence.
Supports:
  1. Offline validation via in-memory DuckDB (simulating Unity Catalog 3-part naming).
  2. Live validation on Databricks SQL Warehouse (if configured).
  3. Ground-truth assertions on food chemistry principles and 100g normalization.

Usage:
  python FOODMECHANICS/evaluation/run_eval.py [--all] [--example-queries] [--benchmark] [--verbose]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Base paths
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
FIXTURES_FILE = PROJECT_DIR / "data" / "fixtures" / "foodmechanics_fixtures.json"
EXAMPLE_QUERIES_FILE = PROJECT_DIR / "genie" / "EXAMPLE_QUERIES.sql"
QUESTIONS_FILE = EVAL_DIR / "QUESTIONS.json"


def setup_duckdb_environment(fixtures_path: Path):
    """
    Initializes an in-memory DuckDB database matching Unity Catalog schemas:
      - workspace.foodmechanics_silver (base tables)
      - workspace.foodmechanics_gold (analytical views)
    """
    try:
        import duckdb
        import pandas as pd
    except ImportError:
        print("ERROR: duckdb and pandas are required. Install with: pip install duckdb pandas")
        sys.exit(1)

    conn = duckdb.connect()
    # Attach workspace catalog
    conn.execute('ATTACH ":memory:" AS workspace')
    conn.execute('CREATE SCHEMA workspace.foodmechanics_silver')
    conn.execute('CREATE SCHEMA workspace.foodmechanics_gold')

    if not fixtures_path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures_data = json.load(f)

    # Populate Silver Tables
    for table_name, rows in fixtures_data.items():
        if rows:
            df = pd.DataFrame(rows)
            conn.register(f"temp_{table_name}", df)
            conn.execute(f"CREATE TABLE workspace.foodmechanics_silver.{table_name} AS SELECT * FROM temp_{table_name}")

    # Create Gold Analytical Views
    conn.execute("""
    CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_ingredient_comparisons AS
    SELECT 
        i.ingredient_id,
        i.common_name AS ingredient_name,
        i.category,
        i.ph_typical,
        np.calories_kcal AS calories_per_100g,
        np.protein_g AS protein_g_per_100g,
        np.fat_total_g AS fat_total_g_per_100g,
        np.carbohydrates_g AS carbs_g_per_100g,
        np.water_g AS water_g_per_100g,
        MAX(CASE WHEN fp.property_category = 'Foam Formation' THEN fp.quantitative_value END) AS foaming_overrun_pct,
        MAX(CASE WHEN fp.property_category = 'Foam Stability' THEN fp.quantitative_value END) AS drainage_half_life_min,
        MAX(CASE WHEN fp.property_category = 'Foam Formation' THEN c.compound_name END) AS primary_functional_protein,
        MAX(m.mechanism_name) AS primary_stabilization_mechanism
    FROM workspace.foodmechanics_silver.ingredients i
    JOIN workspace.foodmechanics_silver.nutritional_profiles np 
        ON i.ingredient_id = np.ingredient_id AND np.is_100g_normalized = TRUE
    LEFT JOIN workspace.foodmechanics_silver.functional_properties fp 
        ON i.ingredient_id = fp.ingredient_id
    LEFT JOIN workspace.foodmechanics_silver.chemical_compounds c 
        ON fp.compound_id = c.compound_id
    LEFT JOIN workspace.foodmechanics_silver.mechanism_relationships mr 
        ON i.ingredient_id = mr.ingredient_id AND mr.relationship_type = 'DRIVES'
    LEFT JOIN workspace.foodmechanics_silver.food_mechanisms m 
        ON mr.mechanism_id = m.mechanism_id
    GROUP BY 
        i.ingredient_id, i.common_name, i.category, i.ph_typical, 
        np.calories_kcal, np.protein_g, np.fat_total_g, np.carbohydrates_g, np.water_g;
    """)

    conn.execute("""
    CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_recipe_nutrition AS
    SELECT 
        r.recipe_id,
        r.recipe_name,
        r.category,
        r.target_structure,
        ROUND(SUM(ri.quantity_g), 1) AS total_recipe_weight_g,
        ROUND(SUM(ri.quantity_g * (np.calories_kcal / 100.0)), 1) AS total_recipe_calories,
        ROUND(SUM(ri.quantity_g * (np.protein_g / 100.0)), 2) AS total_protein_g,
        ROUND(SUM(ri.quantity_g * (np.fat_total_g / 100.0)), 2) AS total_fat_g,
        ROUND(SUM(ri.quantity_g * (np.carbohydrates_g / 100.0)), 2) AS total_carbs_g,
        ROUND((SUM(ri.quantity_g * (np.calories_kcal / 100.0)) / SUM(ri.quantity_g)) * 100, 1) AS calories_per_100g,
        ROUND((SUM(ri.quantity_g * (np.protein_g / 100.0)) / SUM(ri.quantity_g)) * 100, 2) AS protein_per_100g,
        ROUND((SUM(ri.quantity_g * (np.fat_total_g / 100.0)) / SUM(ri.quantity_g)) * 100, 2) AS fat_per_100g,
        ROUND((SUM(ri.quantity_g * (np.carbohydrates_g / 100.0)) / SUM(ri.quantity_g)) * 100, 2) AS carbs_per_100g
    FROM workspace.foodmechanics_silver.recipes r
    JOIN workspace.foodmechanics_silver.recipe_ingredients ri 
        ON r.recipe_id = ri.recipe_id
    JOIN workspace.foodmechanics_silver.nutritional_profiles np 
        ON ri.ingredient_id = np.ingredient_id AND np.is_100g_normalized = TRUE
    GROUP BY r.recipe_id, r.recipe_name, r.category, r.target_structure;
    """)

    conn.execute("""
    CREATE OR REPLACE VIEW workspace.foodmechanics_gold.v_mechanism_evidence AS
    SELECT 
        m.mechanism_id,
        m.mechanism_name,
        m.mechanism_class,
        m.structural_outcome,
        m.primary_stabilizing_agent,
        m.temperature_sensitivity,
        mr.relationship_type,
        mr.impact_summary,
        mr.is_hypothesized,
        i.common_name AS ingredient_name,
        c.compound_name AS active_compound,
        se.evidence_id,
        se.title AS paper_title,
        se.authors,
        se.journal_or_book AS journal_name,
        se.publication_year,
        se.doi_or_url AS doi,
        se.study_methodology,
        se.key_findings,
        se.evidence_confidence
    FROM workspace.foodmechanics_silver.food_mechanisms m
    JOIN workspace.foodmechanics_silver.mechanism_relationships mr 
        ON m.mechanism_id = mr.mechanism_id
    JOIN workspace.foodmechanics_silver.ingredients i 
        ON mr.ingredient_id = i.ingredient_id
    LEFT JOIN workspace.foodmechanics_silver.chemical_compounds c 
        ON mr.compound_id = c.compound_id
    LEFT JOIN workspace.foodmechanics_silver.scientific_evidence se 
        ON m.mechanism_id = se.mechanism_id;
    """)

    return conn


def extract_sql_queries(file_path: Path) -> List[Tuple[str, str]]:
    """
    Extracts individual SQL queries and their comment titles from an SQL file.
    Safely ignores semicolons inside single quotes.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by semicolon outside quotes
    queries = []
    current_query = []
    in_quote = False
    quote_char = ""

    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        # Look for title comments like -- QUERY X: Title
        if stripped.startswith("-- QUERY"):
            current_title = stripped.lstrip("-").strip()

        # Parse character by character for semicolon
        i = 0
        while i < len(line):
            ch = line[i]
            if ch in ("'", '"'):
                if not in_quote:
                    in_quote = True
                    quote_char = ch
                elif ch == quote_char:
                    # check for escaped quote
                    if i + 1 < len(line) and line[i + 1] == quote_char:
                        current_query.append(ch)
                        i += 1
                        ch = line[i]
                    else:
                        in_quote = False
            elif ch == ";" and not in_quote:
                query_str = "".join(current_query).strip()
                if query_str:
                    queries.append(query_str)
                current_query = []
                i += 1
                continue
            current_query.append(ch)
            i += 1
        current_query.append("\n")

    remainder = "".join(current_query).strip()
    if remainder:
        queries.append(remainder)

    # Filter out empty or comment-only strings
    clean_queries = []
    for q in queries:
        non_comment_lines = [l for l in q.splitlines() if not l.strip().startswith("--")]
        clean_sql = "\n".join(non_comment_lines).strip()
        if clean_sql:
            # Extract header if present
            header = "Example Query"
            for l in q.splitlines():
                if "QUERY" in l and ":" in l:
                    header = l.strip("-: ").strip()
                    break
            clean_queries.append((header, clean_sql))

    return clean_queries


def run_example_queries(conn, verbose: bool = False) -> Dict[str, Any]:
    """Runs and validates all queries from EXAMPLE_QUERIES.sql."""
    print("\n" + "=" * 70)
    print(" SUITE 1: Verifying FOODMECHANICS Example SQL Assets")
    print("=" * 70)

    queries = extract_sql_queries(EXAMPLE_QUERIES_FILE)
    results = []
    passed = 0

    for idx, (title, sql) in enumerate(queries, 1):
        t0 = time.perf_counter()
        status = "PASSED"
        err_msg = ""
        row_count = 0
        preview = []

        try:
            res = conn.execute(sql).fetchall()
            row_count = len(res)
            duration_ms = (time.perf_counter() - t0) * 1000
            if row_count == 0:
                status = "WARN_EMPTY"
            else:
                passed += 1
                preview = res[:2]
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            status = "FAILED"
            err_msg = str(e)

        results.append({
            "index": idx,
            "title": title,
            "status": status,
            "duration_ms": duration_ms,
            "row_count": row_count,
            "error": err_msg
        })

        symbol = "✅" if status == "PASSED" else ("⚠️" if status == "WARN_EMPTY" else "❌")
        print(f" {symbol} [{idx:02d}] {title[:48]:<48} | {duration_ms:6.1f}ms | {row_count:2d} rows | {status}")
        if verbose and preview:
            print(f"     Preview (1st row): {preview[0]}")
        if err_msg:
            print(f"     Error: {err_msg}")

    return {
        "suite": "EXAMPLE_QUERIES",
        "total": len(queries),
        "passed": passed,
        "failed": len(queries) - passed,
        "details": results
    }


def run_benchmark_questions(conn, verbose: bool = False) -> Dict[str, Any]:
    """Runs all benchmark test cases in QUESTIONS.json and checks criteria."""
    print("\n" + "=" * 70)
    print(" SUITE 2: Benchmarking Evaluation Suite (QUESTIONS.json)")
    print("=" * 70)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    passed = 0

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        golden_sql = q["golden_sql"]
        criteria = q.get("ground_truth_criteria", {})

        t0 = time.perf_counter()
        status = "PASSED"
        err_msg = ""
        row_count = 0
        assertions_passed = True
        assertion_notes = []

        try:
            cur = conn.execute(golden_sql)
            rows = cur.fetchall()
            row_count = len(rows)
            duration_ms = (time.perf_counter() - t0) * 1000

            if row_count == 0:
                status = "FAILED_NO_ROWS"
                assertions_passed = False
                assertion_notes.append("Returned 0 rows")
            else:
                # Specific question assertions
                if qid == "FM-EVAL-001":
                    # Check egg white protein and heavy cream fat
                    for r in rows:
                        name = r[0]
                        prot = float(r[3])
                        fat = float(r[4])
                        if "Egg White" in name and abs(prot - 10.9) > 0.5:
                            assertions_passed = False
                            assertion_notes.append(f"Egg white protein expected ~10.9g, got {prot}g")
                        if "Heavy Whipping Cream" in name and abs(fat - 36.1) > 1.0:
                            assertions_passed = False
                            assertion_notes.append(f"Heavy cream fat expected ~36.1g, got {fat}g")

                elif qid == "FM-EVAL-005":
                    # Check drainage half life for xanthan
                    for r in rows:
                        val = float(r[3])
                        if val < 180.0:
                            assertions_passed = False
                            assertion_notes.append(f"Xanthan drainage half life expected >= 180m, got {val}m")

                elif qid == "FM-EVAL-010":
                    # Check ovalbumin denaturation temp 84.5C
                    for r in rows:
                        cname = r[1]
                        temp = float(r[4])
                        if "Ovalbumin" in cname and abs(temp - 84.5) > 1.0:
                            assertions_passed = False
                            assertion_notes.append(f"Ovalbumin Tm expected ~84.5°C, got {temp}°C")

                if assertions_passed:
                    passed += 1
                else:
                    status = "FAILED_ASSERTION"

        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            status = "FAILED_SYNTAX"
            err_msg = str(e)

        results.append({
            "id": qid,
            "question": q_text,
            "category": q.get("category"),
            "difficulty": q.get("difficulty"),
            "status": status,
            "duration_ms": duration_ms,
            "row_count": row_count,
            "notes": assertion_notes,
            "error": err_msg
        })

        symbol = "✅" if status == "PASSED" else "❌"
        print(f" {symbol} [{qid}] ({q.get('difficulty',''):<7}) {q_text[:40]:<40} | {duration_ms:5.1f}ms | {row_count:2d} rows | {status}")
        if assertion_notes:
            print(f"     Assertion Warnings: {'; '.join(assertion_notes)}")
        if err_msg:
            print(f"     SQL Error: {err_msg}")

    return {
        "suite": "BENCHMARK_QUESTIONS",
        "total": len(questions),
        "passed": passed,
        "failed": len(questions) - passed,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser(description="FOODMECHANICS Genie SQL Evaluator")
    parser.add_argument("--all", action="store_true", default=True, help="Run all evaluation suites")
    parser.add_argument("--example-queries", action="store_true", help="Run only EXAMPLE_QUERIES.sql suite")
    parser.add_argument("--benchmark", action="store_true", help="Run only QUESTIONS.json benchmark suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose query preview")
    parser.add_argument("--save-report", type=str, help="Save evaluation report to markdown/json file")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" FOODMECHANICS: Databricks Genie Agent Evaluation Harness")
    print(f" Target Schemas: workspace.foodmechanics_gold, workspace.foodmechanics_silver")
    print(f" Local Runner Engine: DuckDB In-Memory Emulation")
    print("=" * 70)

    # Initialize environment
    conn = setup_duckdb_environment(FIXTURES_FILE)

    res_examples = None
    res_benchmarks = None

    if args.example_queries and not args.benchmark:
        res_examples = run_example_queries(conn, verbose=args.verbose)
    elif args.benchmark and not args.example_queries:
        res_benchmarks = run_benchmark_questions(conn, verbose=args.verbose)
    else:
        res_examples = run_example_queries(conn, verbose=args.verbose)
        res_benchmarks = run_benchmark_questions(conn, verbose=args.verbose)

    # Overall Summary
    print("\n" + "=" * 70)
    print(" EVALUATION SUMMARY REPORT")
    print("=" * 70)
    total_tests = 0
    total_passed = 0

    if res_examples:
        total_tests += res_examples["total"]
        total_passed += res_examples["passed"]
        pct = (res_examples["passed"] / res_examples["total"]) * 100
        print(f" Suite 1 (Example Queries):  {res_examples['passed']:2d}/{res_examples['total']:2d} passed ({pct:5.1f}%)")

    if res_benchmarks:
        total_tests += res_benchmarks["total"]
        total_passed += res_benchmarks["passed"]
        pct = (res_benchmarks["passed"] / res_benchmarks["total"]) * 100
        print(f" Suite 2 (Benchmark Cases): {res_benchmarks['passed']:2d}/{res_benchmarks['total']:2d} passed ({pct:5.1f}%)")

    overall_pct = (total_passed / total_tests) * 100 if total_tests else 0
    print("-" * 70)
    print(f" OVERALL RESULT: {total_passed}/{total_tests} passed ({overall_pct:5.1f}%)")
    print("=" * 70 + "\n")

    if args.save_report:
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_percent": overall_pct,
            "example_queries": res_examples,
            "benchmark_questions": res_benchmarks
        }
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved evaluation report to: {args.save_report}")

    if total_passed < total_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
