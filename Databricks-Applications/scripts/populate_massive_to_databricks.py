"""
scripts/populate_massive_to_databricks.py
Populates expanded, high-fidelity datasets directly into Databricks Serverless Delta Lake.
"""

import os
import json
import warnings
from databricks import sql

warnings.filterwarnings('ignore')

# Read credentials from .env
env_vars = {}
with open('.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            env_vars[k] = v

server_hostname = env_vars['DATABRICKS_SERVER_HOSTNAME']
http_path = env_vars['DATABRICKS_HTTP_PATH']
access_token = env_vars['DATABRICKS_TOKEN']

print(f"Connecting to Databricks Serverless: {server_hostname}...")

conn = sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=access_token,
    _tls_no_verify=True
)
cursor = conn.cursor()
cursor.execute("USE CATALOG workspace")
print("Connected to Unity Catalog.")

def chunked(lst, size=25):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# --------------------------------------------------------------------------
# 1. POPULATE SIGNAL
# --------------------------------------------------------------------------
print("\n--- Populating SIGNAL Silver Tables ---")
with open("SIGNAL/data/fixtures/massive_signal_data.json") as f:
    signal_data = json.load(f)

# Models
# model_id, name, provider, release_date, architecture, parameter_count_b, context_window, license, is_open_weights, pricing_input_per_m, pricing_output_per_m
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_models")
for chunk in chunked(signal_data["models"]):
    ph = ", ".join(["(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
    params = []
    for m in chunk:
        # estimate param count
        arch = m.get("architecture", "")
        params_b = 671.0 if "671" in arch else 405.0 if "405" in arch else 70.0 if "70" in arch else 32.0 if "32" in arch else 14.0 if "14" in arch else 8.0 if "8" in arch else 123.0 if "123" in arch else 27.0 if "27" in arch else 104.0 if "104" in arch else 150.0
        params.extend([
            m["model_id"], m["name"], m["provider"], m.get("release_date", "2024-01-01"),
            m["architecture"], float(params_b), int(m.get("context_window", 128000)),
            m.get("license", "Proprietary"), bool(m.get("open_weights", False)),
            float(m.get("input_price_per_m", 1.0)), float(m.get("output_price_per_m", 2.0))
        ])
    cursor.execute(f"INSERT INTO workspace.signal_silver.silver_models VALUES {ph}", params)
print(f"Loaded {len(signal_data['models'])} AI models into silver_models.")

# Benchmarks
# benchmark_id, name, evaluation_metric, task_type
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_benchmarks")
for b in signal_data["benchmarks"]:
    cursor.execute(
        "INSERT INTO workspace.signal_silver.silver_benchmarks VALUES (?, ?, ?, ?)",
        [b["benchmark_id"], b["name"], b["metric"], b["category"]]
    )
print(f"Loaded {len(signal_data['benchmarks'])} benchmarks.")

# Evaluations
# eval_id, model_id, benchmark_id, score, eval_date, methodology
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_evaluations")
for chunk in chunked(signal_data["evaluations"]):
    ph = ", ".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
    params = []
    for ev in chunk:
        params.extend([
            ev["eval_id"], ev["model_id"], ev["benchmark_id"], float(ev["score"]),
            ev.get("evaluation_date", "2025-01-01"), ev.get("methodology", "Official Harness")
        ])
    cursor.execute(f"INSERT INTO workspace.signal_silver.silver_evaluations VALUES {ph}", params)
print(f"Loaded {len(signal_data['evaluations'])} benchmark evaluations.")

# Tools
# tool_id, name, category, repo_url, release_version, ecosystem_role
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_tools")
for t in signal_data["tools"]:
    cursor.execute(
        "INSERT INTO workspace.signal_silver.silver_tools VALUES (?, ?, ?, ?, ?, ?)",
        [t["tool_id"], t["name"], t["category"], t["source_url"], "v1.0.0", t["description"]]
    )
print(f"Loaded {len(signal_data['tools'])} tools & protocols.")

# Developments
# development_id, title, summary, category, date, confidence_level, impact_score, source_url
cursor.execute("TRUNCATE TABLE workspace.signal_silver.silver_developments")
for d in signal_data["developments"]:
    cursor.execute(
        "INSERT INTO workspace.signal_silver.silver_developments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [d["development_id"], d["title"], d["summary"], d["theme"], d["event_date"], d["evidence_classification"], float(d["impact_score"]), d["source_url"]]
    )
print(f"Loaded {len(signal_data['developments'])} ecosystem developments.")

cursor.close()
conn.close()
print("\nAll Databricks Serverless Delta Lake tables successfully populated!")
