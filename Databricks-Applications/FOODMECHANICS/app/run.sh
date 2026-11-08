#!/usr/bin/env bash
# FOODMECHANICS Application Startup Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================="
echo " Starting FOODMECHANICS — Food Science Investigation Platform"
echo "=========================================================="

cd "${SCRIPT_DIR}"

# Locate Python environment
PYTHON_BIN=""
if [ -f "${WORKSPACE_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
elif [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: Python binary not found."
    exit 1
fi

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "Using Python: ${PYTHON_BIN}"
echo "Serving FOODMECHANICS on http://${HOST}:${PORT}"
echo "Dual-Engine: Auto-detecting Databricks SQL or DuckDB Local Fixtures"
echo "=========================================================="

exec "${PYTHON_BIN}" -m uvicorn main:app --host "${HOST}" --port "${PORT}"
