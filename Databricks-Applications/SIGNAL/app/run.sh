#!/usr/bin/env bash
# ==============================================================================
# SIGNAL (AI Landscape Observatory) Application Startup Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "---------------------------------------------------------------"
echo "  SIGNAL — AI Landscape Observatory Application Launcher       "
echo "---------------------------------------------------------------"

# Detect Python virtualenv
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    echo "Error: Python executable not found in .venv or PATH."
    exit 1
fi

echo "Using Python: $PYTHON_EXEC"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if fixtures exist, generate if missing
FIXTURES_FILE="$PROJECT_ROOT/SIGNAL/data/fixtures/signal_fixtures.json"
if [ ! -f "$FIXTURES_FILE" ]; then
    echo "Generating offline fixtures dataset..."
    "$PYTHON_EXEC" "$PROJECT_ROOT/SIGNAL/data/fixtures/generate_fixtures.py"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting SIGNAL FastAPI backend on http://${HOST}:${PORT}"
echo "Press Ctrl+C to terminate."
echo "---------------------------------------------------------------"

exec "$PYTHON_EXEC" -m uvicorn SIGNAL.app.main:app --host "$HOST" --port "$PORT" --reload
