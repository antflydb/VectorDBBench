#!/usr/bin/env bash
# Run Chroma benchmarks in an isolated venv to avoid pydantic version conflicts.
# Usage: bash scripts/run_chroma_bench.sh <case_type>
#   e.g. bash scripts/run_chroma_bench.sh Performance1536D50K
set -euo pipefail

CASE_TYPE="${1:?Usage: $0 <case_type>}"
VENV_DIR=".venv-chroma"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

# Create isolated venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating isolated Chroma venv at $VENV_DIR"
    uv venv "$VENV_DIR"

    # Install VectorDBBench with relaxed pydantic pin for chromadb compatibility
    echo "==> Installing VectorDBBench + chromadb in isolated venv"
    # Create a patched pyproject.toml that allows pydantic>=2
    sed 's/"pydantic<v2"/"pydantic>=2"/' pyproject.toml > "$VENV_DIR/pyproject-chroma.toml"

    # Install chromadb + pydantic v2 first, then project without deps
    echo "==> Installing chromadb with pydantic v2"
    uv pip install --python "$VENV_DIR/bin/python" "pydantic>=2" chromadb httpx numpy environs
    echo "==> Installing VectorDBBench (no-deps to avoid pydantic conflict)"
    uv pip install --python "$VENV_DIR/bin/python" -e . --no-deps
fi

echo "==> Running Chroma benchmark: $CASE_TYPE"
"$VENV_DIR/bin/python" -m vectordb_bench.cli.vectordbbench chroma \
    --host localhost --port 8000 \
    --m 16 --ef-construct 256 --ef-search 256 \
    --case-type "$CASE_TYPE" \
    --drop-old --skip-search-concurrent \
    --db-label chroma-local
