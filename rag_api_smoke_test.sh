#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "[1/2] health check: ${BASE_URL}/api/v1/health"
curl -sS "${BASE_URL}/api/v1/health" | cat
echo

echo "[2/2] search check: ${BASE_URL}/api/v1/search"
curl -sS -X POST "${BASE_URL}/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "electric aircraft magnetic gear",
    "top_k": 3,
    "search_mode": "hybrid",
    "use_reranker": false
  }' | cat
echo

echo "Smoke test completed."
