#!/usr/bin/env bash
# Run the measurement suite against a live server: decode bench at c=1 and
# c=4, then GSM8K (n=${GSM8K_N:-200}). Results land in $1 as JSON.
set -euo pipefail
OUT=$1; PY=${PY:-/home/cc2869/research/magpie/.venv/bin/python}; URL=${URL:-http://127.0.0.1:8100}
D=$(dirname "$0")
"$PY" "$D/bench_decode.py" --url "$URL" --concurrency 1 --num-prompts 8 --out "$OUT/bench_c1.json"
"$PY" "$D/bench_decode.py" --url "$URL" --concurrency 4 --num-prompts 16 --out "$OUT/bench_c4.json"
"$PY" "$D/gsm8k_eval.py" --url "$URL" -n "${GSM8K_N:-200}" --concurrency 4 --out "$OUT/gsm8k.json"
curl -s "$URL/metrics" > "$OUT/metrics-after.prom"
echo SUITE_DONE
