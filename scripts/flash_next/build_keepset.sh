#!/usr/bin/env bash
# Fit an MTP draft keep-set for Qwen3.8-Flash-Next from its own generations
# (the JSON outputs of bench_decode.py / gsm8k_eval.py in RESULT_DIR) and
# union it with the Qwen3.8-27B agent keep-set (same tokenizer, vocab.json
# md5-identical), following docs/mtp-pruning.md.
#   scripts/flash_next/build_keepset.sh RESULT_DIR OUT.pt [COVERAGE=0.99]
set -euo pipefail
RES=$1; OUT=$2; COV=${3:-0.99}
PY=${PY:-/home/cc2869/research/magpie/.venv/bin/python}
MODEL=${MODEL:-/home/cc2869/.cache/huggingface/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/103a7608316173ca6edd49929544244de7ffda70}
AGENT_KEEP=${AGENT_KEEP:-/home/cc2869/research/magpie/experiments/vocab-pruning/mtp-prune/keep_p99.pt}
D=$(dirname "$0"); MP=$D/../mtp_pruning
"$PY" - "$RES" > "$RES/generations.jsonl" <<'PYEOF'
import glob, json, sys
for f in glob.glob(f"{sys.argv[1]}/*.json"):
    j = json.load(open(f))
    for r in j.get("records", []) + j.get("requests", []):
        if r.get("text"):
            print(json.dumps({"text": r["text"]}))
PYEOF
wc -l "$RES/generations.jsonl"
"$PY" "$MP/count_tokens.py" --tokenizer "$MODEL" --out "$RES/counts.npy" "$RES/generations.jsonl"
"$PY" "$MP/build_keepset.py" --counts "$RES/counts.npy" --coverage "$COV" --tokenizer "$MODEL" --out "$RES/keep_fitted.pt"
"$PY" - "$RES/keep_fitted.pt" "$AGENT_KEEP" "$OUT" <<'PYEOF'
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
u = torch.unique(torch.cat([a.to(torch.int64), b.to(torch.int64)]))
torch.save(u, sys.argv[3])
print(f"fitted {a.numel()} + agent {b.numel()} -> union {u.numel()} ids saved to {sys.argv[3]}")
PYEOF
