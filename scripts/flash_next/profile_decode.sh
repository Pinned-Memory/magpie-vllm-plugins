#!/usr/bin/env bash
# Capture a torch-profiler window of steady-state decode on a live server
# launched with --profiler-config '{"profiler":"torch","torch_profiler_dir":DIR}'.
#   scripts/flash_next/profile_decode.sh [URL] [MAX_TOKENS]
set -euo pipefail
URL=${1:-http://127.0.0.1:8100}; N=${2:-64}
PY=${PY:-/home/cc2869/research/magpie/.venv/bin/python}
"$PY" - "$URL" "$N" <<'PYEOF'
import sys, requests, time
url, n = sys.argv[1], int(sys.argv[2])
body = {"model": "Qwen/Qwen3.8-Flash-Next", "temperature": 0, "max_tokens": n,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Explain how a B-tree index speeds up range queries, with an example."}]}
requests.post(f"{url}/v1/chat/completions", json=body).raise_for_status()  # warm (prefix + page cache)
requests.post(f"{url}/start_profile").raise_for_status()
t = time.time(); r = requests.post(f"{url}/v1/chat/completions", json=body).json()
requests.post(f"{url}/stop_profile").raise_for_status()
print("profiled", r["usage"]["completion_tokens"], "tokens in %.2fs" % (time.time() - t))
PYEOF
