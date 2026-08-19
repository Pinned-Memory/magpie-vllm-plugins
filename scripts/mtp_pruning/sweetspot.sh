#!/usr/bin/env bash
# Standalone, resumable vLLM sweet-spot sweep — magpie adaptation of the
# shared rtx-pro-6000-sweetspot.sh, tuned for the GB10 (DGX Spark) box and
# extended with the mtp-pruning arm.
#
# What it measures, per (mtp mode × input length × concurrency), RUNS times
# with a cold prefix cache each run: output_throughput (includes prefill in
# the denominator) AND tpot_implied_decode_tps (decode-only, prefill
# excluded) — the pair separates "how much wall the prefill eats" from "how
# fast decode actually runs".
#
# Differences from the shared script:
#   * MTP_MODES accepts "pruned" in addition to "off"/"on": MTP with the
#     draft head sliced to DRAFT_VOCAB_PATH (magpie-spark-vllm mtp-pruning
#     patch required; the script preflights for it).
#   * MTP_K sets num_speculative_tokens for on/pruned modes (default 3).
#   * Waits for MIN_FREE_GB of unified memory before each server start
#     (this box is shared with other sessions' engines).
#   * Inline python runs with -P: a cwd containing a vllm/ source checkout
#     must not shadow the installed package.
#
# CAVEAT for on/pruned modes: RandomDataset prompts are random tokens, so
# MTP acceptance here does NOT represent real-traffic acceptance (fit and
# judge keep-sets on real captures — see docs/mtp-pruning.md). Head-cost
# and throughput scaling across context lengths are what this sweep is for.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./sweetspot.sh [MODEL_PATH] [RESULT_ROOT]

Defaults for this box:
  MODEL_PATH                  /home/cc2869/models/Qwen3.8-27B-NVFP4
  VLLM_BIN                    /home/cc2869/research/magpie/.venv/bin/vllm

Important environment overrides:
  SERVED_MODEL_NAME           API model name (default: Qwen/Qwen3.8-27B)
  LENGTHS                     Input lengths (default: "4096 16384 65536 131072")
  CONCURRENCIES               Concurrencies (default: "8 16")
  MTP_MODES                   any of "off on pruned" (default: "off on pruned")
  MTP_K                       num_speculative_tokens for on/pruned (default: 3)
  DATASET                     "random" (exact-length synthetic, default) or
                              "sharegpt" (realistic conversations; LENGTHS is
                              ignored and acceptance becomes meaningful)
  SHAREGPT_PATH               ShareGPT_V3_unfiltered_cleaned_split.json
                              (default: ~/datasets/sharegpt/...)
  NUM_PROMPTS                 requests per run in sharegpt mode (default: 64)
  DRAFT_VOCAB_PATH            keep-set for pruned mode
                              (default: magpie keep_p99.pt)
  OUTPUT_LEN                  Generated tokens/request (default: 512)
  RUNS                        Cold-prefix repeats/point (default: 3)
  GPU_MEMORY_UTILIZATION      Auto KV-cache memory target (default: 0.90)
  MIN_FREE_GB, MEM_WAIT_S     Shared-box memory gate (default: 105 GB, 3600 s)
  SERVER_HOST, PORT           Local bind address/port (default: 127.0.0.1:8000)
  MAX_MODEL_LEN               Defaults to max(LENGTHS) + OUTPUT_LEN

Example smoke (one condition, ~12 min):
  RUNS=1 LENGTHS=4096 CONCURRENCIES=8 MTP_MODES=pruned \
    ./sweetspot.sh /home/cc2869/models/Qwen3.8-27B-NVFP4 ./smoke-results

Rerunning the same command and result directory resumes missing repeats.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${MODEL_PATH:-${1:-/home/cc2869/models/Qwen3.8-27B-NVFP4}}"
MODEL_PATH="$(readlink -f "$MODEL_PATH")"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "MODEL_PATH must contain config.json: $MODEL_PATH" >&2
  exit 1
fi

VLLM_BIN="${VLLM_BIN:-/home/cc2869/research/magpie/.venv/bin/vllm}"
[[ -x "$VLLM_BIN" ]] || VLLM_BIN="$(command -v vllm || true)"
if [[ -z "$VLLM_BIN" || ! -x "$VLLM_BIN" ]]; then
  echo "vllm executable not found; set VLLM_BIN." >&2
  exit 1
fi
VLLM_BIN="$(readlink -f "$VLLM_BIN")"
VENV_BIN="$(dirname "$VLLM_BIN")"
PYTHON_BIN="${PYTHON_BIN:-$VENV_BIN/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found next to vllm: $PYTHON_BIN" >&2
  exit 1
fi

for command_name in curl jq nvidia-smi setsid awk; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
if ! "$PYTHON_BIN" -P -c "import pandas" 2>/dev/null; then
  echo "pandas is required by vllm bench dataset loaders:" >&2
  echo "  uv pip install --python $PYTHON_BIN pandas" >&2
  exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B}"
LENGTHS="${LENGTHS:-4096 16384 65536 131072}"
CONCURRENCIES="${CONCURRENCIES:-8 16}"
MTP_MODES="${MTP_MODES:-off on pruned}"
MTP_K="${MTP_K:-3}"
DATASET="${DATASET:-random}"
SHAREGPT_PATH="${SHAREGPT_PATH:-/home/cc2869/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
DRAFT_VOCAB_PATH="${DRAFT_VOCAB_PATH:-/home/cc2869/research/magpie/experiments/vocab-pruning/mtp-prune/keep_p99.pt}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
RUNS="${RUNS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MIN_FREE_GB="${MIN_FREE_GB:-105}"
MEM_WAIT_S="${MEM_WAIT_S:-3600}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="http://$SERVER_HOST:$PORT"
SEED="${SEED:-0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-gb10}"
RESULT_ROOT="${RESULT_ROOT:-${2:-$PWD/sweetspot-results/$RUN_ID}}"
RESULT_ROOT="$(mkdir -p "$RESULT_ROOT" && readlink -f "$RESULT_ROOT")"

if ! [[ "$OUTPUT_LEN" =~ ^[1-9][0-9]*$ && "$RUNS" =~ ^[1-9][0-9]*$ && "$MTP_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "OUTPUT_LEN, RUNS and MTP_K must be positive integers." >&2
  exit 1
fi

case "$DATASET" in random|sharegpt) ;; *)
  echo "DATASET must be random or sharegpt: $DATASET" >&2; exit 1 ;;
esac
if [[ "$DATASET" == "sharegpt" ]]; then
  if [[ ! -f "$SHAREGPT_PATH" ]]; then
    echo "SHAREGPT_PATH not found: $SHAREGPT_PATH" >&2
    echo "Download: curl -L -o \$SHAREGPT_PATH https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" >&2
    exit 1
  fi
  if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROMPTS must be a positive integer." >&2; exit 1
  fi
  LENGTHS="sharegpt"          # natural prompt lengths; one bucket
fi

max_input_len=0
max_concurrency=0
if [[ "$DATASET" == "random" ]]; then
  for input_len in $LENGTHS; do
    if ! [[ "$input_len" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid input length: $input_len" >&2
      exit 1
    fi
    (( input_len > max_input_len )) && max_input_len=$input_len
  done
fi
for concurrency in $CONCURRENCIES; do
  if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid concurrency: $concurrency" >&2
    exit 1
  fi
  (( concurrency > max_concurrency )) && max_concurrency=$concurrency
done
if [[ "$DATASET" == "sharegpt" ]]; then
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((max_input_len + OUTPUT_LEN))}"
if (( MAX_MODEL_LEN < max_input_len + OUTPUT_LEN )); then
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN is smaller than input+output=$((max_input_len + OUTPUT_LEN))." >&2
  exit 1
fi
for mtp_mode in $MTP_MODES; do
  case "$mtp_mode" in off|on|pruned) ;; *)
    echo "MTP_MODES contains unsupported value: $mtp_mode" >&2
    exit 1 ;;
  esac
done

if [[ " $MTP_MODES " == *" pruned "* ]]; then
  if [[ ! -f "$DRAFT_VOCAB_PATH" ]]; then
    echo "DRAFT_VOCAB_PATH not found: $DRAFT_VOCAB_PATH" >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -P -c '
import dataclasses
from vllm.config.speculative import SpeculativeConfig
assert any(f.name == "draft_vocab_path" for f in dataclasses.fields(SpeculativeConfig))
' 2>/dev/null; then
    echo "This vLLM lacks draft_vocab_path — apply the magpie-spark-vllm" >&2
    echo "mtp-pruning patch before using MTP_MODES=pruned." >&2
    exit 1
  fi
fi

export PATH="$VENV_BIN:$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MAX_JOBS="${MAX_JOBS:-4}"

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT -- "-$server_pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

if pgrep -af '[v]llm serve' >/dev/null; then
  echo "A vLLM server is already running; refusing to reuse or stop it." >&2
  exit 1
fi

wait_for_memory() {  # shared box: another session's engine may hold memory
  local waited=0 a b
  while (( waited < MEM_WAIT_S )); do
    a=$(free -g | awk 'NR==2{print $7}')
    if (( a >= MIN_FREE_GB )); then
      sleep 15
      b=$(free -g | awk 'NR==2{print $7}')
      (( b >= MIN_FREE_GB )) && return 0
      waited=$((waited + 15))
    fi
    sleep 20
    waited=$((waited + 20))
  done
  echo "Memory never reached ${MIN_FREE_GB}G within ${MEM_WAIT_S}s (last: ${a}G)." >&2
  return 1
}

config_text="$(printf '%s\n' \
  "MODEL_PATH=$MODEL_PATH" \
  "VLLM_BIN=$VLLM_BIN" \
  "SERVED_MODEL_NAME=$SERVED_MODEL_NAME" \
  "LENGTHS=$LENGTHS" \
  "CONCURRENCIES=$CONCURRENCIES" \
  "MTP_MODES=$MTP_MODES" \
  "MTP_K=$MTP_K" \
  "DATASET=$DATASET" \
  "SHAREGPT_PATH=$SHAREGPT_PATH" \
  "NUM_PROMPTS=$NUM_PROMPTS" \
  "DRAFT_VOCAB_PATH=$DRAFT_VOCAB_PATH" \
  "OUTPUT_LEN=$OUTPUT_LEN" \
  "RUNS=$RUNS" \
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" \
  "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" \
  "SEED=$SEED")"
config_path="$RESULT_ROOT/experiment.env"
if [[ -f "$config_path" && "$(<"$config_path")" != "$config_text" ]]; then
  echo "Existing result configuration differs: $config_path" >&2
  diff -u "$config_path" <(printf '%s\n' "$config_text") >&2 || true
  echo "Use the original settings or choose a new RESULT_ROOT." >&2
  exit 1
fi
printf '%s\n' "$config_text" >"$config_path"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader >"$RESULT_ROOT/gpu.csv"
"$VLLM_BIN" --version >"$RESULT_ROOT/vllm-version.txt"

echo "Results: $RESULT_ROOT"
echo "GPU: $(<"$RESULT_ROOT/gpu.csv")"
echo "Auto KV-cache target: GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"

# Generate exact-token, deterministic fixtures using vLLM's RandomDataset.
[[ "$DATASET" == "random" ]] && for input_len in $LENGTHS; do
  dataset_path="$RESULT_ROOT/datasets/input-$input_len.jsonl"
  if [[ ! -f "$dataset_path" ]]; then
    mkdir -p "$(dirname "$dataset_path")"
    "$PYTHON_BIN" -P - "$MODEL_PATH" "$input_len" "$max_concurrency" "$SEED" "$dataset_path" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm.benchmarks.datasets import RandomDataset

model_path, input_len, count, seed, output_path = sys.argv[1:]
input_len, count, seed = map(int, (input_len, count, seed))
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
requests = RandomDataset(random_seed=seed).sample(
    tokenizer,
    num_requests=count,
    prefix_len=0,
    range_ratio=0.0,
    input_len=input_len,
    output_len=1,
)
with Path(output_path).open("w", encoding="utf-8") as handle:
    for request in requests:
        if request.prompt_len != input_len:
            raise RuntimeError(f"expected {input_len} tokens, got {request.prompt_len}")
        handle.write(json.dumps({"prompt": request.prompt}) + "\n")
PY
  fi
done

completed_runs() {
  local summary_path="$1"
  if [[ ! -f "$summary_path" ]]; then
    echo 0
  else
    awk 'NR > 1 { count++ } END { print count + 0 }' "$summary_path"
  fi
}

wait_for_server() {
  for ((second = 0; second < SERVER_START_TIMEOUT; second++)); do
    if curl --fail --silent "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    kill -0 "$server_pid" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

for mtp_mode in $MTP_MODES; do
  for input_len in $LENGTHS; do
    server_dir="$RESULT_ROOT/mtp-$mtp_mode/input-$input_len"
    mkdir -p "$server_dir"
    needs_server=0
    for concurrency in $CONCURRENCIES; do
      summary_path="$server_dir/c$concurrency/summary.csv"
      if (( $(completed_runs "$summary_path") < RUNS )); then
        needs_server=1
      else
        echo "Skipping complete: mtp=$mtp_mode input=$input_len c=$concurrency"
      fi
    done
    (( needs_server == 1 )) || continue

    server_args=(
      serve "$MODEL_PATH"
      --served-model-name "$SERVED_MODEL_NAME"
      --host "$SERVER_HOST"
      --port "$PORT"
      --tensor-parallel-size 1
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$max_concurrency"
      --kv-cache-dtype fp8
      --enable-prefix-caching
    )
    case "$mtp_mode" in
      on)
        server_args+=(--speculative-config \
          "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}") ;;
      pruned)
        server_args+=(--speculative-config \
          "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K,\"draft_vocab_path\":\"$DRAFT_VOCAB_PATH\"}") ;;
    esac

    echo "Waiting for >=${MIN_FREE_GB}G free before: mtp=$mtp_mode input=$input_len"
    wait_for_memory
    echo "Starting: mtp=$mtp_mode input=$input_len"
    setsid "$VLLM_BIN" "${server_args[@]}" >>"$server_dir/server.log" 2>&1 &
    server_pid=$!
    if ! wait_for_server; then
      echo "Server failed to become healthy; log tail:" >&2
      tail -100 "$server_dir/server.log" >&2 || true
      exit 1
    fi
    if [[ "$mtp_mode" == "pruned" ]] && \
       ! grep -q "MTP draft vocab pruned" "$server_dir/server.log"; then
      echo "Pruned mode requested but the prune log line is missing — the" >&2
      echo "arm would silently measure the baseline. Aborting." >&2
      exit 1
    fi
    nvidia-smi >"$server_dir/nvidia-smi-after-start.txt"

    for concurrency in $CONCURRENCIES; do
      condition_dir="$server_dir/c$concurrency"
      mkdir -p "$condition_dir"
      summary_path="$condition_dir/summary.csv"
      current_runs="$(completed_runs "$summary_path")"
      (( current_runs < RUNS )) || continue

      if [[ ! -f "$summary_path" ]]; then
        printf '%s\n' \
          'run,completed,failed,duration_s,output_throughput_tps,tpot_implied_decode_tps,mean_tpot_ms,p99_tpot_ms,mean_itl_ms,p99_itl_ms,mean_ttft_ms,p99_ttft_ms,mtp_acceptance_rate,mtp_acceptance_length' \
          >"$summary_path"
      fi
      dataset_path="$RESULT_ROOT/datasets/input-$input_len.jsonl"   # unused in sharegpt mode

      for ((run = current_runs + 1; run <= RUNS; run++)); do
        echo "Run $run/$RUNS: mtp=$mtp_mode input=$input_len c=$concurrency"
        reset_result="$(curl --fail --silent --show-error \
          --request POST "$BASE_URL/reset_prefix_cache")"
        jq --exit-status '.success == true' <<<"$reset_result" >/dev/null
        printf '%s\n' "$reset_result" >"$condition_dir/reset-before-run-$run.json"
        curl --fail --silent "$BASE_URL/metrics" \
          >"$condition_dir/metrics-before-run-$run.prom"

        if [[ "$DATASET" == "sharegpt" ]]; then
          dataset_args=(--dataset-name sharegpt --dataset-path "$SHAREGPT_PATH"
                        --sharegpt-output-len "$OUTPUT_LEN"
                        --num-prompts "$NUM_PROMPTS")
        else
          dataset_args=(--dataset-name custom --dataset-path "$dataset_path"
                        --skip-chat-template --disable-shuffle
                        --custom-output-len "$OUTPUT_LEN"
                        --num-prompts "$concurrency")
        fi
        "$VLLM_BIN" bench serve \
          --backend openai \
          --base-url "$BASE_URL" \
          --endpoint /v1/completions \
          --model "$SERVED_MODEL_NAME" \
          --tokenizer "$MODEL_PATH" \
          "${dataset_args[@]}" \
          --num-warmups 0 \
          --request-rate inf \
          --max-concurrency "$concurrency" \
          --ignore-eos \
          --temperature 0 \
          --seed "$SEED" \
          --disable-tqdm \
          --percentile-metrics ttft,tpot,itl,e2el \
          --metric-percentiles 50,90,99 \
          --save-result \
          --save-detailed \
          --result-dir "$condition_dir" \
          --result-filename "run-$run.json" \
          >"$condition_dir/run-$run.log" 2>&1

        curl --fail --silent "$BASE_URL/metrics" \
          >"$condition_dir/metrics-after-run-$run.prom"
        jq --raw-output --argjson run "$run" --argjson concurrency "$concurrency" '[
          $run, .completed, .failed, .duration, .output_throughput,
          ($concurrency * 1000 / .mean_tpot_ms), .mean_tpot_ms, .p99_tpot_ms,
          .mean_itl_ms, .p99_itl_ms, .mean_ttft_ms, .p99_ttft_ms,
          (.spec_decode_acceptance_rate // null),
          (.spec_decode_acceptance_length // null)
        ] | @csv' "$condition_dir/run-$run.json" >>"$summary_path"
      done
    done
    stop_server
  done
done

# Produce one portable aggregate CSV and validate every requested condition.
"$PYTHON_BIN" -P - "$RESULT_ROOT" "$MTP_MODES" "$LENGTHS" "$CONCURRENCIES" "$RUNS" "$DATASET" "$NUM_PROMPTS" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
modes, lengths, concurrencies = (value.split() for value in sys.argv[2:5])
expected_runs = int(sys.argv[5])
dataset, num_prompts = sys.argv[6], int(sys.argv[7])
columns = [
    "duration_s", "output_throughput_tps", "tpot_implied_decode_tps",
    "mean_tpot_ms", "p99_tpot_ms", "mean_ttft_ms", "p99_ttft_ms",
    "mtp_acceptance_rate", "mtp_acceptance_length",
]
output_rows = []
for mode in modes:
    for length in lengths:
        for concurrency in map(int, concurrencies):
            expected_completed = num_prompts if dataset == "sharegpt" else concurrency
            path = root / f"mtp-{mode}" / f"input-{length}" / f"c{concurrency}" / "summary.csv"
            rows = list(csv.DictReader(path.open()))
            if len(rows) != expected_runs:
                raise RuntimeError(f"{path}: expected {expected_runs} runs, got {len(rows)}")
            if any(int(row["completed"]) != expected_completed or int(row["failed"]) for row in rows):
                raise RuntimeError(f"failed/incomplete requests in {path}")
            aggregate = {"mtp": mode, "input_len": length, "concurrency": concurrency}
            for column in columns:
                values = [float(row[column]) for row in rows if row[column]]
                aggregate[column] = statistics.mean(values) if values else ""
            output_rows.append(aggregate)

with (root / "aggregate.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["mtp", "input_len", "concurrency", *columns])
    writer.writeheader()
    writer.writerows(output_rows)
print(f"Validated {len(output_rows)} conditions; wrote {root / 'aggregate.csv'}")
PY

trap - EXIT INT TERM
echo "Sweet-spot sweep complete: $RESULT_ROOT"
