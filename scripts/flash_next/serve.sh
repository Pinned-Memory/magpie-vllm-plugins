#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next-NVFP4 on one GB10 with the 102 GB n-gram (PLE)
# table parked on SSD (patches/ple-ssd applied to the vLLM checkout).
#
#   scripts/flash_next/serve.sh LOGFILE [extra vllm flags...]
#
# Env: MODEL (snapshot dir), PORT (8100), MTP (3; 0 = off),
#      DRAFT_VOCAB_PATH (keep-set .pt -> mtp-pruning), MAX_MODEL_LEN (65536),
#      MAX_NUM_SEQS (8), GPU_MEM (0.80), VLLM_BIN.
set -euo pipefail
LOG=$1; shift
MODEL=${MODEL:-/home/cc2869/.cache/huggingface/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/103a7608316173ca6edd49929544244de7ffda70}
VLLM_BIN=${VLLM_BIN:-/home/cc2869/research/magpie/.venv/bin/vllm}
PORT=${PORT:-8100}
MTP=${MTP:-3}
SPEC=()
if [ "$MTP" != "0" ]; then
  if [ -n "${DRAFT_VOCAB_PATH:-}" ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"draft_vocab_path\":\"$DRAFT_VOCAB_PATH\"}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
  fi
fi
# FlashInfer JIT-builds its sm_120 MoE kernels on first use: it needs ninja
# (in the venv) and nvcc on PATH.
export PATH="$(dirname "$VLLM_BIN"):/usr/local/cuda/bin:$PATH"
cd /  # a cwd containing a vllm/ source tree must not shadow the installed package
exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name Qwen/Qwen3.8-Flash-Next \
  --hf-overrides '{"ple_mmap": true}' \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --gpu-memory-utilization "${GPU_MEM:-0.80}" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --port "$PORT" \
  "${SPEC[@]}" "$@" > "$LOG" 2>&1
