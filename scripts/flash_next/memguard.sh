#!/usr/bin/env bash
# Kill the vLLM server if MemAvailable drops below MIN_GB (default 6) so the
# unified-memory GB10 never reaches the OOM killer. Logs to $1.
LOG=$1; MIN_GB=${MIN_GB:-6}
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  echo "$(date +%T) avail_gb=$avail" >> "$LOG"
  if [ "$avail" -lt "$MIN_GB" ]; then
    echo "$(date +%T) MemAvailable ${avail} GB < ${MIN_GB} GB: killing vllm" >> "$LOG"
    pkill -f "[v]llm serve"; sleep 2; pkill -9 -f "[v]llm serve"; pkill -9 -f "[E]ngineCore"
  fi
  sleep 2
done
