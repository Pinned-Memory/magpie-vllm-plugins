#!/usr/bin/env python3
"""Summarize a torch-profiler Chrome trace of a decode window.

Prints wall time, GPU kernel busy time, CUDA-graph launches, and the CPU
time spent in the ple-ssd staging (stage_rows / gather_rows and the ops it
calls), so the SSD gather cost per decode step is a measured number.
  scripts/flash_next/trace_summary.py TRACE.json[.gz]
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
    events = json.load(f)["traceEvents"]
dur = [e for e in events if e.get("ph") == "X"]
t0 = min(e["ts"] for e in dur)
t1 = max(e["ts"] + e["dur"] for e in dur)
wall_ms = (t1 - t0) / 1e3
kernels = [e for e in dur if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
gpu_busy_ms = sum(e["dur"] for e in kernels) / 1e3
graph_launches = sum(1 for e in dur if "cudaGraphLaunch" in e.get("name", ""))
by_name = defaultdict(float)
for e in dur:
    by_name[e["name"]] += e["dur"]
def total(sub):
    return sum(v for k, v in by_name.items() if sub in k) / 1e3
staging = {k: v / 1e3 for k, v in by_name.items() if "stage_rows" in k or "gather_rows" in k}
print(f"window {wall_ms:.0f} ms, GPU busy {gpu_busy_ms:.0f} ms ({100*gpu_busy_ms/wall_ms:.0f}%), cudaGraphLaunch x{graph_launches}")
for k, v in sorted(staging.items(), key=lambda kv: -kv[1])[:4]:
    print(f"  {k.split('/')[-1][:70]:70s} {v:8.1f} ms")
print(f"  cpu 'unique' ops {total('unique'):.1f} ms, 'index_select' {total('index_select'):.1f} ms, 'Memcpy HtoD' {total('Memcpy HtoD'):.1f} ms")
top = sorted(((e["name"], v) for e, v in ((e, by_name[e["name"]]) for e in kernels)), key=lambda kv: -kv[1])
seen, shown = set(), 0
print("  top GPU kernels:")
for name, v in top:
    if name in seen: continue
    seen.add(name); shown += 1
    print(f"    {v/1e3:8.1f} ms  {name[:90]}")
    if shown == 8: break
