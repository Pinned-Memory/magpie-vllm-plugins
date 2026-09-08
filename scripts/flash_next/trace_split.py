#!/usr/bin/env python3
"""Worker vs. everything-else split of a vLLM worker (EngineCore) torch trace.

Per engine step (core.py _process_engine_step) the wall time is split into
  worker      = model_runner.execute_model + sample_tokens + get_output
                (CPU launch work + waits; GPU kernels run inside this span)
  scheduler   = scheduler.schedule + update_from_output
  engine I/O  = input-queue processing + zmq send of outputs
  other       = the remainder of the step loop
and inside the worker: GPU-busy time (union of kernel intervals), kernel
launch count, and kernel time by family.
  trace_split.py TRACE.json.gz [--label NAME]
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

CATS = [
    ("NVFP4 MoE grouped GEMM (routed experts)", r"GroupProblemShape|fp4|nvfp4|FP4"),
    ("MoE routing / permute / finalize", r"cutlass_kernels::|moe|topk_gating|router"),
    ("GEMV (lm_head)", r"gemvx|gemv"),
    ("BF16 GEMM (dense side layers)", r"cutlass.*gemm|cublas|gemm|wmma|nvjet"),
    ("HC gated-residual mix / norm", r"_hc_"),
    ("GDN linear attention", r"gdn|chunk_delta|chunk_gated|fused_recurrent|fla_|_fwd_kernel_h|causal_conv1d|solve_tril|chunk_o|chunk_h|chunk_|recompute_w_u|_fused_post_conv|wy_fast|cumsum_local"),
    ("QSA sparse attention + indexer", r"qsa|indexer|sparse_attn|persistent_topk|cooperative_topk|flash_fwd|flash_attn|mqa|attention|attn|paged"),
    ("PLE n-gram / embeddings / short conv", r"ple|short_conv|conv1d|embedding|index_select|gather|unique|scatter|sort|cub::|radix"),
    ("norms / activations / elementwise", r"rms|norm|silu|sigmoid|gelu|elementwise|vectorized|unrolled|fill|copy_|cat_|where|clamp|mul|add|reduce|softmax|argmax|gumbel|sample|rejection|cumsum|arange|index_put|triton_|hc_"),
    ("memcpy / memset", r"Memcpy|Memset|memcpy"),
]
ROLES = {
    "worker": r"model_runner\.py\(\d+\): (execute_model|sample_tokens)",
    "scheduler": r"scheduler\.py\(\d+\): (schedule|update_from_output)",
    "engine I/O": r"core\.py\(\d+\): (_process_input_queue|_send_msg_tracking_payload)",
}

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--label", default="")
a = ap.parse_args()
ev = json.load((gzip.open if a.trace.endswith(".gz") else open)(a.trace, "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]
py = [e for e in X if e.get("cat") == "python_function"]
steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in py if "_process_engine_step" in e["name"])
kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
kern.sort(key=lambda e: e["ts"])
launches = sum(1 for e in X if e.get("cat") == "cuda_runtime" and "cudaGraphLaunch" in e["name"])

def covered(lo, hi):
    tot, cs, ce = 0.0, None, None
    for e in kern:
        s, t = max(e["ts"], lo), min(e["ts"] + e["dur"], hi)
        if t <= s: continue
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, t
        else: ce = max(ce, t)
    if ce is not None: tot += ce - cs
    return tot

role_ms = defaultdict(float)
for e in py:
    for role, rx in ROLES.items():
        if re.search(rx, e["name"]):
            role_ms[role] += e["dur"]; break
wall = sum(hi - lo for lo, hi in steps)
n = len(steps)
gpu = sum(covered(lo, hi) for lo, hi in steps)
other = wall - sum(role_ms.values())
print(f"== {a.label or a.trace.split('/')[-3]}: {n} engine steps, {wall/1e3:.0f} ms of step loop")
print(f"{'role':28s} {'ms':>8s} {'%':>6s} {'ms/step':>8s}")
for role, ms in (("worker (execute+sample+output)", role_ms["worker"]), ("scheduler", role_ms["scheduler"]), ("engine I/O (zmq, input queue)", role_ms["engine I/O"]), ("other loop overhead", other)):
    print(f"{role:28s} {ms/1e3:8.1f} {100*ms/wall:6.1f} {ms/1e3/n:8.2f}")
print(f"\ninside the worker span: GPU busy {gpu/1e3:.0f} ms ({100*gpu/role_ms['worker']:.0f}% of worker, {100*gpu/wall:.0f}% of wall); worker CPU not covered by GPU work {(role_ms['worker']-gpu)/1e3:.0f} ms ({(role_ms['worker']-gpu)/1e3/n:.1f} ms/step)")
ksum = sum(e["dur"] for e in kern)
print(f"kernels: {len(kern)} launches ({len(kern)/n:.0f} per step), {ksum/1e3:.0f} ms summed ({ksum/1e3/n:.1f} ms/step), mean {ksum/len(kern):.1f} us each; cudaGraphLaunch calls: {launches} ({launches/n:.1f} per step)")
by = defaultdict(lambda: [0, 0.0])
for e in kern:
    cat = next((c for c, rx in CATS if re.search(rx, e["name"])), "other")
    by[cat][0] += 1; by[cat][1] += e["dur"]
print(f"{'kernel family':45s} {'count':>7s} {'ms':>8s} {'%GPU':>6s} {'ms/step':>8s}")
for cat, (c, ms) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"{cat:45s} {c:7d} {ms/1e3:8.0f} {100*ms/ksum:6.1f} {ms/1e3/n:8.1f}")
