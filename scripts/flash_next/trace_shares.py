#!/usr/bin/env python3
"""Time shares of a torch-profiler Chrome trace, bucketed by kernel family.

Usage: trace_shares.py TRACE.json.gz [--steps N]
Prints GPU time per category (% of GPU busy and % of wall), GPU idle share,
the top kernels, and per-step figures when --steps is given.
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

CATS = [
    ("NVFP4 MoE grouped GEMM (routed experts)", r"GroupProblemShape|fp4|nvfp4|FP4"),
    ("MoE routing / permute / finalize", r"cutlass_kernels::(finalizeMoeRouting|expandInputRows|blockExpertPrefixSum|buildExpertMaps|computeTotalRows|threeStepBuild|fusedBuildExpertMaps)|moe|topk_gating|softmax_topk|router"),
    ("GEMV (lm_head, M<=4)", r"gemvx|gemv"),
    ("BF16 GEMM (dense side layers)", r"cutlass.*gemm|cublas|Sgemm|Hgemm|gemm_bf16|wmma"),
    ("GDN linear attention", r"gdn|chunk_delta|chunk_gated|fused_recurrent|fla_|_fwd_kernel_h|causal_conv1d|solve_tril|chunk_o|chunk_h"),
    ("QSA sparse attention + indexer", r"qsa|indexer|sparse_attn|persistent_topk|cooperative_topk|flash_fwd|flash_attn|mqa|attention|attn|paged|fp8_kv|lightning"),
    ("PLE n-gram / embeddings / short conv", r"ple|short_conv|conv1d|embedding|index_select|gather|unique|scatter|sort|cub::|radix"),
    ("norms / activations / elementwise", r"rms|norm|silu|sigmoid|gelu|elementwise|vectorized|unrolled|fill|copy_|cat_|where|clamp|mul|add|reduce|softmax|argmax|gumbel|sample|rejection|cumsum|arange|index_put|triton_"),
    ("memcpy / memset", r"Memcpy|Memset|memcpy"),
]

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--steps", type=float, default=0)
a = ap.parse_args()
with (gzip.open if a.trace.endswith(".gz") else open)(a.trace, "rt") as f:
    ev = json.load(f)["traceEvents"]
dur = [e for e in ev if e.get("ph") == "X"]
t0 = min(e["ts"] for e in dur); t1 = max(e["ts"] + e["dur"] for e in dur)
wall = (t1 - t0) / 1e3
kern = [e for e in dur if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
by = defaultdict(float)
for e in kern:
    by[e["name"]] += e["dur"] / 1e3
busy = sum(by.values())
# GPU idle: union of kernel intervals on the GPU stream(s)
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in kern)
covered, cur_s, cur_e = 0.0, None, None
for s, e in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None: covered += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
if cur_e is not None: covered += cur_e - cur_s
covered /= 1e3
cat_ms = defaultdict(float); assigned = {}
for name, ms in by.items():
    for cat, rx in CATS:
        if re.search(rx, name):
            cat_ms[cat] += ms; assigned[name] = cat; break
    else:
        cat_ms["other"] += ms; assigned[name] = "other"
print(f"window {wall:.0f} ms | GPU covered {covered:.0f} ms ({100*covered/wall:.0f}% busy, {100*(1-covered/wall):.0f}% idle) | kernel sum {busy:.0f} ms")
if a.steps:
    print(f"per step ({a.steps:g} steps): wall {wall/a.steps:.1f} ms, GPU {covered/a.steps:.1f} ms, idle {(wall-covered)/a.steps:.1f} ms")
print(f"{'category':45s} {'ms':>8s} {'%GPU':>6s} {'%wall':>6s}" + ("   ms/step" if a.steps else ""))
for cat, ms in sorted(cat_ms.items(), key=lambda kv: -kv[1]):
    print(f"{cat:45s} {ms:8.0f} {100*ms/busy:6.1f} {100*ms/wall:6.1f}" + (f"   {ms/a.steps:6.1f}" if a.steps else ""))
print("top kernels:")
for name, ms in sorted(by.items(), key=lambda kv: -kv[1])[:14]:
    print(f"  {ms:7.0f} ms {100*ms/busy:5.1f}%  [{assigned[name][:22]:22s}] {name[:80]}")
