#!/usr/bin/env python3
"""Torch-profile PURE DECODE steps at 16K context.

Drives the engine step-by-step: prefill 4x16K prompts to completion first,
then wraps torch.profiler around exactly 30 decode steps. Reports wall vs
GPU-busy (the launch-overhead question), and the kernel-time breakdown by
subsystem (GDN / attention / GEMM / vortex / other).
"""
import argparse, json, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TVM_FFI_GPU_BACKEND", "cuda")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FULL_ATTN_LAYERS = tuple(range(3, 64, 4))
MODEL = "RadixArk/Qwen3.8-27B-NVFP4"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["stock", "skip", "sparse"], required=True)
ap.add_argument("--compile", action="store_true", help="drop enforce_eager: piecewise compile + cudagraphs")
ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--nreq", type=int, default=4)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--kv-gib", type=float, default=0.0)
ap.add_argument("--mtp", type=int, default=0)
ap.add_argument("--util", type=float, default=0.0)
ap.add_argument("--capture", default="", help="comma-separated cudagraph capture sizes override (tokens, i.e. (1+mtp)*bs)")
ap.add_argument("--trace", default="", help="export a chrome trace of the profiled steps to this path (CUPTI needs a few hundred MB of free GPU memory or the trace comes back empty)")
args = ap.parse_args()

additional_config = {}
if args.mode != "stock":
    skip = list(FULL_ATTN_LAYERS) if args.mode == "skip" else []
    additional_config = {"vortex": {
        "topk_val": args.topk, "block_reserved_bos": 1,
        "block_reserved_eos": 2, "layers_skip": skip, "block_size": 64}}

import torch
from collections import Counter
import vllm.v1.cudagraph_dispatcher as _cgd
_DISPATCH_COUNTS = Counter()
_MISSES = Counter()
_orig_dispatch = _cgd.CudagraphDispatcher.dispatch
def _counting_dispatch(self, num_tokens, uniform_decode=False, **kw):
    mode, desc = _orig_dispatch(self, num_tokens, uniform_decode=uniform_decode, **kw)
    _DISPATCH_COUNTS[str(mode)] += 1
    if "NONE" in str(mode):
        _MISSES[(num_tokens, uniform_decode, str(kw.get("invalid_modes")))] += 1
    return mode, desc
_cgd.CudagraphDispatcher.dispatch = _counting_dispatch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_kw = dict(model=MODEL, max_model_len=20480, enforce_eager=not args.compile,
           gpu_memory_utilization=args.util or (0.80 if args.compile else 0.90),
           max_num_batched_tokens=2048, max_num_seqs=16,
           compilation_config={"cudagraph_capture_sizes":
               [int(x) for x in args.capture.split(",")] if args.capture
               else [(1 + args.mtp) * b for b in ((1, 2) if args.mtp else (1, 2, 4, 8, 10, 16))]}
           if args.compile else None,
           additional_config=additional_config,
           trust_remote_code=True)
if args.kv_gib:
    _kw["kv_cache_memory_bytes"] = int(args.kv_gib * (1 << 30))
if args.mtp:
    _kw["speculative_config"] = {"method": "mtp",
                                 "num_speculative_tokens": args.mtp}
llm = LLM(**_kw)
tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open("ruler16k.jsonl")][: args.nreq]
prompts = [tok.apply_chat_template([{"role": "user", "content": r["input"]}],
           tokenize=False, add_generation_prompt=True, enable_thinking=False)
           for r in rows]

eng = llm.llm_engine
sp = SamplingParams(temperature=0.0, max_tokens=4096, ignore_eos=True)
for i, p in enumerate(prompts):
    eng.add_request(str(i), p, sp)

# ---- burn through prefill + a few decode steps -----------------------------
# BOUNDED warm gate: if the pool can't admit all nreq requests, profile the
# resident batch instead of spinning forever (the 3h hang: scheduler admits
# fewer than nreq, len(outs)==nreq is unreachable, loop never exits).
warm, resident, stable = 0, 0, 0
for _i in range(3000):
    outs = eng.step()
    n = len(outs) if outs else 0
    if n and all(len(o.outputs[0].token_ids) >= 1 for o in outs):
        if n == resident:
            stable += 1
        else:
            resident, stable = n, 0
        if n == args.nreq or stable >= 40:   # full batch, or settled residency
            warm += 1
            if warm >= 8:
                break
else:
    raise RuntimeError("warm gate never settled after 3000 steps")
if resident and resident < args.nreq:
    print(f"NOTE: pool admits only {resident}/{args.nreq} requests; "
          f"profiling the resident batch")

# ---- profiled decode steps -------------------------------------------------
torch.cuda.synchronize()
_times = []
_tok0 = None
_tokN = 0
for _ in range(args.steps):
    _t = time.perf_counter()
    outs = eng.step()
    torch.cuda.synchronize()
    _times.append(time.perf_counter() - _t)
    _n = sum(len(o.outputs[0].token_ids) for o in outs) if outs else 0
    if _tok0 is None:
        _tok0 = _n
    _tokN = _n
unprofiled = sum(_times) / len(_times)
_s = sorted(_times)
print("dispatch modes:", dict(_DISPATCH_COUNTS))
print("NONE inputs (nreq, ntok, uniform, maxq):", dict(_MISSES))
_DISPATCH_COUNTS.clear()
if _tok0 is not None and _tokN > _tok0:
    acc = (_tokN - _tok0) / (len(_times) - 1)
    print(f"accepted tokens/step: {acc:.2f}  ->  "
          f"{acc / (sum(_times)/len(_times)):.1f} tok/s decode-phase")
print(f"step ms: min {_s[0]*1e3:.1f} p50 {_s[len(_s)//2]*1e3:.1f} "
      f"p90 {_s[int(len(_s)*.9)]*1e3:.1f} max {_s[-1]*1e3:.1f}")

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(args.steps):
        eng.step()
    torch.cuda.synchronize()

if args.trace:
    prof.export_chrome_trace(args.trace)
    print(f"chrome trace written: {args.trace}")
evs = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in evs) / 1e6 / args.steps
cpu_total = sum(e.self_cpu_time_total for e in evs) / 1e6 / args.steps
n_kernels = sum(e.count for e in evs if e.self_device_time_total > 0) / args.steps

def bucket(name):
    n = name.lower()
    if any(k in n for k in ("gated_delta", "conv1d", "recurrent", "mamba", "gdn")): return "GDN"
    if any(k in n for k in ("decode", "prefill", "attention", "flashinfer", "batchdecode")): return "attention"
    if any(k in n for k in ("gemm", "matmul", "cutlass", "nvfp4", "fp4", "mm_", "sm120")): return "GEMM/weights"
    if any(k in n for k in ("vortex", "fused_indexer", "plan_")): return "vortex"
    if any(k in n for k in ("cat", "copy", "elementwise", "vectorized", "reduce", "norm", "rope", "rotary", "silu", "mul", "add", "fill", "index", "softmax", "topk", "scatter", "gather", "triton")): return "elementwise/misc"
    return "other"

buckets = {}
for e in evs:
    if e.self_device_time_total > 0:
        b = bucket(e.key)
        buckets.setdefault(b, [0.0, 0])
        buckets[b][0] += e.self_device_time_total / 1e6 / args.steps
        buckets[b][1] += e.count / args.steps

print(f"\n==== {args.mode} topk={args.topk} nreq={args.nreq} ====")
print(f"wall/step (unprofiled): {unprofiled*1e3:8.1f} ms")
print(f"GPU busy/step:          {cuda_total*1e3:8.1f} ms   ({cuda_total/unprofiled*100:.0f}% of wall)")
print(f"CPU op time/step:       {cpu_total*1e3:8.1f} ms")
print(f"kernel launches/step:   {n_kernels:8.0f}")
print("\nGPU time by subsystem (ms/step, launches/step):")
for b, (t, c) in sorted(buckets.items(), key=lambda x: -x[1][0]):
    print(f"  {b:18s} {t*1e3:7.2f} ms  {c:6.0f} launches")
print("\nTop 12 kernels by GPU time (ms/step):")
for e in sorted(evs, key=lambda e: -e.self_device_time_total)[:12]:
    if e.self_device_time_total > 0:
        print(f"  {e.self_device_time_total/1e6/args.steps*1e3:7.2f} ms  x{e.count/args.steps:5.1f}  {e.key[:80]}")
