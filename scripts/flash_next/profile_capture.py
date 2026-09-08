#!/usr/bin/env python3
"""Capture torch-profiler windows on a live server: decode c=1, prefill of a
long fresh prompt, decode c=4. Traces land in TRACE_DIR/<window>/.
  profile_capture.py TRACE_DIR [--url URL]
"""
import argparse, glob, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
import requests

ap = argparse.ArgumentParser(); ap.add_argument("trace_dir"); ap.add_argument("--url", default="http://127.0.0.1:8100")
a = ap.parse_args(); U = a.url
M = "Qwen/Qwen3.8-Flash-Next"
def chat(prompt, n):
    return requests.post(f"{U}/v1/chat/completions", json={"model": M, "temperature": 0, "max_tokens": n,
        "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}).json()
def window(name, fn):
    before = set(glob.glob(f"{a.trace_dir}/*.gz"))
    requests.post(f"{U}/start_profile").raise_for_status(); t = time.time(); r = fn()
    requests.post(f"{U}/stop_profile").raise_for_status(); wall = time.time() - t
    for _ in range(60):
        new = set(glob.glob(f"{a.trace_dir}/*.gz")) - before
        if len(new) >= 2: break
        time.sleep(2)
    os.makedirs(f"{a.trace_dir}/{name}", exist_ok=True)
    for f in new: shutil.move(f, f"{a.trace_dir}/{name}/")
    print(f"{name}: {r} in {wall:.1f}s -> {len(new)} trace files")

base = "Explain how a B-tree index speeds up range queries, with an example."
chat(base, 64)  # warm
window("decode_c1", lambda: chat(base, 64)["usage"]["completion_tokens"])

# ~4k-token fresh prompt: paragraphs of distinct technical text
words = ("kernel scheduler latency throughput cache coherence pipeline register allocation branch predictor "
         "page table translation lookaside buffer interrupt handler memory barrier atomic compare exchange").split()
para = " ".join(f"{words[(i*7)%len(words)]} {words[(i*11)%len(words)]} {i}" for i in range(2600))
long_prompt = "Summarize the following notes in three bullet points:\n\n" + para
def prefill():
    r = chat(long_prompt, 1); return f"prompt_tokens={r['usage']['prompt_tokens']}"
window("prefill_4k", prefill)

prompts = [f"Write a {k} in Python with tests and explain the design." for k in ("rate limiter", "trie", "LRU cache", "topological sort")]
with ThreadPoolExecutor(4) as ex: list(ex.map(lambda p: chat(p, 8), prompts))  # warm prefixes
def c4():
    with ThreadPoolExecutor(4) as ex: rs = list(ex.map(lambda p: chat(p, 64), prompts))
    return sum(r["usage"]["completion_tokens"] for r in rs)
window("decode_c4", c4)
