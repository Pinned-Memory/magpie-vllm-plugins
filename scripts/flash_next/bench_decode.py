#!/usr/bin/env python3
"""Decode-throughput benchmark over the chat API, from streaming timestamps.

Per request: TTFT and decode tok/s = (completion_tokens - 1) / (t_last - t_first).
Reports per-request mean/median, aggregate output tok/s, and the MTP
acceptance length from the server's /metrics counters (delta over the run).
Real coding/agent-style prompts, thinking off, greedy, no ignore_eos.
"""
import argparse
import asyncio
import json
import re
import statistics
import time

import aiohttp

PROMPTS = [
    "Write a Python module that implements an LRU cache with TTL expiry, thread safety, and unit tests. Explain the design choices in comments.",
    "Implement a bash script that finds duplicate files under a directory by content hash, handles filenames with spaces and newlines, and prints a report grouped by hash.",
    "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery) with a worked numerical example.",
    "Write a C function that parses an IPv4 CIDR string into network address and mask, with input validation, and a small test harness.",
    "Refactor this into idiomatic Rust and explain each change:\n\nint sum_even(int *a, int n){int s=0;for(int i=0;i<n;i++){if(a[i]%2==0)s+=a[i];}return s;}",
    "Describe a step-by-step plan for debugging a memory leak in a long-running Python service, including the tools you would use and what each reveals.",
    "Write a SQL schema for a small library system (books, authors, members, loans) and five queries covering overdue loans, most-borrowed authors, and member activity.",
    "Explain the difference between speculative decoding, multi-token prediction heads, and n-gram lookahead for LLM inference, with a comparison table.",
    "Implement Dijkstra's algorithm in Go with a binary heap, including a test with a small graph and commentary on complexity.",
    "Write a detailed code review for a pull request that adds retry logic with exponential backoff to an HTTP client; list concrete issues and suggested fixes.",
    "Give a tutorial on using git rebase interactively to squash, reorder, and edit commits, with example command output.",
    "Write a Python asyncio worker pool that pulls jobs from a queue, limits concurrency, retries failures, and shuts down gracefully on SIGTERM.",
    "Explain Kubernetes pod scheduling: taints, tolerations, affinity, resource requests and limits, with YAML examples.",
    "Implement a tiny JSON parser in JavaScript without using JSON.parse, supporting nested objects, arrays, strings with escapes, and numbers.",
    "Describe how to profile and optimize a slow pandas groupby-apply pipeline; show before/after code.",
    "Write a Makefile for a C++ project with debug/release targets, automatic dependency generation, and a test target.",
]
COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


async def one(session, url, model, prompt, args, sem):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    async with sem:
        t0 = time.perf_counter()
        t_first = t_last = None
        usage = None
        text = []
        async with session.post(f"{url}/v1/chat/completions", json=body) as r:
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                j = json.loads(line[5:])
                if j.get("usage"):
                    usage = j["usage"]
                if j.get("choices"):
                    delta = j["choices"][0]["delta"]
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        text.append(piece)
    n = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": n,
        "ttft_s": t_first - t0,
        "decode_tps": (n - 1) / (t_last - t_first),
        "total_s": t_last - t0,
        "text": "".join(text),
    }


async def metrics(session, url):
    async with session.get(f"{url}/metrics") as r:
        text = await r.text()
    out = {}
    for key in COUNTERS:
        m = re.search(rf"^{re.escape(key)}(?:\{{[^}}]*\}})? ([0-9.e+]+)", text, re.M)
        out[key] = float(m.group(1)) if m else 0.0
    return out


async def main(args):
    sem = asyncio.Semaphore(args.concurrency)
    reps = (args.num_prompts + len(PROMPTS) - 1) // len(PROMPTS)
    prompts = (PROMPTS * reps)[: args.num_prompts]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        await one(s, args.url, args.model, "Say hello in five languages.", args, sem)
        m0 = await metrics(s, args.url)
        t0 = time.perf_counter()
        res = await asyncio.gather(
            *[one(s, args.url, args.model, p, args, sem) for p in prompts]
        )
        wall = time.perf_counter() - t0
        m1 = await metrics(s, args.url)
    drafts = m1[COUNTERS[0]] - m0[COUNTERS[0]]
    accepted = m1[COUNTERS[2]] - m0[COUNTERS[2]]
    summary = {
        "concurrency": args.concurrency,
        "num_prompts": len(res),
        "think": args.think,
        "max_tokens": args.max_tokens,
        "mean_completion_tokens": statistics.mean(r["completion_tokens"] for r in res),
        "decode_tps_per_request_mean": statistics.mean(r["decode_tps"] for r in res),
        "decode_tps_per_request_median": statistics.median(r["decode_tps"] for r in res),
        "ttft_s_median": statistics.median(r["ttft_s"] for r in res),
        "aggregate_output_tps": sum(r["completion_tokens"] for r in res) / wall,
        "wall_s": wall,
        "mtp_acceptance_length": (1 + accepted / drafts) if drafts else None,
    }
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "requests": res}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default="bench.json")
    asyncio.run(main(ap.parse_args()))
