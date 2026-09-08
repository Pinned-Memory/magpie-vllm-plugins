#!/usr/bin/env python3
"""GSM8K accuracy against an OpenAI-compatible vLLM server (chat completions).

Zero-shot instruction, thinking off by default (--think turns it on), greedy.
The answer is the number after the last '#### ' (else the last number in the
reply). Writes a JSON summary plus per-question records.
"""
import argparse
import asyncio
import json
import re
import time

import aiohttp
import pandas as pd

GSM8K = (
    "/home/cc2869/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"
)
PROMPT = (
    "Solve this math problem step by step. At the very end, output ONLY the "
    "final numeric answer on a new line in the exact format:\n#### <number>\n\n"
    "Problem: {q}"
)


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def pred(text: str) -> str | None:
    hits = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if hits:
        return hits[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "") if nums else None


def same(a: str | None, b: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


async def ask(session, url, model, question, args, sem):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(q=question)}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    async with sem, session.post(f"{url}/v1/chat/completions", json=body) as r:
        j = await r.json()
    choice = j["choices"][0]
    return (
        choice["message"].get("content") or "",
        j["usage"]["completion_tokens"],
        choice["finish_reason"],
    )


async def main(args):
    df = pd.read_parquet(GSM8K).iloc[args.offset : args.offset + args.n]
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        outs = await asyncio.gather(
            *[ask(s, args.url, args.model, q, args, sem) for q in df.question]
        )
    wall = time.time() - t0
    recs, correct, toks = [], 0, 0
    for (q, a), (text, ntok, fin) in zip(zip(df.question, df.answer), outs):
        g, p = gold(a), pred(text)
        ok = same(p, g)
        correct += ok
        toks += ntok
        recs.append(
            {"gold": g, "pred": p, "correct": ok, "tokens": ntok, "finish": fin, "text": text}
        )
    summary = {
        "n": len(recs),
        "accuracy": correct / len(recs),
        "think": args.think,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "wall_s": wall,
        "completion_tokens": toks,
        "output_tok_per_s": toks / wall,
        "truncated": sum(r["finish"] == "length" for r in recs),
    }
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default="gsm8k.json")
    asyncio.run(main(ap.parse_args()))
