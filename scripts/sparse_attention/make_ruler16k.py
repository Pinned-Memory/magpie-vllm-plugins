#!/usr/bin/env python3
"""Regenerate the 16K-token RULER NIAH-uuid set used for validation.

Filler text comes from vortex_torch's examples/validation.jsonl (same essay
corpus, needles stripped); needle/question phrasing and the jsonl schema match
it exactly, so run_ruler-style substring scoring applies unchanged.
"""
import argparse, json, random, re, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="/home/cc2869/magpie/vortex_torch/examples/validation.jsonl")
ap.add_argument("--model", default="RadixArk/Qwen3.8-27B-NVFP4")
ap.add_argument("--out", default="ruler16k.jsonl")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--target-tokens", type=int, default=16000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)

rows = [json.loads(l) for l in open(args.source)]
needle_re = re.compile(
    r"One of the special magic uuids? for [^.]*\.|"
    r"A special magic uuid is hidden[^\n]*\n?|"
    r"Make sure to memorize it[^\n]*\n?|I will quiz you[^\n]*\n?")
fillers = []
for r in rows:
    body = r["input"]
    q = body.rfind("What is the special magic uuid")
    fillers.append(needle_re.sub("", body[:q] if q > 0 else body))
corpus = "\n".join(fillers)

words = ["amused-quart", "brave-otter", "calm-finch", "dizzy-maple",
         "eager-heron", "fuzzy-lemur", "gentle-aspen", "happy-crane",
         "icy-poplar", "jolly-mink"]
out = []
for i in range(args.n):
    uid = str(uuid.uuid4())
    name = f"{random.choice(words)}-{i}"
    header = ("A special magic uuid is hidden within the following text. "
              "Make sure to memorize it. I will quiz you about the uuid "
              "afterwards.\n")
    needle = f"\nOne of the special magic uuids for {name} is: {uid}.\n"
    question = (f"\nWhat is the special magic uuid for {name} "
                f"mentioned in the provided text?")
    lo, hi = 40000, 90000
    start = random.randrange(0, len(corpus) - hi - 1)
    depth = random.random()
    best = None
    for _ in range(8):
        mid = (lo + hi) // 2
        filler = corpus[start:start + mid]
        cut = int(len(filler) * depth)
        text = header + filler[:cut] + needle + filler[cut:] + question
        n = len(tok(text).input_ids)
        if n < args.target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
        best = (text, n)
    out.append({"index": i, "input": best[0], "outputs": [uid],
                "length": best[1]})

with open(args.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
lens = [r["length"] for r in out]
print(f"wrote {len(out)} examples, tokens "
      f"min/med/max: {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
