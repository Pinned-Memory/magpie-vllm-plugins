#!/usr/bin/env python3
"""Boot Qwen3.8-27B with/without the vortex backend and generate.

Modes:
  stock  -- untouched vLLM (baseline)
  skip   -- vortex backend installed, ALL 16 full-attention layers in
            layers_skip -> every decode goes through the dense folded
            wrapper[0]; output must match stock (plumbing equivalence test)
  sparse -- vortex sparse decode with the centroid flow
"""
import argparse, json, os, sys, time

os.environ.setdefault("TVM_FFI_GPU_BACKEND", "cuda")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FULL_ATTN_LAYERS = tuple(range(3, 64, 4))          # [3, 7, ..., 63]
MODEL = "RadixArk/Qwen3.8-27B-NVFP4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stock", "skip", "sparse"], required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mtp", type=int, default=0, help="MTP speculative tokens")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--kv-gib", type=float, default=0.0,
                    help="explicit KV pool (kv_cache_memory_bytes)")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--bos", type=int, default=1)
    ap.add_argument("--eos", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--prompt-file", default=None, help="jsonl with 'input'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    additional_config = {}
    if args.mode != "stock":
        skip = list(FULL_ATTN_LAYERS) if args.mode == "skip" else []
        additional_config = {"vortex": {
            "topk_val": args.topk, "block_reserved_bos": args.bos,
            "block_reserved_eos": args.eos, "layers_skip": skip,
            "block_size": 64}}
        print(f"[vortex] plugin config mode={args.mode} topk={args.topk} "
              f"skip={len(skip)} layers (loads via vllm.general_plugins "
              f"in every worker)")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=MODEL, max_model_len=args.max_model_len,
              enforce_eager=not args.compile,
              gpu_memory_utilization=0.80 if args.compile else (0.78 if args.mtp else 0.90),
              max_num_batched_tokens=2048,
              max_num_seqs=16,
              compilation_config={"cudagraph_capture_sizes":
                  [(1 + args.mtp) * b for b in (1, 2, 4, 8)]}
              if args.compile else None,
              kv_cache_memory_bytes=int(args.kv_gib * (1 << 30))
              if args.kv_gib else None,
              additional_config=additional_config,
              speculative_config={"method": "mtp",
                                  "num_speculative_tokens": args.mtp}
              if args.mtp else None,
              trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL)

    if args.prompt_file:
        rows = [json.loads(l) for l in open(args.prompt_file)]
        if args.limit:
            rows = rows[: args.limit]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": r["input"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in rows]
        golds = [r["outputs"][0] for r in rows]
    else:
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content":
              "Count from 1 to 10, then name three prime numbers."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)]
        golds = None

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        ignore_eos=args.ignore_eos)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    results, hits, gen_toks = [], 0, 0
    for i, o in enumerate(outs):
        text = o.outputs[0].text
        gen_toks += len(o.outputs[0].token_ids)
        row = {"i": i, "text": text}
        if golds:
            row["gold"] = golds[i]
            row["hit"] = int(golds[i] in text)
            hits += row["hit"]
        results.append(row)

    summary = {"mode": args.mode, "topk": args.topk, "n": len(outs),
               "wall_s": round(dt, 2), "gen_tok_per_s": round(gen_toks / dt, 1)}
    if golds:
        summary["accuracy"] = round(hits / len(outs), 4)
    print("SUMMARY", json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=1)
    else:
        for r in results[:2]:
            print("---", repr(r["text"][:200]))


if __name__ == "__main__":
    main()
