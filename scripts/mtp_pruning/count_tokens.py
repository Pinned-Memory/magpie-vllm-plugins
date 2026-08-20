# SPDX-License-Identifier: Apache-2.0
"""Count token-id frequencies of model *generations* for keep-set building.

The draft head only ever proposes tokens the model generates, so the counts
that matter are the tokenization of your model's OUTPUT side — assistant
text and tool-call markup as your logs store it — not prompts.

    python scripts/mtp_pruning/count_tokens.py --tokenizer /path/to/model --out counts.npy \
        responses.jsonl more_logs.jsonl

Input files, decided per line:

  - JSON object with "messages": an OpenAI-style chat record — every
    assistant message is counted (its content plus a rendering of any
    tool_calls arguments);
  - JSON object with "conversations": a ShareGPT record — every
    "from": "gpt" turn is counted;
  - JSON object with "text" (or "content", or "response"): that string;
  - JSON string: the string itself;
  - anything else / non-JSON: the raw line is counted as text.

Then:

    python scripts/mtp_pruning/build_keepset.py --counts counts.npy --coverage 0.99 \
        --tokenizer /path/to/model --out keep.pt

Pass --add-to to accumulate into an existing counts file (e.g. nightly).
"""

import argparse
import json
import sys

import numpy as np

_TEXT_KEYS = ("text", "content", "response")


def _extract_texts(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return [line]
    if isinstance(rec, str):
        return [rec]
    if not isinstance(rec, dict):
        return [line]
    if "conversations" in rec and isinstance(rec["conversations"], list):
        # ShareGPT record: count only the assistant ("gpt") turns
        return [t["value"] for t in rec["conversations"]
                if isinstance(t, dict) and t.get("from") in ("gpt", "assistant")
                and t.get("value")]
    if "messages" in rec and isinstance(rec["messages"], list):
        out = []
        for m in rec["messages"]:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            c = m.get("content")
            if isinstance(c, str) and c:
                out.append(c)
            elif isinstance(c, list):  # multimodal-style parts
                out.extend(p.get("text", "") for p in c
                           if isinstance(p, dict) and p.get("text"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args)
                out.append(f"{name}({args})")
        return out
    for k in _TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v:
            return [v]
    return [line]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="jsonl / text files of generations")
    ap.add_argument("--tokenizer", required=True,
                    help="model or tokenizer dir / HF id")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--add-to", default=None,
                    help="existing counts .npy to accumulate into")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "transformers is required for counting (it is always present in "
            "a vLLM environment; run this tool from that environment)"
        )
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = max(len(tok), int(getattr(tok, "vocab_size", 0) or 0))
    # Prefer the model's (padded) vocab size when a config is present, so the
    # counts array lines up with the lm_head row count.
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.tokenizer)
        for c in (getattr(cfg, "text_config", None), cfg):
            v = getattr(c, "vocab_size", None) if c is not None else None
            if isinstance(v, int) and v > vocab:
                vocab = v
                break
    except Exception:
        pass

    counts = np.zeros(vocab, dtype=np.int64)
    if args.add_to:
        prev = np.load(args.add_to)
        if prev.ndim != 1:
            raise SystemExit(f"--add-to must be 1-D, got {prev.shape}")
        if prev.size > counts.size:
            counts = np.zeros(prev.size, dtype=np.int64)
        counts[: prev.size] += prev.astype(np.int64)

    n_texts = n_tokens = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal n_tokens
        if not buf:
            return
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        for ids in enc:
            a = np.asarray(ids, dtype=np.int64)
            if a.size:
                np.add.at(counts, a, 1)
                n_tokens += a.size
        buf.clear()

    for path in args.inputs:
        with open(path, errors="replace") as fh:
            for line in fh:
                for t in _extract_texts(line):
                    buf.append(t)
                    n_texts += 1
                    if len(buf) >= args.batch:
                        flush()
    flush()

    if n_tokens == 0:
        raise SystemExit("no tokens counted — check the input format")
    np.save(args.out, counts)
    nz = int((counts > 0).sum())
    print(f"{n_texts} generation texts -> {n_tokens:,} tokens, "
          f"{nz:,} distinct ids ({nz / vocab:.2%} of the {vocab:,}-id vocab) "
          f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
