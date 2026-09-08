# mtp-pruning

Frequency-based draft-vocab pruning for Qwen3.5 MTP speculative decoding.

## Why

The MTP draft model shares the target's full `lm_head`. On a 248,320-id
vocab that is a 2.5 GB BF16 GEMV read **k+1 times per decode step**, and it
dominates speculative-decoding overhead on bandwidth-bound hardware (~26% of
decode wall clock on GB10 at k=3). Real agent traffic concentrates on a few
percent of the vocabulary — half of all generated tokens are one of just 78
ids — so the draft can propose from a K-row slice instead.

**Lossless by construction**: the target still verifies every draft with its
full head, so the output distribution is unchanged. A keep-set miss only
costs draft acceptance.

## Measured

GB10, Qwen3.8-27B-NVFP4, MTP k=3, 4k-token agent contexts, 42-request real
workload, concurrency 1/4/8; 99%-coverage keep-set = 7,696 of 248,320 ids:

| Metric | Full head | Pruned (99%) |
|---|---|---|
| Draft-head time | 45.0 ms/step | ~1 ms/step |
| Decode throughput c=1 / c=4 / c=8 | 15.3 / 42.5 / 57.2 tok/s | 20.6 (1.35×) / 54.4 (1.28×) / 75.0 (1.31×) |
| Acceptance length L | 3.034 (mean) | 3.004 (−1.0%, inside run noise ±1.1%) |

A 99.9% keep-set (11,201 ids) gains slightly less (1.10–1.28×) with the same
~1% acceptance cost.

## Mechanism

`patches/mtp-pruning/0001-mtp-draft-vocab-pruning.patch`, 2 files:

- `vllm/config/speculative.py` — adds `draft_vocab_path: str | None` to
  `SpeculativeConfig`. It rides the config broadcast, so it reaches every
  worker under any executor (an env var would not survive Ray's filter).
- `vllm/model_executor/models/qwen3_5_mtp.py` — when the field is set:
  - validates the keep-set file loudly (integer dtype, non-empty, strictly
    ascending, in `[0, vocab)`);
  - builds the draft `lm_head` as `ParallelLMHead(K)` +
    `LogitsProcessor(K)` and slices the checkpoint's head rows at load;
  - registers Eagle3/TorchSpec-style `d2t` offset buffers
    (`d2t[i] = target_id − i`), compatible with
    `use_local_argmax_reduction`;
  - sets `has_own_lm_head = True`, which routes **both** of vLLM's
    lm_head-sharing sites (V1 proposer and V2 gpu-worker loader) into a
    weight comparison that keeps the sliced head — without this the full
    head silently clobbers the slice and nothing is measured;
  - scatters draft logits back to full-vocab positions (`-inf` outside the
    keep-set) in `compute_logits`.

## Workflow

```bash
PY=/path/to/vllm-venv/bin/python

# 1) count token frequencies of YOUR model's generations (output side only —
#    that is the only distribution the draft head ever samples):
$PY scripts/mtp_pruning/count_tokens.py \
    --tokenizer /path/to/model --out counts.npy response_logs.jsonl

# 2) build the keep-set: smallest top-K reaching the coverage target,
#    unioned with the tokenizer's special ids, sorted, saved as int64 .pt
$PY scripts/mtp_pruning/build_keepset.py \
    --counts counts.npy --coverage 0.99 \
    --tokenizer /path/to/model --out keep_p99.pt

# 3) serve — activation is part of the MTP arguments:
vllm serve <model> --speculative-config \
    '{"method":"mtp","num_speculative_tokens":3,"draft_vocab_path":"/abs/path/keep_p99.pt"}'
```

`count_tokens.py` ingests, per line: OpenAI-style `{"messages": [...]}`
(counts every assistant message, content + tool-call name/args),
`{"text"|"content"|"response": ...}`, JSON strings, or raw text.
`--add-to prev.npy` accumulates across runs for workload drift.

### Verify in the serve log

```
MTP draft vocab pruned: 7696 of 248320 ids (3.10%), keep-set ...
Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head weights ...
```

Both lines are required. The second shows the sliced head survived vLLM's
lm_head-sharing; if it is missing you are silently benchmarking the
baseline.

## Picking coverage — and how many tokens to count

- **Counts need volume.** Frequencies over a ~250k-id vocabulary stabilize
  only after hundreds of thousands of generated tokens (Heaps' law): a
  keep-set fitted on a 50k-token sample covered just ~77% of a 541k-token
  corpus, versus 99% when fitted on all of it.
- 99% coverage (≈3% of vocab) was the throughput sweet spot
  in-distribution; small sample or drifting workload → prefer
  `--coverage 0.999`, or union a low-id BPE prefix (`--extra-ids`; BPE ids
  are merge-frequency ordered, so `[0, K)` transfers to unseen tasks far
  better than a small fitted top-K).
- Held-out caveat: a top-8k fitted set misses ~9% of tokens on tasks outside
  its fitting corpus, which costs roughly 15–20% of L. Re-count on your real
  traffic; do not ship someone else's keep-set.

## Benchmarking

`scripts/mtp_pruning/sweetspot.sh` sweeps (MTP off / on / pruned) ×
input-length × concurrency on controlled datasets, with a cold prefix cache
per run and both throughput views: `output_throughput_tps` (prefill in the
denominator) and `tpot_implied_decode_tps` (decode-only). Two dataset modes:

- `DATASET=random` (default): exact-length synthetic prompts — use for
  head-cost and prefill/decode scaling. Acceptance on random tokens is NOT
  representative (measured: L≈1.85 vs ≈3.0 on real agent traffic).
- `DATASET=sharegpt`: real conversations (`SHAREGPT_PATH`) — acceptance is
  meaningful for chat-style traffic, and stresses an agent-fitted keep-set
  out-of-distribution.

Requires `pandas` in the vLLM venv (`vllm[bench]`). For production numbers,
replay your own captured traffic instead.

Keep-sets are workload-specific — measured cross-coverage on GB10: an
agent-fitted 99% set (7,696 ids) covers only ~73% of ShareGPT chat
generations, while chat needs ~49k ids for 99% (general multilingual text
touches half the vocabulary). `count_tokens.py` reads ShareGPT records
natively (`conversations`/`from: gpt`), so recalibrating is: count the new
corpus, then `np.union1d` the keep-sets — a ~50k-id union covers 99%+ of
both at a still-5× head shrink.

## Qwen3.8-Flash-Next (Qwen4Exp MTP)

The same `draft_vocab_path` field drives the `Qwen4ExpMTP` drafter
(`vllm/models/qwen4_exp/nvidia/mtp.py`); the keep-set loading and logits
scatter now live in `vllm/model_executor/models/draft_vocab.py`, shared with
the Qwen3.5 drafters. Flash-Next's draft head is the full 248,320 × 2560
BF16 `lm_head` (1.2 GB) read once per draft step, k=3 → three reads per
decode step on a 273 GB/s part.

Keep-set for this box: `scripts/flash_next/build_keepset.sh RESULT_DIR OUT.pt`
counts the model's own generations (the JSON outputs of `bench_decode.py`
and `gsm8k_eval.py`), builds the 99 % keep-set and unions it with the
Qwen3.8-27B agent keep-set — the two models share the tokenizer
(`vocab.json` md5-identical). Fitted on 224 generations / 87k tokens:
4,173 ids for 99 %; union with the 7,696-id agent set = **10,010 ids
(4.0 % of vocab, head 1212 MB → 49 MB)**. The corpus is small (see
"Picking coverage" above); recount on real agent traffic before shipping.

Measured (GB10, ple-ssd 0001+0002, MTP k=3, 512-token answers to 16
coding/agent prompts, greedy, thinking off; `scripts/flash_next/results/`):

| | Full draft head | Pruned (10,010 ids) |
|---|---|---|
| Decode tok/s per request, c=1 | 34.1 | **41.8 (1.23×)** |
| Decode tok/s per request / aggregate, c=4 | 22.0 / 82.8 | 23.2 / 86.5 (1.05×) |
| TTFT median, c=1 | 0.36 s | 0.31 s |
| Acceptance length, c=1 / c=4 | 3.10 / 3.24 | 2.98 / 3.02 (−4 to −7 %) |
| GSM8K (200, thinking off) | 97.0 % | 97.0 % |

The c=1 gain is larger than on the 27B because Flash-Next's draft is
tiny (one layer, 10 of 512 experts) and the 1.2 GB head read was ~22 % of
every decode step (profiler, `docs/ple-ssd.md`). The acceptance cost is
higher than the 27B's −1 % because this keep-set was fitted on 87k tokens;
a larger in-distribution count should recover it.

## Limitations

- Qwen3.5 MTP architectures only (`Qwen3_5MTP`, `Qwen3_5MoeMTP`); the
  pattern generalizes to other MTP/Eagle drafts.
- Quantized `lm_head` checkpoints are refused loudly (companion scale
  tensors would need the same row slice). Excluded/BF16 heads — the
  ModelOpt default — work.
- `tie_word_embeddings` models are refused (input embedding must stay
  full-vocab).
- Benchmarked at TP=1 single-node; TP>1 is plumbed (`ParallelLMHead`
  shards, `LocalArgmaxMixin` maps offsets) but untested.

## Provenance

Full experiment record — corpus study, 3-arm × 3-concurrency sweep,
profiler evidence, reproduction harness with pass/fail tolerances — lives in
the magpie repo under `experiments/vocab-pruning/`.
