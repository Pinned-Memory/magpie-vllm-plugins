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
