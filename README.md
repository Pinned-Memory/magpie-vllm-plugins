# magpie-spark-vllm

The magpie team's vLLM optimizations for DGX Spark (GB10) serving. Each
optimization is a patch over the vLLM checkout plus plain supporting scripts
— no plugins, no entry points, no env vars, nothing to install: apply the
patch, drive the feature through vLLM's own arguments.

| Optimization | Patch | Activation |
|---|---|---|
| **mtp-pruning** | `patches/mtp-pruning/0001-mtp-draft-vocab-pruning.patch` | `draft_vocab_path` in `--speculative-config` |

## mtp-pruning

Frequency-based draft-vocab pruning for Qwen3.5 MTP speculative decoding.
The MTP draft shares the target's full `lm_head` — a 2.5 GB BF16 GEMV read
k+1 times per decode step on a 248,320-id vocab. Real agent traffic uses a
few percent of that vocabulary, so the draft proposes from a K-row slice;
the target still verifies with its full head, so the output distribution is
unchanged — a keep-set miss only costs draft acceptance.

Measured (GB10, Qwen3.8-27B-NVFP4, MTP k=3, 4k-token agent contexts,
99%-coverage keep-set of 7,696 ids): draft-head time **45 → ~1 ms/step**,
decode throughput **1.28–1.35×** at concurrency 1–8, acceptance length
**−1%** (within run noise).

### Setup

```bash
# once per vLLM checkout:
git -C /path/to/vllm apply patches/mtp-pruning/0001-mtp-draft-vocab-pruning.patch
```

### Workflow

```bash
# 1) count token frequencies of YOUR model's generations (output side only):
python scripts/mtp_pruning/count_tokens.py --tokenizer /path/to/model --out counts.npy response_logs.jsonl

# 2) build the keep-set (top-K to coverage, unioned with special tokens):
python scripts/mtp_pruning/build_keepset.py --counts counts.npy --coverage 0.99 \
    --tokenizer /path/to/model --out keep_p99.pt

# 3) serve — activation is part of the MTP arguments:
vllm serve <model> --speculative-config \
    '{"method":"mtp","num_speculative_tokens":3,"draft_vocab_path":"/abs/path/keep_p99.pt"}'
```

Verify in the serve log:

```
MTP draft vocab pruned: 7696 of 248320 ids (3.10%), keep-set ...
Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head weights ...
```

The second line shows vLLM's lm_head-sharing kept the sliced head instead of
clobbering it. Without `draft_vocab_path` the patched files behave stock.

### What the patch changes (2 files, ~125 lines)

- `vllm/config/speculative.py`: adds `draft_vocab_path: str | None` to
  `SpeculativeConfig` — it rides the config broadcast, so it reaches every
  worker under any executor (unlike an env var, which Ray does not forward).
- `vllm/model_executor/models/qwen3_5_mtp.py`: when the field is set, builds
  the draft `lm_head` as a validated K-row slice of the target head
  (`ParallelLMHead(K)` + `LogitsProcessor(K)`), registers Eagle3-style `d2t`
  offset buffers (compatible with `use_local_argmax_reduction`), sets
  `has_own_lm_head = True` so both of vLLM's lm_head-sharing sites keep the
  sliced head, slices the checkpoint rows at load, and scatters draft logits
  back to full-vocab positions in `compute_logits`.

### Notes

- Counts need volume: fit on hundreds of thousands of generated tokens
  (a 50k-token sample covered only ~77% of a 541k-token corpus). Small
  sample? Use `--coverage 0.999` or union a low-id BPE prefix
  (`--extra-ids`), and re-count as your workload drifts (`--add-to`).
- Quantized `lm_head` checkpoints are refused loudly (ModelOpt's default
  exclusion → BF16 head works). `tie_word_embeddings` models are refused.
- Benchmarked at TP=1; TP>1 is plumbed but untested.
- Full experiment record, reproduction harness, and the corpus study:
  `experiments/vocab-pruning/` in the magpie repo.
