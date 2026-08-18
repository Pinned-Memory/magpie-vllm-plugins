# sparse-attention

Vortex per-KV-head block-sparse decode attention for the Qwen3.5 family,
delivered as a pip-installed vLLM plugin — **zero vLLM lines patched**.

## Why

At long context, dense decode attention reads the entire KV cache every
step, per request. This plugin ports the vortex_torch sparse-attention
mechanism from sglang: per (request, kv-head) row, a per-64-token-block
mean-K centroid index selects the top-k blocks each step, and a FlashInfer
paged-decode wrapper attends over only those. Selection semantics are
identical to vortex-on-sglang (budget / BOS / reverse-EOS / hole contract,
per-KV-head fold), so accuracy measured there carries over.

## Measured

RTX 5090 (SM120, 32 GB), Qwen3.8-27B-NVFP4, fp8 KV, vLLM @ eee538d5da,
FULL CUDA-graph decode capture, RULER NIAH-uuid 16K (60 examples, greedy):

| config | accuracy | RULER wall | decode step p50 |
|---|---|---|---|
| stock | 1.000 | 104.4 s | 16.1 ms (32K resident) / 19.2 ms (160K) |
| sparse topk=8 (4.4% of ctx attended) | 0.983 | 108.3 s | 16.7 ms / **16.9 ms — 1.14× vs stock at 160K** |
| random selection at the same budget (control) | **0.000** | — | — |

Dense decode grows linearly with resident KV over a constant ~14 ms NVFP4
GEMM floor; sparse stays flat (crossover ≈ 64K resident tokens on this
card). Attention GPU time itself: 3.49 → 1.11 ms (3.1×). Indexer cost:
0.008 ms/layer (one fused Triton launch). The negative control (random
selection scores 0.000) is what certifies the 0.983 as real selection.

## Mechanism

`plugins/sparse-attention/` — installable package `magpie_vllm`:

- `plugin.py` — `vllm.general_plugins` entry point; loads in every vLLM
  process. Registers the backend under `AttentionBackendEnum.CUSTOM` and
  overrides the `Qwen3_5ForConditionalGeneration` arch mapping. **Inert
  without config**: no `{"vortex": ...}` in `--additional-config` means the
  model constructs bit-identically to stock.
- `model.py` — the registered model subclass. During construction it swaps
  `Qwen3NextAttention` for a subclass that injects
  `Attention(attn_backend=VortexFlashInferBackend)` (scoped, `finally`
  -restored). Only the 16 `full_attention` layers are touched; the 48 GDN
  layers are structurally unreachable.
- `backend.py` — backend (stock packed KV layout; the per-KV-head fold is a
  pure contiguous view), metadata builder (host CSR → two FlashInfer
  wrappers over persistent buffers → planner indices kernel; `UNIFORM_BATCH`
  cudagraph support with worst-case capture planning), impl (stock
  `reshape_and_cache_flash` + block-completion hook; in-place top-k hole
  write; FlashInfer paged dense prefill — prefill is never sparse).
- `planner.py` — vortex's decode planner on `block_table` addressing
  (BOS | **unwritten hole** | reverse-EOS), with a pure-python reference
  self-test (`python -m magpie_vllm.planner`).
- `kernels.py` — the fused indexer: centroid gather + GQA-mean dot +
  candidate mask + in-register top-k + hole write, one launch per layer.
- `flow.py` / `scatter.py` / `config.py` — centroid maintenance on
  newly-completed blocks; strict-keyed `VortexConfig`.

## Install & activate

```bash
uv pip install -e plugins/sparse-attention --python /path/to/vllm-venv/bin/python

vllm serve RadixArk/Qwen3.8-27B-NVFP4 \
  --additional-config '{"vortex": {"topk_val": 8, "block_reserved_bos": 1,
                                   "block_reserved_eos": 2, "block_size": 64}}'
```

Knobs mirror the vortex sglang submission JSON: `topk_val`, `topk_ratio`
(0 only, for now), `block_reserved_bos/eos` (`eos >= 1` is load-bearing),
`layers_skip`, `block_size` (64 max on SM120). Misspelled keys raise.

Validation harness: `scripts/sparse_attention/` — `make_ruler16k.py`
(dataset), `run_qwen38.py --mode {stock,skip,sparse} [--compile]`
(skip = all layers dense through the vortex plumbing; must match stock
exactly), `profile_decode.py` (pure-decode step profiler with cudagraph
dispatch-mode counter).

## Limits & interactions

- **Not composable with mtp-pruning yet.** Spec decode makes
  `decode_query_len = k+1`; the sparse decode path requires uniform
  single-token decode, so under MTP those rows fall to the dense path —
  correct output, sparsity inert. Spec-decode support
  (`supports_spec_as_decode`, per-draft-token planning) is the open item.
- Qwen3.5-family only (the model override targets
  `Qwen3_5ForConditionalGeneration`); the backend itself is model-agnostic
  GQA.
- `topk_ratio > 0`, custom `schedule_policy`, and vortex `Save`/`Load`
  cross-step state are not ported.
- Per-step wrapper `plan()` costs ~5 ms host; vLLM's `fast_plan_decode`
  was measured to leave stale replay state (0.125 vs 1.000) and is not used.
- fp8 KV supported (scale-invariant ranking); nvfp4 KV refused (different
  packing breaks the fold view).
