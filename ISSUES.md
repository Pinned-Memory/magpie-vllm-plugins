# Known issues & planned fixes

Working tracker for the catalog. Each entry: evidence (measured, with the
run that produced it), impact, and fix sketch. Ordered by priority.

## P1-1 · sparse-attention: aux memory invisible to vLLM's profiler

**Evidence:** centroid buffers (~110 MB at 8 GiB pools), the 128 MB
flashinfer workspace, and per-capture-size wrapper pools are allocated at
runtime, outside `determine_available_memory`. On permissive configs this
is silent; with MTP loaded on the 32 GB card it produced a chain of
failures (pool-alloc OOM at kv=6.0, cudnn-handle failure at 5.4 masking
stream-pool OOM, warmup OOM at 5.0/4.4/util-0.845) until the footprint
itself was shrunk. Forces hand-tuned `gpu_memory_utilization` /
`kv_cache_memory_bytes` per deployment.

**Fix:** Option A accounting — `AttentionBackend.customize_spec` adds the
per-block centroid bytes to `page_size_bytes` (page padding; in-tree
precedent: DeepSeek-V4 `state_content_bytes`), and the workspace moves to
vLLM's shared workspace helper. Removes every manual headroom knob and
unblocks P2-4.

## P1-2 · sparse-attention: 2x original `plan()` per decode step (~14 ms host)

**Evidence:** residency-1 MTP profile — vortex legs 38-39 ms wall vs
stock 21.2 with GPU busy only +3 ms (65% vs 104% utilization). Hidden at
batch >= 4 under GPU time; dominant at small batch. Root cause of keeping
the slow path: `fast_plan_decode` left stale replay state (RULER 0.125 vs
1.000 at identical config, isolated by A/B).

**Fix:** debug the fast path — diff what original `plan()` updates that
`fast_plan_decode` skips for OUR wrapper construction (suspect: per-wrapper
plan-info the replayed kernel reads that the fast path only refreshes for
vLLM's own buffer wiring). Acceptance test exists (the A/B that caught it).

## P2-1 · sparse-attention: draft acceptance -8% under sparse+MTP

**Evidence:** accepted tokens/step 3.41 (sparse) vs 3.69 (stock) vs 3.83
(vortex-dense) at k=3, 16K, greedy. Skip's 3.83 brackets run noise ~ +-0.15,
so the sparse dip is plausibly real. Mechanism: the draft imitates the
dense target; the sparse target's distribution drifts slightly.

**Fix directions:** monitor first (report acceptance in result records);
if it matters, verify-side options: larger topk on verify steps, or
shared-selection (P2-3) which also changes this.

## P2-2 · sparse-attention: folded dense wrapper slower than stock dense

**Evidence:** vortex-dense (skip) GPU 25.4 ms vs stock 22.0 at the same
workload (and 17.2 vs 16.1 in the non-MTP profile). The per-KV-head fold
(eff_bs rows x G=6 queries) underuses the kernel vs stock's layout.

**Fix:** layers in `layers_skip` (and the all-dense mode) should attend via
an unfolded stock-style wrapper; the fold is only needed where per-head
selection exists.

## P2-3 · sparse-attention: per-draft-token selection scales KV reads with k+1

**Evidence:** by construction — each verify token selects independently
(topk x (k+1) block reads, overlapping blocks re-read per row). Not yet the
bottleneck (P1-2 dominates), but it discounts the sparse margin under MTP
by up to (k+1)x at high occupancy.

**Fix:** optional shared selection across the k+1 positions (select with
the last token's query, or union of per-token holes) as a config knob —
trades a little verify fidelity for (k+1)x fewer selection reads.

## P2-4 · sparse-attention: above-crossover sparse+MTP throughput unmeasured

**Evidence:** every attempt to co-locate MTP (draft weights + ~3x mamba
state per request) with >= 4x16K resident KV and our unbudgeted aux on the
32 GB card failed on memory; the composed measurement exists only at
residency 1 (below the ~64K crossover). Blocked by P1-1; alternatively
measurable on a larger GPU. The non-MTP magnitude (1.14x at 160K resident)
plus P2-3's discount is the current best estimate.

## P3 · sparse-attention: unported vortex features

`topk_ratio > 0` (variable budgets; eager+capture paths assume fixed hole),
custom `schedule_policy` strings (host-CSR path pins the default policy),
`Save`/`Load` cross-step state (needs prefix-cache interaction rules).
Port on demand.

## Upstream (vLLM) — fixed here, worth submitting

- `llm_base_proposer.py` reads `config.image_token_index`; Qwen3.5 configs
  name it `image_token_id` — breaks ALL MTP on Qwen3.5 VL checkpoints,
  plugin or not. Shipped as `patches/qwen3_5-mtp-config-fix/`.
