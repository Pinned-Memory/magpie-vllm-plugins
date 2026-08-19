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

## P1-2 · sparse-attention: RESOLVED — 65-way KV-cache group explosion under MTP (was misfiled as "2x `plan()` host cost")

**Root cause (measured, instrumented):** the residency-1 MTP wall gap
(vortex 38-39 ms vs stock 21.2, GPU 65% idle-bubbled) was NOT `plan()`:
the two per-step `plan()` calls measure 0.14 ms combined and do not block
against a busy stream. The real cost: vLLM buckets KV-cache layers by
spec equality, and `indexes_kv_by_block_stride` (derived from the
attention backend) is part of the spec. The target's 16 vortex layers say
False; the MTP draft's stock-flashinfer full-attn layer says True → a
lone 1-layer bucket → hybrid group size = min(bucket sizes) = 1 → **65
single-layer KV-cache groups** → 65 metadata builds per step (48x
gdn.build 9.3 ms + 16x vortex.build 6.0 ms, plus per-group scheduler/
block-table work). Non-MTP runs never hit this (16-layer bucket → 4
groups), which is why it only surfaced with MTP.

**Fix (landed):** vortex overrides for `Qwen3_5MTP` / `Qwen3_5MoeMTP`
construct the draft's full-attention layer with the vortex backend in
forced-dense mode (`vortex_force_dense=True`; the dense folded wrapper is
numerically stock, M2-verified). All 17 full-attn specs match → 4 groups,
stock shape. Measured after (same card, 1x16K, FULL capture, k=3):
sparse+MTP 38.3 → **22.6 ms/step** wall, GPU busy 22.5 (99% pipelined,
was 65%); decode-phase 89 → 152.6 tok/s; RULER 16K **1.000**; acceptance
3.45. Stock at 21.2/22.0 → residual gap ~1.4 ms wall ≈ P2-2's folded
machinery. vortex-dense (skip) similarly 39.1 → 23.0.

**fast_plan_decode post-mortem** (the earlier suspect, now explained):
its stale replay state (RULER 0.125) came from the cudagraph branch
skipping the H2D refresh of the wrapper's device indptr/last_page_len
buffers — stock vLLM maintains those buffers itself; our capture path
takes them from `plan()`'s copy, and both the indices kernel and the
replayed kernel read them. If per-step `plan()` ever matters (it does
not at 2 calls/step): pinned-staging H2D refresh of both indptrs + lpl,
then `flashinfer.decode.fast_decode_plan`.

**Upstream note:** one odd-spec layer collapsing the hybrid group size to
1 (65 groups, ~15 ms/step host at batch 1) is a vLLM sharp edge worth
reporting — grouping could tolerate specs differing only in
`indexes_kv_by_block_stride`, or bound the group count.

## P2-1 · sparse-attention: draft acceptance -8% under sparse+MTP

**Evidence:** accepted tokens/step 3.41 (sparse) vs 3.69 (stock) vs 3.83
(vortex-dense) at k=3, 16K, greedy. Skip's 3.83 brackets run noise ~ +-0.15,
so the sparse dip is plausibly real. Mechanism: the draft imitates the
dense target; the sparse target's distribution drifts slightly.
*Post-P1-2-fix re-measurement:* 3.45 (sparse) vs 3.54 (stock) vs 3.31
(vortex-dense) — the ordering inverted between runs, so the dip is within
run noise at this operating point. Keep monitoring; do not spend on it yet.

**Fix directions:** monitor first (report acceptance in result records);
if it matters, verify-side options: larger topk on verify steps, or
shared-selection (P2-3) which also changes this.

## P2-2 · sparse-attention: folded dense wrapper slower than stock dense

**Evidence:** post-P1-2-fix, vortex-dense (skip) 23.0 ms wall / 22.9 GPU
vs stock 21.2 / 22.0 at the same MTP workload (and 17.2 vs 16.1 GPU in the
non-MTP profile) — a ~0.9-1.8 ms folded-machinery tax. The per-KV-head
fold (eff_bs rows x G=6 queries) underuses the kernel vs stock's layout.

**Fix:** layers in `layers_skip` (and the all-dense mode) should attend via
an unfolded stock-style wrapper; the fold is only needed where per-head
selection exists.

## P2-3 · sparse-attention: per-draft-token selection scales KV reads with k+1

**Evidence:** by construction — each verify token selects independently
(topk x (k+1) block reads, overlapping blocks re-read per row). Not the
bottleneck at residency 1 (sparse+MTP already beats vortex-dense there),
but it discounts the sparse margin under MTP by up to (k+1)x at high
occupancy.

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
