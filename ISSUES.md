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

## P2-4 · sparse-attention: sparse+MTP crossover — MEASURED to 112K resident (GPU-busy win confirmed; wall win gated on P2-5)

**Evidence (post-P1-2 fix):** the group fix also unblocked multi-request
MTP on the 32 GB card (the old co-location failures were under the
65-group layout). Residency sweep, 16K/request, k=3, FULL capture
(`profile_decode.py --capture 4,8,16,32`):

| resident KV | stock wall / GPU busy | sparse wall / GPU busy |
|---|---|---|
| 1x16K | 21.2 / 22.0 ms | 22.6 / 22.5 ms |
| 2x16K | 21.4 / 22.5 | 22.2 / 22.2 |
| 4x16K | 22.6 / 23.8 | 23.2 / 22.8 |
| 6x16K | 24.4 / 25.8 | 24.6 / 24.0 |

GPU-busy crossover lands at ~64K resident and grows linearly (-1.0 ms at
64K, -1.8 ms at 96K): dense verify reads scale with resident KV, sparse
selection stays flat. Wall stays a tie because sparse is host-limited at
98-99% utilization (P2-5) while stock overlaps to 106%. Capacity win at
the card ceiling: sparse boots at kv=8.8 GiB and serves 7x16K residents
(983 tok/s decode-phase) vs stock's kv~8.2 cap and 6 residents
(~830-875 tok/s; stock's own workspaces OOM above that) — +12-18%
aggregate. Same-residency tok/s deltas ride acceptance noise
(+-0.5-1 tokens/step run-to-run); step time and GPU busy are the stable
metrics. Chrome traces of the 96K-resident pair: `traces/` (untracked).

**Remaining:** >128K resident needs P1-1 or a bigger GPU. On this card
attention is ~3 ms of a ~24 ms GEMM-dominated step, capping the possible
wall win at a few percent; the linear GPU savings vs fixed ~1-2 ms folded
tax (P2-2) project clear wins at DGX-Spark-scale residency, consistent
with the non-MTP 1.14x at 160K.

## P2-5 · sparse-attention: drafter-side build blocks on a seq_lens D2H sync

**Evidence:** instrumented residency-1 step: the 3 drafter-invoked
`vortex.build` calls block ~17 ms total inside `spec.propose`, waiting on
the busy stream. Verified source: `llm_base_proposer.py:685` nulls the
metadata's `_seq_lens_cpu` after adjusting device `seq_lens` in place
("Invalidate the CPU-side shadows to avoid H<>D sync"), so our capture
path's `m.seq_lens_cpu` access lazily runs `seq_lens.to("cpu")` — a
blocking D2H against the in-flight step, once per draft step. Below the
crossover it hides under GPU time (wall impact ~0); at >=96K resident it
is exactly what keeps P2-4's measured GPU-busy win (-1.8 ms) out of the
wall number (sparse pipelines at 98-99% vs stock 106%).

**Fix sketch:** stop touching `seq_lens_cpu` in the capture path when the
CPU shadow is invalidated — derive the drafter rows' host CSR from the
target step's host seq_lens plus the per-draft-step +1 offsets (the
drafter mutation is exactly `seq_lens += 1` per step, minus rejected
tokens available host-side), or keep a persistent pinned mirror updated
without sync. Est. converts 96K-resident wall to ~24.0 vs stock 24.4.

## P3 · sparse-attention: unported vortex features

`topk_ratio > 0` (variable budgets; eager+capture paths assume fixed hole),
custom `schedule_policy` strings (host-CSR path pins the default policy),
`Save`/`Load` cross-step state (needs prefix-cache interaction rules).
Port on demand.

## Upstream (vLLM) — fixed here, worth submitting

- `llm_base_proposer.py` reads `config.image_token_index`; Qwen3.5 configs
  name it `image_token_id` — breaks ALL MTP on Qwen3.5 VL checkpoints,
  plugin or not. Shipped as `patches/qwen3_5-mtp-config-fix/`.
