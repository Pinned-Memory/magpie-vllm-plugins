"""Summary-worklist derivation for the write path.

``newly_completed_pages`` answers: of the tokens this step just wrote, which
64-token vortex pages are now COMPLETE -- so ``forward_cache`` may summarize
each of them (``c["centroids"] = Mean(c["k"], dim=1)``) exactly once.

It is a DELTA, not a scan: each page crosses the completion threshold exactly
once in its content's lifetime and appears in exactly one step's worklist, so
cache maintenance is O(newly-completed) per step, never O(sequence). This is
the vLLM realisation of vortex's cache-kernel gate
``(token_position + 1) % BLOCK_SIZE == 0``.

WHEN A PAGE *CAN* REAPPEAR (both required for correctness, neither wasteful):
  * speculative decode rollback -- if the page-completing token was a rejected
    draft, the accepting step rewrites that slot and the gate re-fires,
    recomputing the summary over the corrected content. The first summary
    included a rejected token's K; the re-fire is what fixes it.
  * block recycling -- a freed physical block refilled by a new request
    completes again with new content, overwriting the stale summary.
Prefix-cache hits never re-fire (tokens are not rewritten), so a reused
block keeps the summary computed at its first fill -- correct, because a
centroid is a pure function of the block's own K.

CALL IT ONCE PER STEP, NOT PER LAYER. All 16 sparse layers share one KV-cache
group and therefore ONE slot_mapping tensor, so the worklist is identical
across layers. The metadata builder computes it in ``build()`` and stores it
on ``VortexDecodeMetadata.newly_completed_pages``; per-layer
``do_kv_cache_update`` just reads it. (The per-layer part that cannot be
shared is the summary math itself -- each layer summarizes its own K bytes.)
"""

from __future__ import annotations

import torch


def newly_completed_pages(
    slot_mapping: torch.Tensor,
    vortex_block_size: int,      # 64
    manager_block_size: int,     # 1600 on Qwen3.8; == vortex block on dense
    num_kv_heads: int,
) -> torch.Tensor:
    """Folded page ids of vortex pages COMPLETED by this step's writes.

    Completion is at VORTEX-page granularity (the summary unit) -- and because
    ``vortex_block | manager_block``, ``slot % 64`` equals
    ``(slot % 1600) % 64``, so the gate works on the raw slot:

        completed  <=>  slot >= 0  and  (slot + 1) % 64 == 0

    The pad guard comes FIRST: vLLM pads slot_mapping with -1, and
    ``-1 % 64 == 63`` under python-mod semantics would pass a naive test.

    A completing page is emitted once per KV head (the scatter writes all
    heads of a token together, so all heads' pages complete simultaneously):
    shape ``[num_completed * H_kv]`` int64, ready to index the folded
    ``k_view[page]`` or an aux buffer keyed the same way. Uniqueness is
    structural -- only one slot per page satisfies the gate, and slots are
    unique within a step.

    CAPTURE CAVEAT: the boolean-mask gather makes the output shape
    data-dependent -- fine for eager decode (milestone 3); a fixed-shape
    masked variant (junk rows scattered to the never-allocated null page 0)
    is needed before FULL-graph capture at milestone 4.
    """
    live = slot_mapping >= 0
    completing = live & ((slot_mapping + 1) % vortex_block_size == 0)
    slots = slot_mapping[completing]                    # [C]
    nbp = manager_block_size // vortex_block_size
    mgr = slots // manager_block_size
    sub = (slots % manager_block_size) // vortex_block_size
    heads = torch.arange(num_kv_heads, device=slots.device)
    # folded page = (mgr*H_kv + h)*nbp + sub, broadcast over heads
    pages = ((mgr[:, None] * num_kv_heads + heads[None, :]) * nbp
             + sub[:, None])
    return pages.reshape(-1)
