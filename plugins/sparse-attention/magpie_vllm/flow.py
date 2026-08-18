"""Mean-K-centroid flow: the BlockTopK concept, production path only.

    forward_cache:   centroid[page] = Mean(k_view[page], dim=token)
    forward_indexer: one fused Triton launch -- gather + GQA-mean dot +
                     candidate mask + in-register top-k + hole write
                     (see kernels.py; 0.008 ms/layer, selections verified
                     against the eager reference in tests/test_fused_indexer)

fp8 note: the cache stores K / k_scale (per-tensor). A dot-product score is
scaled uniformly per row, so the top-k RANKING is scale-invariant -- no
dequant on the scoring path.

Capture contract: ``ensure()`` must run during the eager warmup pass so the
centroid buffers exist before graph capture records forward_indexer; a
capture-time ``cent is None`` bakes the indexer as a no-op.
"""

from __future__ import annotations

import torch

from .config import VortexConfig
from .kernels import fused_indexer


class VortexCentroidFlow:
    def __init__(self, cfg: VortexConfig):
        assert cfg.topk_ratio == 0.0, (
            "fixed hole size requires topk_ratio == 0")
        self.cfg = cfg
        self.centroids: dict[int, torch.Tensor] = {}   # layer_idx -> [P, D]

    def ensure(self, layer_idx: int, P: int, D: int, device) -> torch.Tensor:
        """Allocate centroid state OUTSIDE any capture (called every step
        from impl.forward; allocation happens in the eager warmup pass)."""
        cent = self.centroids.get(layer_idx)
        if cent is None or cent.shape[0] != P:
            cent = torch.zeros(P, D, dtype=torch.bfloat16, device=device)
            self.centroids[layer_idx] = cent
        return cent

    # ---- forward_cache: summarize pages completed this step ---------------
    def forward_cache(self, k_view: torch.Tensor, pages: torch.Tensor,
                      layer_idx: int) -> None:
        """k_view: (P, BLK, 1, D) folded view. pages: the worklist from the
        metadata builder -- fixed-shape (null-page redirected) on captured
        decode steps, exact delta on prefill/mixed steps."""
        if pages.numel() == 0:
            return
        cent = self.centroids.get(layer_idx)
        if cent is None or cent.shape[0] != k_view.shape[0]:
            cent = torch.zeros(k_view.shape[0], k_view.shape[-1],
                               dtype=torch.bfloat16, device=k_view.device)
            self.centroids[layer_idx] = cent
        blk = k_view[pages, :, 0, :]                      # [C, BLK, D]
        cent[pages] = blk.to(torch.float32).mean(1).to(torch.bfloat16)

    # ---- forward_indexer: one fused launch, writes the hole in place ------
    @torch.no_grad()
    def forward_indexer(self, q: torch.Tensor, layer_idx: int, md,
                        o: torch.Tensor) -> None:
        cent = self.centroids.get(layer_idx)
        if cent is None:
            return          # never taken after ensure(); kept as a guard
        fused_indexer(q.contiguous(), cent,
                      md.dense_kv_indptr, md.dense_kv_indices,
                      md.sparse_kv_indptr, o,
                      q.shape[0], int(md.max_block_len),
                      self.cfg.block_reserved_bos,
                      self.cfg.block_reserved_eos,
                      self.cfg.topk_val)
