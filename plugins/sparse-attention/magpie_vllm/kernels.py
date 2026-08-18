"""Fused indexer kernel (M4, part 1).

Replaces the eager python indexer's ~15 dispatched ops per layer per step with
ONE Triton launch that does, per (request, kv-head) row:

  1. q-mean over the GQA group (in-register)
  2. centroid gather via the dense CSR + dot-product scores (tiled over D)
  3. candidate masking [bos, row_len - eos)
  4. iterative top-k over the register-resident score row
  5. the hole write into sparse_kv_indices, at absolute CSR offsets

Rows where row_len <= budget exit immediately -- the planner copied the whole
dense row (the matched pair of the top-k early return, preserved exactly).

Scores stay in registers: MAXBLK_PAD = next_pow2(max_block_len) <= 4096, the
same per-row cap vortex's CUB kernel has. Selection ties may resolve in a
different order than torch.topk; the selected SET is identical up to ties.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_indexer_kernel(
    q_ptr,            # [eff, G, D] bf16, folded decode queries
    cent_ptr,         # [P, D] bf16 centroids
    d_ip, d_ix,       # dense CSR (int32)
    s_ip, s_ix,       # sparse CSR (int32); s_ix is the WRAPPER's buffer
    bos, eos, k_hole,
    G: tl.constexpr,
    D: tl.constexpr,
    DC: tl.constexpr,          # D-chunk width
    MAXBLK: tl.constexpr,      # padded max blocks per row (pow2, <= 4096)
):
    r = tl.program_id(0)
    row_start = tl.load(d_ip + r)
    row_len = tl.load(d_ip + r + 1) - row_start
    s_start = tl.load(s_ip + r)
    budget = tl.load(s_ip + r + 1) - s_start
    if row_len <= budget:
        return                               # planner copied the full row

    offs = tl.arange(0, MAXBLK)
    valid = offs < row_len
    pages = tl.load(d_ix + row_start + offs, mask=valid, other=0)

    # ---- scores: sum_dc  mean_g(q[r,g,dc]) . cent[page, dc] ----------------
    score = tl.zeros([MAXBLK], dtype=tl.float32)
    for dc0 in range(0, D, DC):
        dcols = dc0 + tl.arange(0, DC)
        qm = tl.zeros([DC], dtype=tl.float32)
        for g in range(G):
            qm += tl.load(q_ptr + (r * G + g) * D + dcols).to(tl.float32)
        qm = qm / G
        cent = tl.load(cent_ptr + pages[:, None] * D + dcols[None, :],
                       mask=valid[:, None], other=0.0).to(tl.float32)
        score += tl.sum(cent * qm[None, :], axis=1)

    # ---- candidate mask: [bos, row_len - eos) ------------------------------
    cand = valid & (offs >= bos) & (offs < row_len - eos)
    score = tl.where(cand, score, float("-inf"))

    # ---- iterative top-k, in-register; write holes directly ----------------
    for j in range(k_hole):
        m = tl.max(score, axis=0)
        idx = tl.argmax(score, axis=0)
        page = tl.load(d_ix + row_start + idx)
        tl.store(s_ix + s_start + bos + j, page)
        score = tl.where(offs == idx, float("-inf"), score)


def fused_indexer(q: torch.Tensor, cent: torch.Tensor,
                  d_ip: torch.Tensor, d_ix: torch.Tensor,
                  s_ip: torch.Tensor, s_ix: torch.Tensor,
                  eff: int, max_block_len: int,
                  bos: int, eos: int, k_hole: int) -> None:
    if eff == 0 or max_block_len == 0:
        return
    MAXBLK = max(triton.next_power_of_2(max_block_len), 16)
    assert MAXBLK <= 4096, "per-row block cap is 4096 (matches vortex)"
    G, D = q.shape[1], q.shape[2]
    _fused_indexer_kernel[(eff,)](
        q, cent, d_ip, d_ix, s_ip, s_ix,
        bos, eos, k_hole,
        G=G, D=D, DC=64, MAXBLK=MAXBLK,
        num_warps=8 if MAXBLK >= 1024 else 4,
    )
