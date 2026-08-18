"""Fused vs eager indexer equivalence on real planner output."""
import sys, torch
from types import SimpleNamespace
from magpie_vllm.planner import plan_decode_sparse
from magpie_vllm.config import VortexConfig
from magpie_vllm.kernels import fused_indexer


def eager_reference(q, cent, d_ip, d_ix, s_ip, o, eff, maxblk, bos, eos, k_hole):
    """Plain-torch reference for the fused kernel (moved here from flow.py
    when the eager path was pruned from production)."""
    import torch as t
    row_len = (d_ip[1:eff+1] - d_ip[:eff]).long()
    budget = (s_ip[1:eff+1] - s_ip[:eff]).long()
    need = row_len > budget
    pos = t.arange(maxblk, device=q.device)
    idx = (d_ip[:eff, None].long() + pos[None, :]).clamp_(max=int(d_ip[eff]) - 1)
    pages = d_ix[idx].long()
    valid = pos[None, :] < row_len[:, None]
    cand = valid & (pos[None, :] >= bos) & (pos[None, :] < (row_len[:, None] - eos))
    qm = q.to(t.float32).mean(1)
    score = (cent[pages].to(t.float32) * qm[:, None, :]).sum(-1)
    score = score.masked_fill(~cand, float("-inf"))
    k = min(k_hole, maxblk)
    top = score.topk(k, dim=-1).indices
    hole = pages.gather(1, top).to(t.int32)
    dest = s_ip[:eff, None].long() + bos + t.arange(k, device=q.device)[None, :]
    mask = need[:, None].expand_as(dest)
    o[dest[mask]] = hole[mask]

torch.manual_seed(3); dev = "cuda"
H, G, D, BLK = 4, 6, 256, 64
R = 5
seq = [900, 4096, 16000, 200, 9999]           # short row 200 -> fits budget
cfg = VortexConfig(topk_val=8, block_reserved_bos=1, block_reserved_eos=2)
MB = max(-(-s_ // BLK) for s_ in seq)
bt = torch.randperm(4096, device=dev)[: R * MB].view(R, MB).to(torch.int32)
sl = torch.tensor(seq, dtype=torch.int32, device=dev)
eff = R * H
tot = sum(-(-s_ // BLK) for s_ in seq) * H + 8
d_ip = torch.zeros(eff + 1, dtype=torch.int32, device=dev)
s_ip = torch.zeros(eff + 1, dtype=torch.int32, device=dev)
d_ix = torch.zeros(tot, dtype=torch.int32, device=dev)
s_ix = torch.full((tot,), -7, dtype=torch.int32, device=dev)
lpl = torch.ones(eff, dtype=torch.int32, device=dev)
plan_decode_sparse(block_table=bt, seq_lens=sl, num_reqs=R, num_kv_heads=H,
                   block_size=BLK, num_blocks_per_page=1, cfg=cfg,
                   dense_kv_indptr=d_ip, dense_kv_indices=d_ix,
                   sparse_kv_indptr=s_ip, sparse_kv_indices=s_ix,
                   kv_last_page_len=lpl)

P = 4096 * H
cent = torch.randn(P, D, dtype=torch.bfloat16, device=dev)
q = torch.randn(eff, G, D, dtype=torch.bfloat16, device=dev)
maxblk = max(-(-s_ // BLK) for s_ in seq)

md = SimpleNamespace(dense_kv_indptr=d_ip, dense_kv_indices=d_ix,
                     sparse_kv_indptr=s_ip, max_block_len=maxblk)

# eager reference
o_eager = s_ix.clone()
eager_reference(q, cent, d_ip, d_ix, s_ip, o_eager, eff, maxblk,
                cfg.block_reserved_bos, cfg.block_reserved_eos, cfg.topk_val)

# fused
o_fused = s_ix.clone()
fused_indexer(q.contiguous(), cent, d_ip, d_ix, s_ip, o_fused,
              eff, maxblk, cfg.block_reserved_bos, cfg.block_reserved_eos,
              cfg.topk_val)
torch.cuda.synchronize()

s_ip_h, d_ip_h = s_ip.cpu(), d_ip.cpu()
oe, of = o_eager.cpu(), o_fused.cpu()
bos, eos = cfg.block_reserved_bos, cfg.block_reserved_eos
n_sel = 0
for r in range(eff):
    a, b = int(s_ip_h[r]), int(s_ip_h[r + 1])
    row_len = int(d_ip_h[r + 1] - d_ip_h[r])
    budget = b - a
    if row_len <= budget:
        assert torch.equal(oe[a:b], of[a:b]), f"row {r}: full-copy row disturbed"
        continue
    n_sel += 1
    hole_e = set(oe[a + bos: b - eos].tolist())
    hole_f = set(of[a + bos: b - eos].tolist())
    assert -7 not in hole_f, f"row {r}: fused left hole slots unwritten"
    assert hole_e == hole_f, f"row {r}: sets differ e={hole_e} f={hole_f}"
    # BOS/EOS untouched by both
    assert torch.equal(oe[a:a+bos], of[a:a+bos]) and torch.equal(oe[b-eos:b], of[b-eos:b])
print(f"fused == eager on {n_sel} selecting rows + full-copy rows PASS")

# timing
import time
for name, fn in [("eager", lambda o: eager_reference(q, cent, d_ip, d_ix, s_ip, o, eff,
                                                      maxblk, bos, eos, cfg.topk_val)),
                 ("fused", lambda o: fused_indexer(q.contiguous(), cent, d_ip, d_ix, s_ip, o,
                                                   eff, maxblk, bos, eos, cfg.topk_val))]:
    o = s_ix.clone(); fn(o)  # warm
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50): fn(o)
    torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter()-t0)/50*1e3:.3f} ms per layer-call")
