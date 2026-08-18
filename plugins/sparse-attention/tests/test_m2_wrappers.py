"""M2 smoke test: planner -> two flashinfer wrappers -> in-place topK hole write.

Verifies, at Qwen3.8 dims (H_kv=4, G=6, D=256, vortex block 64, nbp=25):
  1. plan() ALIASES our sparse_kv_indices (data_ptr equality) -- the trick's
     load-bearing assumption;
  2. dense wrapper[0] output matches reference attention (per (req,head) row,
     folded layout, GQA group queries);
  3. full-budget sparse == dense (planner copied whole rows; topK early-return);
  4. small-topk sparse: eager topK writes the hole; wrapper[1] matches
     reference attention restricted to the selected pages.
"""
import sys, torch

from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from magpie_vllm.planner import plan_decode_sparse
from magpie_vllm.config import VortexConfig

torch.manual_seed(7)
dev = "cuda"
H, G, D = 4, 6, 256           # kv heads, gqa group, head dim
BLK, NBP = 64, 25
MGR = BLK * NBP               # 1600
R = 3
seq = [1700, 4096, 9000]      # spans 2..6 manager blocks
NB = 24                       # physical manager blocks
DT = torch.bfloat16

# ---- stock KV buffer + folded view ----------------------------------------
kv = torch.randn(NB, H, MGR, 2 * D, dtype=DT, device=dev) * 0.3
folded = kv.view(NB * H * NBP, BLK, 2 * D)
k_view = folded[:, :, :D].unsqueeze(2)     # (P, 64, 1, D)
v_view = folded[:, :, D:].unsqueeze(2)

# ---- block table (unique physical blocks per request) ----------------------
MB = max(-(-s // MGR) for s in seq)
perm = torch.randperm(NB)[: R * MB].view(R, MB).to(torch.int32).to(dev)
bt = perm.contiguous()
sl = torch.tensor(seq, dtype=torch.int32, device=dev)

# ---- metadata buffers ------------------------------------------------------
eff = R * H
maxblk = sum(-(-s // BLK) for s in seq) * H + 16
d_ip = torch.zeros(eff + 1, dtype=torch.int32, device=dev)
s_ip = torch.zeros(eff + 1, dtype=torch.int32, device=dev)
d_ix = torch.zeros(maxblk, dtype=torch.int32, device=dev)
s_ix = torch.zeros(maxblk, dtype=torch.int32, device=dev)
lpl = torch.ones(eff, dtype=torch.int32, device=dev)

ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
w_dense = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
w_sparse = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)

q = torch.randn(eff, G, D, dtype=DT, device=dev) * 0.3
scale = D ** -0.5


def run_plan(cfg):
    plan_decode_sparse(
        block_table=bt, seq_lens=sl, num_reqs=R, num_kv_heads=H,
        block_size=BLK, num_blocks_per_page=NBP, cfg=cfg,
        dense_kv_indptr=d_ip, dense_kv_indices=d_ix,
        sparse_kv_indptr=s_ip, sparse_kv_indices=s_ix,
        kv_last_page_len=lpl)
    common = dict(last_page_len=lpl[:eff], num_qo_heads=G, num_kv_heads=1,
                  head_dim=D, page_size=BLK, q_data_type=DT, kv_data_type=DT, sm_scale=scale)
    w_dense.plan(indptr=d_ip[: eff + 1], indices=d_ix, **common)
    w_sparse.plan(indptr=s_ip[: eff + 1], indices=s_ix, **common)


def ref_attention(row, page_list, n_tokens_last):
    """Reference: row's G queries over the given folded pages."""
    ks, vs = [], []
    for i, p in enumerate(page_list):
        n = BLK if i < len(page_list) - 1 else n_tokens_last
        ks.append(k_view[p, :n, 0, :])
        vs.append(v_view[p, :n, 0, :])
    K = torch.cat(ks).float()          # [T, D]
    V = torch.cat(vs).float()
    s = (q[row].float() @ K.T) * scale  # [G, T]
    return torch.softmax(s, dim=-1) @ V  # [G, D]


def check(out, rows_pages, tag, atol=2e-2):
    worst = 0.0
    for row in range(eff):
        pages, lastn = rows_pages[row]
        want = ref_attention(row, pages, lastn)
        got = out[row].float()
        err = (got - want).abs().max().item()
        worst = max(worst, err)
    assert worst < atol, f"{tag}: worst abs err {worst}"
    print(f"{tag}: PASS (worst abs err {worst:.4f})")


# ============================ test 1+2: dense ================================
cfg_full = VortexConfig(topk_val=4096, block_reserved_bos=1, block_reserved_eos=1)
run_plan(cfg_full)

assert w_sparse._paged_kv_indices_buf.data_ptr() == s_ix.data_ptr(), \
    "plan() COPIED the indices buffer -- the in-place trick is dead"
print("aliasing: PASS (wrapper._paged_kv_indices_buf IS our sparse_kv_indices)")

d_ip_h, d_ix_h, lpl_h = d_ip.cpu(), d_ix.cpu(), lpl.cpu()
dense_rows = {}
for row in range(eff):
    pages = d_ix_h[d_ip_h[row]: d_ip_h[row + 1]].tolist()
    dense_rows[row] = (pages, int(lpl_h[row]))

out_d = w_dense.run(q, (k_view, v_view))
check(out_d, dense_rows, "dense wrapper vs reference")

# ============================ test 3: full-budget sparse =====================
out_s = w_sparse.run(q, (k_view, v_view))
err = (out_s.float() - out_d.float()).abs().max().item()
assert err < 1e-3, f"full-budget sparse != dense ({err})"
print(f"full-budget sparse == dense: PASS (max diff {err:.5f})")

# ============================ test 4: real top-k =============================
cfg_k = VortexConfig(topk_val=8, block_reserved_bos=1, block_reserved_eos=2)
run_plan(cfg_k)

# eager topK: score = q-mean · mean-K centroid per candidate page, write hole
s_ip_h = s_ip.cpu()
sel_rows = {}
for row in range(eff):
    pages = dense_rows[row][0]
    bl = len(pages)
    budget = s_ip_h[row + 1] - s_ip_h[row]
    bos, eos = cfg_k.block_reserved_bos, cfg_k.block_reserved_eos
    if bl <= budget:
        sel_rows[row] = dense_rows[row]
        continue
    # candidates exclude BOS front and EOS tail positions (vortex topK range)
    cand = pages[bos: bl - eos]
    cent = k_view[torch.tensor(cand, device=dev).long(), :, 0, :].float().mean(1)
    qm = q[row].float().mean(0)
    score = cent @ qm
    k_hole = int(budget) - bos - eos
    top = score.topk(k_hole).indices
    hole_ids = [cand[i] for i in top.tolist()]
    # write the hole IN PLACE into the wrapper's buffer
    base = int(s_ip_h[row])
    w_sparse._paged_kv_indices_buf[base + bos: base + bos + k_hole] = \
        torch.tensor(hole_ids, dtype=torch.int32, device=dev)
    sel = pages[:bos] + hole_ids + pages[bl - eos:]
    sel_rows[row] = (sel, dense_rows[row][1])

out_k = w_sparse.run(q, (k_view, v_view))
check(out_k, sel_rows, "top-k sparse vs reference-on-selected")
print("ALL M2 TESTS PASS")
