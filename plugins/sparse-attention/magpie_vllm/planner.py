"""The vortex decode planner, ported from sglang ``req_to_token`` addressing to
vLLM ``block_table`` addressing.

Faithful port of ``vortex_torch/indexer/planner_sglang.py`` kernels (a) indptr
and (c) indices -- same launch shapes, same BOS / reverse-EOS / hole contract.
Kernel (b), the ``winfo_*`` workload schedule, is NOT ported yet: it only
feeds vortex's compiled Schedule.W Triton kernels, and the eager reference
indexer scores via ``block_table`` directly. Port it when the fused indexer
kernels come over.

Addressing (the one change; see backend.py get_kv_cache_shape):

  sglang:  slot = req_to_token[req][pos * block_size]
           page = (slot / page_size) * H_kv + h
           blk  = page * nbp + (slot % page_size) / block_size
  vLLM:    mgr  = block_table[req][pos / nbp]
           blk  = (mgr * H_kv + h) * nbp + pos % nbp

with nbp = manager_block // vortex_block (25 on Qwen3.8-27B, 1 on dense).

CONTRACT (must never change silently):
  * sparse row = [BOS blocks | UNWRITTEN HOLE | EOS blocks reverse-written]
    -- the hole [bos, budget-eos) is the terminal topK's exclusive output
    region; initialising it would let a flow that forgets topK attend to
    stale-but-plausible ids instead of failing loudly.
  * rows with block_len <= budget get the WHOLE dense row copied -- the
    matched pair of topK's early return.
  * EOS is reverse-written so slot budget-1 always holds the true last block,
    which is what lets both wrappers share one kv_last_page_len.
"""

from __future__ import annotations

import functools
import hashlib

import torch
from torch.utils.cpp_extension import load_inline

from .config import VortexConfig

_DEFAULT_POLICY = """
    return max(topk_val + block_reserved_bos + block_reserved_eos,
               (int)(cached_block_len * topk_ratio));
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ int ComputeKvBudget(
    const int cached_block_len,
    const int topk_val,
    const float topk_ratio,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
__POLICY_BODY__
}

struct Int2Sum {
    __device__ int2 operator()(const int2 &a, const int2 &b) const {
        return make_int2(a.x + b.x, a.y + b.y);
    }
};

// ---------------------------------------------------------------------------
// (a) indptr -- one thread per request, both prefix sums in one CUB pass,
//     rows replicated per KV head: row r = req*H_kv + h.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) Vllm_Plan_Indptr_Kernel(
    const int* __restrict__ seq_lens,
    int* __restrict__ dense_kv_indptr,
    int* __restrict__ sparse_kv_indptr,
    const int num_reqs,
    const int num_kv_heads,
    const int block_size,
    const int topk_val,
    const float topk_ratio,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
    typedef cub::BlockScan<int2, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp;

    const int tx = threadIdx.x;
    int cached_block_len = 0, sparse_len = 0;
    if (tx < num_reqs) {
        const int kv_len = seq_lens[tx];
        cached_block_len = (kv_len + block_size - 1) / block_size;
        const int kv_budget = ComputeKvBudget(
            cached_block_len, topk_val, topk_ratio,
            block_reserved_bos, block_reserved_eos);
        sparse_len = min(kv_budget, cached_block_len);
    }
    int2 v = make_int2(cached_block_len, sparse_len);
    int2 agg;
    BlockScan(temp).ExclusiveScan(v, v, make_int2(0, 0), Int2Sum(), agg);

    if (tx < num_reqs) {
        for (int h = 0; h < num_kv_heads; ++h) {
            const int row = tx * num_kv_heads + h;
            dense_kv_indptr[row]  = v.x * num_kv_heads + h * cached_block_len;
            sparse_kv_indptr[row] = v.y * num_kv_heads + h * sparse_len;
        }
    }
    if (tx == 0) {
        dense_kv_indptr[num_reqs * num_kv_heads]  = agg.x * num_kv_heads;
        sparse_kv_indptr[num_reqs * num_kv_heads] = agg.y * num_kv_heads;
    }
}

// ---------------------------------------------------------------------------
// (c) indices -- grid (num_reqs, num_kv_heads), 512 threads.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(512) Vllm_Plan_Indices_Kernel(
    const int* __restrict__ block_table,   // [num_reqs, bt_stride] MANAGER blocks
    const int* __restrict__ seq_lens,
    const int* __restrict__ dense_kv_indptr,
    const int* __restrict__ sparse_kv_indptr,
    int* __restrict__ dense_kv_indices,
    int* __restrict__ sparse_kv_indices,
    int* __restrict__ kv_last_page_len,
    const int bt_stride,
    const int block_size,
    const int num_blocks_per_page,
    const int num_kv_heads,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
    const int nthreads = blockDim.x;
    const int bx = blockIdx.x;                 // request
    const int by = blockIdx.y;                 // kv head
    const int tx = threadIdx.x;
    const int row = bx * num_kv_heads + by;

    const int* bt = block_table + (long)bx * bt_stride;

    const int kv_len    = seq_lens[bx];
    const int block_len = (kv_len + block_size - 1) / block_size;
    const int last_len  = kv_len % block_size;

    if (tx == 0) {
        kv_last_page_len[row] = (last_len == 0) ? block_size : last_len;
    }

    int* dense_output  = dense_kv_indices  + dense_kv_indptr[row];
    int* sparse_output = sparse_kv_indices + sparse_kv_indptr[row];
    const int kv_budget = sparse_kv_indptr[row + 1] - sparse_kv_indptr[row];

    for (int pos = tx; pos < block_len; pos += nthreads) {
        const int mgr = bt[pos / num_blocks_per_page];
        const int sub = pos % num_blocks_per_page;
        dense_output[pos] =
            (mgr * num_kv_heads + by) * num_blocks_per_page + sub;
    }

    if (block_len <= kv_budget) {
        for (int p = tx; p < block_len; p += nthreads) {
            sparse_output[p] = dense_output[p];
        }
    } else {
        for (int p = tx; p < block_reserved_bos; p += nthreads) {
            const int mgr = bt[p / num_blocks_per_page];
            sparse_output[p] =
                (mgr * num_kv_heads + by) * num_blocks_per_page
                + p % num_blocks_per_page;
        }
        for (int p = tx; p < block_reserved_eos; p += nthreads) {
            const int pos = block_len - p - 1;
            const int mgr = bt[pos / num_blocks_per_page];
            sparse_output[kv_budget - p - 1] =
                (mgr * num_kv_heads + by) * num_blocks_per_page
                + pos % num_blocks_per_page;
        }
        // middle [bos, kv_budget-eos) DELIBERATELY UNWRITTEN.
    }
}

void vortex_plan_indices(
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    at::Tensor& dense_kv_indices,
    at::Tensor& sparse_kv_indices,
    at::Tensor& kv_last_page_len,
    const int64_t num_reqs,
    const int64_t num_kv_heads,
    const int64_t block_size,
    const int64_t num_blocks_per_page,
    const int64_t block_reserved_bos,
    const int64_t block_reserved_eos
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)num_reqs, (unsigned)num_kv_heads);
    Vllm_Plan_Indices_Kernel<<<grid, 512, 0, stream>>>(
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        (int)block_table.stride(0),
        (int)block_size, (int)num_blocks_per_page, (int)num_kv_heads,
        (int)block_reserved_bos, (int)block_reserved_eos);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "indices launch failed: ",
                cudaGetErrorString(err));
}

void vortex_plan_decode(
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& dense_kv_indptr,
    at::Tensor& sparse_kv_indptr,
    at::Tensor& dense_kv_indices,
    at::Tensor& sparse_kv_indices,
    at::Tensor& kv_last_page_len,
    const int64_t num_reqs,
    const int64_t num_kv_heads,
    const int64_t block_size,
    const int64_t num_blocks_per_page,
    const int64_t topk_val,
    const double topk_ratio,
    const int64_t block_reserved_bos,
    const int64_t block_reserved_eos
) {
    TORCH_CHECK(num_reqs >= 1 && num_reqs <= 1024, "num_reqs must be in [1, 1024]");
    TORCH_CHECK(topk_val >= 1);
    TORCH_CHECK(block_reserved_bos >= 0);
    TORCH_CHECK(block_reserved_eos >= 1,
        "eos >= 1 is load-bearing: the trailing partial block has no summary");
    TORCH_CHECK(topk_ratio >= 0.0 && topk_ratio <= 1.0);
    TORCH_CHECK(block_table.dtype() == at::kInt);
    TORCH_CHECK(seq_lens.dtype() == at::kInt);
    TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Vllm_Plan_Indptr_Kernel<<<1, 1024, 0, stream>>>(
        seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        (int)num_reqs, (int)num_kv_heads, (int)block_size,
        (int)topk_val, (float)topk_ratio,
        (int)block_reserved_bos, (int)block_reserved_eos);

    dim3 grid((unsigned)num_reqs, (unsigned)num_kv_heads);
    Vllm_Plan_Indices_Kernel<<<grid, 512, 0, stream>>>(
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        (int)block_table.stride(0),
        (int)block_size, (int)num_blocks_per_page, (int)num_kv_heads,
        (int)block_reserved_bos, (int)block_reserved_eos);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "planner launch failed: ",
                cudaGetErrorString(err));
}
"""


@functools.lru_cache(maxsize=8)
def _build(policy_body: str):
    src = _CUDA_SRC.replace("__POLICY_BODY__", policy_body)
    tag = hashlib.sha256(src.encode()).hexdigest()[:12]
    decl = """
void vortex_plan_decode(
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& dense_kv_indptr, at::Tensor& sparse_kv_indptr,
    at::Tensor& dense_kv_indices, at::Tensor& sparse_kv_indices,
    at::Tensor& kv_last_page_len,
    const int64_t num_reqs, const int64_t num_kv_heads,
    const int64_t block_size, const int64_t num_blocks_per_page,
    const int64_t topk_val, const double topk_ratio,
    const int64_t block_reserved_bos, const int64_t block_reserved_eos);
void vortex_plan_indices(
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    const at::Tensor& dense_kv_indptr, const at::Tensor& sparse_kv_indptr,
    at::Tensor& dense_kv_indices, at::Tensor& sparse_kv_indices,
    at::Tensor& kv_last_page_len,
    const int64_t num_reqs, const int64_t num_kv_heads,
    const int64_t block_size, const int64_t num_blocks_per_page,
    const int64_t block_reserved_bos, const int64_t block_reserved_eos);
"""
    return load_inline(
        name=f"magpie_vllm_planner_{tag}",
        cpp_sources=decl,
        cuda_sources=src,
        functions=["vortex_plan_decode", "vortex_plan_indices"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def plan_decode_sparse(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    num_kv_heads: int,
    block_size: int,
    num_blocks_per_page: int,
    cfg: VortexConfig,
    dense_kv_indptr: torch.Tensor,
    dense_kv_indices: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    **_unused_winfo,
) -> None:
    """Launch the planner. Writes everything in place; nothing allocated."""
    mod = _build(cfg.schedule_policy or _DEFAULT_POLICY)
    mod.vortex_plan_decode(
        block_table, seq_lens.to(torch.int32),
        dense_kv_indptr, sparse_kv_indptr,
        dense_kv_indices, sparse_kv_indices,
        kv_last_page_len,
        num_reqs, num_kv_heads, block_size, num_blocks_per_page,
        cfg.topk_val, cfg.topk_ratio,
        cfg.block_reserved_bos, cfg.block_reserved_eos,
    )


def launch_indices_kernel(*, block_table, seq_lens, num_reqs, num_kv_heads,
                          block_size, num_blocks_per_page, cfg,
                          dense_kv_indptr, dense_kv_indices,
                          sparse_kv_indptr, sparse_kv_indices,
                          kv_last_page_len):
    """Indices kernel only -- the capture path computes indptr on the host and
    lets the wrappers' plan() copy it into the device buffers first."""
    mod = _build(cfg.schedule_policy or _DEFAULT_POLICY)
    mod.vortex_plan_indices(
        block_table, seq_lens.to(torch.int32),
        dense_kv_indptr, sparse_kv_indptr,
        dense_kv_indices, sparse_kv_indices, kv_last_page_len,
        num_reqs, num_kv_heads, block_size, num_blocks_per_page,
        cfg.block_reserved_bos, cfg.block_reserved_eos)


# ---------------------------------------------------------------------------
# Pure-python reference, for the unit test.
# ---------------------------------------------------------------------------
def _reference(block_table, seq_lens, H, BLK, NBP, topk_val, ratio, bos, eos):
    R = len(seq_lens)
    d_indptr, s_indptr = [0], [0]
    lens, budgets = [], []
    for r in range(R):
        bl = -(-seq_lens[r] // BLK)
        budget = max(topk_val + bos + eos, int(bl * ratio))
        lens.append(bl); budgets.append(min(budget, bl))
    for r in range(R):
        for h in range(H):
            d_indptr.append(d_indptr[-1] + lens[r])
            s_indptr.append(s_indptr[-1] + budgets[r])
    dense = [0] * d_indptr[-1]
    sparse = [None] * s_indptr[-1]
    lastlen = [0] * (R * H)
    for r in range(R):
        for h in range(H):
            row = r * H + h
            lastlen[row] = seq_lens[r] % BLK or BLK
            base_d, base_s = d_indptr[row], s_indptr[row]
            ids = [
                (block_table[r][p // NBP] * H + h) * NBP + p % NBP
                for p in range(lens[r])
            ]
            dense[base_d: base_d + lens[r]] = ids
            if lens[r] <= budgets[r]:
                sparse[base_s: base_s + lens[r]] = ids
            else:
                sparse[base_s: base_s + bos] = ids[:bos]
                for p in range(eos):
                    sparse[base_s + budgets[r] - p - 1] = ids[lens[r] - p - 1]
    return d_indptr, s_indptr, dense, sparse, lastlen


def _unit_test():
    torch.manual_seed(0)
    dev = "cuda"
    H, BLK, NBP = 4, 64, 25
    MGR = BLK * NBP
    cfg = VortexConfig(topk_val=6, topk_ratio=0.0,
                       block_reserved_bos=1, block_reserved_eos=2)
    # seq lens: short row (fits budget), boundary row, long rows
    seq_lens = [130, (6 + 3) * BLK, 5000, 16000]
    R = len(seq_lens)
    MB = max(-(-s // MGR) for s in seq_lens) + 1
    bt = torch.randint(0, 500, (R, MB), dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)

    eff = R * H
    total_blocks = sum(-(-s // BLK) for s in seq_lens) * H + 8
    d_ip = torch.zeros(eff + 1, dtype=torch.int32, device=dev)
    s_ip = torch.zeros(eff + 1, dtype=torch.int32, device=dev)
    SENTINEL = -12345
    d_ix = torch.full((total_blocks,), SENTINEL, dtype=torch.int32, device=dev)
    s_ix = torch.full((total_blocks,), SENTINEL, dtype=torch.int32, device=dev)
    lpl = torch.ones(eff, dtype=torch.int32, device=dev)

    plan_decode_sparse(
        block_table=bt, seq_lens=sl, num_reqs=R, num_kv_heads=H,
        block_size=BLK, num_blocks_per_page=NBP, cfg=cfg,
        dense_kv_indptr=d_ip, dense_kv_indices=d_ix,
        sparse_kv_indptr=s_ip, sparse_kv_indices=s_ix,
        kv_last_page_len=lpl,
    )
    torch.cuda.synchronize()

    rd_ip, rs_ip, rdense, rsparse, rll = _reference(
        bt.cpu().tolist(), seq_lens, H, BLK, NBP,
        cfg.topk_val, cfg.topk_ratio,
        cfg.block_reserved_bos, cfg.block_reserved_eos)

    assert d_ip.cpu().tolist() == rd_ip, "dense indptr mismatch"
    assert s_ip.cpu().tolist() == rs_ip, "sparse indptr mismatch"
    assert lpl.cpu().tolist() == rll, "last_page_len mismatch"
    got_d = d_ix.cpu().tolist()
    assert got_d[: len(rdense)] == rdense, "dense indices mismatch"
    got_s = s_ix.cpu().tolist()
    holes = 0
    for i, want in enumerate(rsparse):
        if want is None:
            assert got_s[i] == SENTINEL, f"hole at {i} was WRITTEN ({got_s[i]})"
            holes += 1
        else:
            assert got_s[i] == want, f"sparse[{i}]: got {got_s[i]} want {want}"
    print(f"planner unit test PASS: {len(rdense)} dense ids, "
          f"{len(rsparse)} sparse slots ({holes} verified-unwritten holes)")


if __name__ == "__main__":
    _unit_test()
