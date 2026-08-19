"""VortexFlashInferBackend for vLLM V1 -- working implementation.

Port of vortex's sglang flashinfer backend, per-KV-head selection semantics
preserved exactly: eff_bs = num_decode_reqs * H_kv, both wrappers planned with
num_kv_heads=1 / num_qo_heads=G / page_size=vortex_block, sparse row =
[BOS | hole | reverse-EOS], terminal topK writes the hole in place into the
sparse wrapper's indices buffer.

Design decisions carried from the M0-M2 investigation:
  * STOCK packed KV layout (NB, H, BLK, 2D); the per-KV-head fold is a pure
    view (M2-verified against reference attention).
  * kernel block sizes = [vortex_block]; vLLM's kernel-block machinery then
    reshapes the cache and expands the block table for us, so the spec this
    builder receives already has block_size == vortex_block and nbp == 1.
  * prefill is DENSE (vortex never sparsifies prefill) via a gather + SDPA
    fallback -- correctness-first; flashinfer prefill is an optimisation TODO.
  * eager only for this landing: _cudagraph_support NEVER, run with
    enforce_eager=True.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper)

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from .config import VortexConfig
from .planner import plan_decode_sparse
from .scatter import newly_completed_pages

# Config/flow resolution. Plugin-grade path: everything derives from
# VllmConfig.additional_config["vortex"], which pickles across the process
# boundary -- no in-process requirement. configure() remains as an explicit
# override for tests.
_ACTIVE_CFG: VortexConfig | None = None
_ACTIVE_FLOW = None


def configure(cfg: VortexConfig, flow) -> None:
    global _ACTIVE_CFG, _ACTIVE_FLOW
    _ACTIVE_CFG, _ACTIVE_FLOW = cfg, flow


def _resolve_cfg(vllm_config=None) -> VortexConfig:
    if _ACTIVE_CFG is not None:
        return _ACTIVE_CFG
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        return VortexConfig.from_vllm_config(vllm_config)
    return VortexConfig()


def _resolve_flow(cfg: VortexConfig):
    """Per-process flow singleton (centroid state lives on it)."""
    global _ACTIVE_FLOW
    if _ACTIVE_FLOW is None:
        from .flow import VortexCentroidFlow
        _ACTIVE_FLOW = VortexCentroidFlow(cfg)
    return _ACTIVE_FLOW


@dataclass
class VortexAttnMetadata:
    # decode half
    decode_wrappers: list | None
    eff_bs: int
    num_decodes: int
    num_decode_tokens: int
    dense_kv_indptr: torch.Tensor | None
    dense_kv_indices: torch.Tensor | None
    sparse_kv_indptr: torch.Tensor | None
    max_block_len: int
    # prefill half (dense flashinfer paged prefill -- vortex never
    # sparsifies prefill, but it must not be SLOW either)
    num_prefills: int
    num_prefill_tokens: int
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None
    # forward_cache worklist, once per step
    newly_completed_pages: torch.Tensor | None = None


class VortexFlashInferBackend(AttentionBackend):
    # ABC defaults True (impl caches inside forward); ours uses the separate
    # unified_kv_cache_update op -> do_kv_cache_update, like stock flashinfer
    # (flashinfer.py:569). Without this the KV write NEVER HAPPENS.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"          # registered under AttentionBackendEnum.CUSTOM

    @staticmethod
    def get_impl_cls():
        return VortexFlashInferImpl

    @staticmethod
    def get_builder_cls():
        return VortexFlashInferMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256, 512]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # vLLM reshapes the cache + expands block tables to this granularity
        # (v1/worker/utils.py:261, gpu/block_table.py:44) -> nbp == 1 for us.
        return [_resolve_cfg().block_size]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype in (None, "auto", "bfloat16", "fp8",
                                  "fp8_e4m3", "fp8_e5m2")

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str: str = "auto"):
        if cache_dtype_str and cache_dtype_str.startswith("nvfp4"):
            raise ValueError("vortex needs packed (B,H,N,2D); nvfp4 KV packs "
                             "(B,2H,N,full_dim). Use fp8 or bf16 KV.")
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)


class VortexFlashInferMetadataBuilder(AttentionMetadataBuilder):
    # UNIFORM_BATCH: FULL decode graphs. Everything per-step (host CSR,
    # plan(), the indices kernel, the fixed worklist) runs in build() OUTSIDE
    # the graph; the captured graph replays impl.forward over the fixed
    # buffers. Non-uniform (prefill/mixed) steps fall back to the dynamic
    # path and run piecewise.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

        pc, mc = vllm_config.parallel_config, vllm_config.model_config
        self.num_qo_heads = mc.get_num_attention_heads(pc)
        self.num_kv_heads = mc.get_num_kv_heads(pc)
        self.group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = mc.get_head_size()
        self.cfg = _resolve_cfg(vllm_config)

        # Kernel-block reshaping already happened: the spec we receive is at
        # kernel granularity, which we declared == vortex block.
        assert kv_cache_spec.block_size == self.cfg.block_size, (
            f"spec block {kv_cache_spec.block_size} != vortex block "
            f"{self.cfg.block_size} -- kernel-block selection went wrong")
        self.vortex_block = self.cfg.block_size

        # STATIC per-row block cap for the fused indexer: its MAXBLK is a
        # baked triton constexpr, and capture-time dummy metadata has ~1-block
        # rows -- deriving it per-step bakes a 16-block scan into the graph
        # (measured: RULER 0.000 under capture). Masked loads make the static
        # bound free for shorter rows.
        self.static_max_blk = -(-mc.max_model_len // self.vortex_block)
        max_bs = vllm_config.scheduler_config.max_num_seqs
        spec = vllm_config.speculative_config
        self.dql_max = 1 + (spec.num_speculative_tokens if spec else 0)
        # Per-token rows under spec decode: each of the k+1 uniform draft
        # tokens is its own planner/wrapper row with its own causal extent.
        eff_max = max_bs * self.dql_max * self.num_kv_heads
        max_blocks_per_req = -(-mc.max_model_len // self.vortex_block)
        max_blocks = eff_max * max_blocks_per_req

        i32, dev = torch.int32, device
        self.dense_kv_indptr = torch.zeros(eff_max + 1, dtype=i32, device=dev)
        self.sparse_kv_indptr = torch.zeros(eff_max + 1, dtype=i32, device=dev)
        self.dense_kv_indices = torch.zeros(max_blocks, dtype=i32, device=dev)
        self.sparse_kv_indices = torch.zeros(max_blocks, dtype=i32, device=dev)
        self.kv_last_page_len = torch.ones(eff_max, dtype=i32, device=dev)

        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True),
            BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True),
        ]
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        self.kv_dtype = kv_cache_spec.dtype
        cache_str = str(vllm_config.cache_config.cache_dtype)
        if self.kv_dtype == torch.uint8 or "fp8" in cache_str:
            self.kv_dtype = (torch.float8_e5m2 if "e5m2" in cache_str
                             else torch.float8_e4m3fn)
        self.q_dtype = mc.dtype
        self.sm_scale = self.head_dim ** -0.5

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        # ---- FULL-capture support ------------------------------------------
        # Per padded-batch-size wrapper pairs, constructed over SLICES of the
        # persistent buffers (use_cuda_graph=True). plan() then copies host
        # indptr/lpl into those same device buffers, which both the captured
        # kernels AND our indices kernel read.
        self.graph_wrappers: dict[int, list] = {}
        self._graph_ws = ws
        # Fixed-shape forward_cache worklist for captured decode steps:
        # one slot per (req, head); non-closing rows point at page 0, the
        # never-allocated null block (block_pool reserves physical block 0).
        self.fixed_worklist = torch.zeros(eff_max, dtype=torch.int64, device=dev)

    def _get_graph_wrappers(self, eff: int) -> list:
        if eff not in self.graph_wrappers:
            self.graph_wrappers[eff] = [
                BatchDecodeWithPagedKVCacheWrapper(
                    self._graph_ws, "NHD", use_cuda_graph=True,
                    paged_kv_indptr_buffer=self.dense_kv_indptr[: eff + 1],
                    paged_kv_indices_buffer=self.dense_kv_indices,
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:eff],
                    use_tensor_cores=True),
                BatchDecodeWithPagedKVCacheWrapper(
                    self._graph_ws, "NHD", use_cuda_graph=True,
                    paged_kv_indptr_buffer=self.sparse_kv_indptr[: eff + 1],
                    paged_kv_indices_buffer=self.sparse_kv_indices,
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:eff],
                    use_tensor_cores=True),
            ]
        return self.graph_wrappers[eff]

    def _host_csr(self, seq_lens_cpu: torch.Tensor, nr: int):
        """Host mirror of the planner's indptr kernel (default policy only).
        Padded rows have seq_len 0 -> zero blocks, zero budget."""
        import numpy as np
        sl = seq_lens_cpu[:nr].numpy().astype(np.int64)
        lens = -(-sl // self.vortex_block)
        budget = np.maximum(
            self.cfg.topk_val + self.cfg.block_reserved_bos
            + self.cfg.block_reserved_eos,
            (lens * self.cfg.topk_ratio).astype(np.int64))
        sparse_len = np.minimum(budget, lens)
        H = self.num_kv_heads
        d = np.zeros(nr * H + 1, dtype=np.int32)
        sp = np.zeros(nr * H + 1, dtype=np.int32)
        d[1:] = np.repeat(lens, H).cumsum()
        sp[1:] = np.repeat(sparse_len, H).cumsum()
        lpl = sl % self.vortex_block
        lpl[lpl == 0] = self.vortex_block
        lpl[sl == 0] = 1
        lpl = np.repeat(lpl, H).astype(np.int32)
        return (torch.from_numpy(d), torch.from_numpy(sp),
                torch.from_numpy(lpl), int(lens.max()) if nr else 0)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        """Plan the capture-mode wrappers against WORST-CASE rows.

        FlashInfer bakes its split-KV work partition into the captured
        kernel from whatever plan() saw at capture. The runner's dummy
        capture metadata has ~1-token rows, which bakes a partition that
        can only ever read ~1 block per row -- measured as RULER 0.000
        for BOTH wrappers. This is the same reason vortex's sglang backend
        plans its capture graphs at max context."""
        m = common_attn_metadata
        max_len = self.vllm_config.model_config.max_model_len
        m.seq_lens.fill_(max_len)
        try:
            m.seq_lens_cpu.fill_(max_len)
        except (AssertionError, AttributeError, RuntimeError):
            pass
        try:
            m.max_seq_len = max_len
        except Exception:
            pass
        return self.build(common_prefix_len=0, common_attn_metadata=m)

    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> VortexAttnMetadata:
        m = common_attn_metadata
        nd_reqs, np_reqs, nd_toks, np_toks = split_decodes_and_prefills(
            m, decode_threshold=self.reorder_batch_threshold or 1,
            require_uniform=True)

        max_block_len = 0
        if np_reqs == 0 and nd_reqs > 0:
            dql = nd_toks // nd_reqs
            assert nd_toks == nd_reqs * dql, "non-uniform decode batch"
            eff = nd_toks * self.num_kv_heads
            # ---- pure uniform decode: FULL-graph path ----------------------
            assert self.cfg.schedule_policy is None, \
                "custom schedule_policy not supported on the capture path"
            try:
                sl_cpu = m.seq_lens_cpu
            except (AssertionError, AttributeError):
                sl_cpu = m.seq_lens.cpu()
            sl_dev = m.seq_lens[:nd_reqs]
            bt_dev = m.block_table_tensor[:nd_reqs]
            if dql > 1:
                # token t of request r attends to seq_len_r - (dql-1) + t
                # tokens -- earlier draft tokens included (their K/V is
                # already scattered before attention runs).
                off = torch.arange(dql, dtype=torch.int32)
                sl_cpu = (sl_cpu[:nd_reqs, None].to(torch.int32)
                          - (dql - 1) + off[None, :]).reshape(-1).clamp_(min=0)
                sl_dev = (sl_dev[:, None].to(torch.int32)
                          - (dql - 1) + off[None, :].to(sl_dev.device)
                          ).reshape(-1).clamp_(min=0)
                bt_dev = bt_dev.repeat_interleave(dql, dim=0)
            d_cpu, s_cpu, lpl_cpu, _ = self._host_csr(sl_cpu, nd_toks)
            max_block_len = self.static_max_blk
            wr = self._get_graph_wrappers(eff)
            common = dict(num_qo_heads=self.group_size, num_kv_heads=1,
                          head_dim=self.head_dim, page_size=self.vortex_block,
                          q_data_type=self.q_dtype, kv_data_type=self.kv_dtype,
                          sm_scale=self.sm_scale)
            # Original plan() per step: correct-by-construction, and
            # cheap -- measured 0.14 ms/step for the PAIR, and it does not
            # block against a busy stream. (The old "~14 ms plan cost" was
            # a misattribution: the real cost was the 65-way KV-cache group
            # explosion under MTP, fixed via VortexQwen3_5MTP; see
            # ISSUES.md P1-2.) The earlier fast_plan_decode attempt broke
            # (RULER 0.125) because its cudagraph branch skips the H2D
            # refresh of the wrapper's DEVICE indptr/last_page_len buffers:
            # stock vLLM maintains those buffers itself, while this capture
            # path takes them from plan()'s copy -- the indices kernel and
            # the replayed kernel both read them. If plan() ever matters
            # again: pinned-staging H2D refresh of both indptrs + lpl, then
            # flashinfer.decode.fast_decode_plan.
            wr[0].plan(indptr=d_cpu, indices=self.dense_kv_indices,
                       last_page_len=lpl_cpu, **common)
            wr[1].plan(indptr=s_cpu, indices=self.sparse_kv_indices,
                       last_page_len=lpl_cpu, **common)
            from .planner import launch_indices_kernel
            launch_indices_kernel(
                block_table=bt_dev.contiguous(),
                seq_lens=sl_dev,
                num_reqs=nd_toks, num_kv_heads=self.num_kv_heads,
                block_size=self.vortex_block, num_blocks_per_page=1,
                cfg=self.cfg,
                dense_kv_indptr=self.dense_kv_indptr,
                dense_kv_indices=self.dense_kv_indices,
                sparse_kv_indptr=self.sparse_kv_indptr,
                sparse_kv_indices=self.sparse_kv_indices,
                kv_last_page_len=self.kv_last_page_len)
            # fixed-shape forward_cache worklist, from slot_mapping: one
            # slot per (token, head); non-closing rows redirect to the null
            # block's pages (physical block 0 is never allocated). Correct
            # for dql > 1 (any draft token can close a block) and refires on
            # spec-decode rejection rewrites.
            slots = m.slot_mapping[:nd_toks]
            closing = (slots >= 0) & ((slots + 1) % self.vortex_block == 0)
            kb = torch.where(closing, slots // self.vortex_block,
                             torch.zeros_like(slots))
            heads = torch.arange(self.num_kv_heads, device=slots.device)
            self.fixed_worklist[:eff] = (
                kb[:, None] * self.num_kv_heads + heads[None, :]).reshape(-1)
            return VortexAttnMetadata(
                decode_wrappers=wr, eff_bs=eff, num_decodes=nd_reqs,
                num_decode_tokens=nd_toks,
                dense_kv_indptr=self.dense_kv_indptr,
                dense_kv_indices=self.dense_kv_indices,
                sparse_kv_indptr=self.sparse_kv_indptr,
                max_block_len=max_block_len,
                num_prefills=0, num_prefill_tokens=0, prefill_wrapper=None,
                newly_completed_pages=self.fixed_worklist[:eff])

        eff = 0
        if nd_reqs > 0:
            dql = nd_toks // nd_reqs
            eff = nd_toks * self.num_kv_heads
            sl_dev = m.seq_lens[:nd_reqs]
            bt_dev = m.block_table_tensor[:nd_reqs]
            if dql > 1:
                off = torch.arange(dql, dtype=torch.int32,
                                   device=sl_dev.device)
                sl_dev = (sl_dev[:, None].to(torch.int32) - (dql - 1)
                          + off[None, :]).reshape(-1).clamp_(min=0)
                bt_dev = bt_dev.repeat_interleave(dql, dim=0).contiguous()
            plan_decode_sparse(
                block_table=bt_dev,
                seq_lens=sl_dev,
                num_reqs=nd_toks, num_kv_heads=self.num_kv_heads,
                block_size=self.vortex_block, num_blocks_per_page=1,
                cfg=self.cfg,
                dense_kv_indptr=self.dense_kv_indptr,
                dense_kv_indices=self.dense_kv_indices,
                sparse_kv_indptr=self.sparse_kv_indptr,
                sparse_kv_indices=self.sparse_kv_indices,
                kv_last_page_len=self.kv_last_page_len)
            common = dict(last_page_len=self.kv_last_page_len[:eff],
                          num_qo_heads=self.group_size, num_kv_heads=1,
                          head_dim=self.head_dim, page_size=self.vortex_block,
                          q_data_type=self.q_dtype, kv_data_type=self.kv_dtype,
                          sm_scale=self.sm_scale)
            self.decode_wrappers[0].plan(
                indptr=self.dense_kv_indptr[: eff + 1],
                indices=self.dense_kv_indices, **common)
            self.decode_wrappers[1].plan(
                indptr=self.sparse_kv_indptr[: eff + 1],
                indices=self.sparse_kv_indices, **common)
            max_block_len = self.static_max_blk

        if np_reqs > 0:
            # Dense prefill plan over the STOCK (unfolded) layout at kernel
            # granularity: num_kv_heads=H, page_size=64, causal. CSR built
            # host-side (flashinfer plan is host work anyway; prefill steps
            # only).
            sl_cpu = m.seq_lens[nd_reqs:].cpu()
            bt_cpu = m.block_table_tensor[nd_reqs:].cpu()
            qsl = m.query_start_loc_cpu[nd_reqs:] - m.query_start_loc_cpu[nd_reqs]
            lens = (sl_cpu + self.vortex_block - 1) // self.vortex_block
            p_indptr = torch.zeros(np_reqs + 1, dtype=torch.int32)
            p_indptr[1:] = torch.cumsum(lens, 0)
            p_indices = torch.cat([bt_cpu[i, : int(lens[i])] for i in range(np_reqs)])
            lpl = sl_cpu % self.vortex_block
            lpl[lpl == 0] = self.vortex_block
            self.prefill_wrapper.plan(
                qo_indptr=qsl.to(torch.int32),
                paged_kv_indptr=p_indptr,
                paged_kv_indices=p_indices.to(torch.int32),
                paged_kv_last_page_len=lpl.to(torch.int32),
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim, page_size=self.vortex_block,
                causal=True, q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype, sm_scale=self.sm_scale)

        pages = newly_completed_pages(
            m.slot_mapping, self.vortex_block,
            self.vortex_block,               # nbp==1: manager==kernel==vortex
            self.num_kv_heads)

        return VortexAttnMetadata(
            decode_wrappers=self.decode_wrappers if nd_reqs else None,
            eff_bs=eff, num_decodes=nd_reqs, num_decode_tokens=nd_toks,
            dense_kv_indptr=self.dense_kv_indptr,
            dense_kv_indices=self.dense_kv_indices,
            sparse_kv_indptr=self.sparse_kv_indptr,
            max_block_len=max_block_len,
            num_prefills=np_reqs, num_prefill_tokens=np_toks,
            prefill_wrapper=self.prefill_wrapper if np_reqs else None,
            newly_completed_pages=pages)


class VortexFlashInferImpl(AttentionImplBase):
    def __init__(self, num_heads, head_size, scale, num_kv_heads,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                 logits_soft_cap=None, attn_type="decoder",
                 kv_sharing_target_layer_name=None, *,
                 vortex_layer_idx: int = -1, vortex_force_dense: bool = False,
                 **_ignored):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.fp8_view = None
        if kv_cache_dtype and "fp8" in str(kv_cache_dtype):
            self.fp8_view = (torch.float8_e5m2 if "e5m2" in str(kv_cache_dtype)
                             else torch.float8_e4m3fn)
        self.layer_idx = vortex_layer_idx
        self.cfg = _resolve_cfg()      # model builds under set_current_vllm_config
        self.flow = _resolve_flow(self.cfg)
        self.use_sparsity = (not vortex_force_dense
                             and vortex_layer_idx not in self.cfg.layers_skip)

    # -- folded view (M2-verified) ------------------------------------------
    def _folded(self, kv_cache):
        NB, H, BLK, twoD = kv_cache.shape
        D = self.head_size
        f = kv_cache.view(NB * H, BLK, twoD)
        k, v = f[:, :, :D], f[:, :, D:]
        if kv_cache.dtype == torch.uint8 and self.fp8_view is not None:
            # vLLM stores fp8 caches as uint8; flashinfer keys NVFP4 off
            # uint8, so reinterpret to the true fp8 dtype (1-byte view is
            # legal on the last-dim-contiguous slice).
            k, v = k.view(self.fp8_view), v.view(self.fp8_view)
        return k.unsqueeze(2), v.unsqueeze(2)

    def _md(self, layer):
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            return md.get(layer.layer_name)
        return md

    # -- HOOK 1: stock scatter + forward_cache ------------------------------
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, k_cache, v_cache, slot_mapping,
            self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        if not self.use_sparsity:
            return          # centroids are only read by this layer's indexer
        md = self._md(layer)
        if md is None or self.flow is None:
            return
        fk, _ = self._folded(kv_cache)
        self.flow.forward_cache(fk, md.newly_completed_pages, self.layer_idx)

    # -- HOOK 2: indexer + attend -------------------------------------------
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        md: VortexAttnMetadata = attn_metadata
        if md is None:                        # memory-profiling run
            output.zero_()
            return output
        H, G, D = self.num_kv_heads, self.group_size, self.head_size
        q = query.view(-1, self.num_heads, D)
        k_view, v_view = self._folded(kv_cache)
        ks = getattr(layer, "_k_scale_float", 1.0) or 1.0
        vs = getattr(layer, "_v_scale_float", 1.0) or 1.0

        nd = md.num_decode_tokens
        if md.num_decodes > 0:
            qd = q[:nd].view(nd, H, G, D).reshape(nd * H, G, D).contiguous()
            if self.use_sparsity and self.flow is not None:
                # allocate centroid state OUTSIDE capture (warmup runs eager
                # first); capture must never see cent is None.
                self.flow.ensure(self.layer_idx, kv_cache.shape[0] * H, D,
                                 q.device)
                self.flow.forward_indexer(
                    qd, self.layer_idx, md,
                    o=md.decode_wrappers[1]._paged_kv_indices_buf)
                w = md.decode_wrappers[1]
            else:
                w = md.decode_wrappers[0]
            od = w.run(qd, (k_view, v_view), k_scale=ks, v_scale=vs)
            output[:nd] = od.view(nd, H * G, D).to(output.dtype)

        # ---- prefill rows: DENSE flashinfer paged prefill ------------------
        # Unfolded NHD view of the stock cache: (NBk, H, 64, 2D) ->
        # permute -> [NBk, 64, H, 2D] -> split K/V. Strided views are fine --
        # this is exactly what stock vLLM feeds the wrapper
        # (flashinfer.py:2023-2056).
        if md.num_prefills > 0:
            perm = kv_cache.permute(0, 2, 1, 3)
            ku, vu = perm[..., :D], perm[..., D:]
            if kv_cache.dtype == torch.uint8 and self.fp8_view is not None:
                ku, vu = ku.view(self.fp8_view), vu.view(self.fp8_view)
            qp = q[nd:]
            op = md.prefill_wrapper.run(qp, (ku, vu), k_scale=ks, v_scale=vs)
            output[nd:] = op.to(output.dtype)
        return output
