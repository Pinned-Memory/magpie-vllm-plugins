"""Model wiring, plugin-grade.

``VortexQwen3_5ForConditionalGeneration`` is registered over the stock
architecture name from the ``vllm.general_plugins`` entry point, so it loads
in EVERY vLLM process (frontend, engine core, workers) with no monkey-patching
at import time and no in-process requirement.

Activation is config-gated: with no ``{"vortex": ...}`` in
``--additional-config`` the class constructs the model bit-identically to
stock (the swap never engages), so the plugin is safe to leave installed.

The one genuine patch survives here, tightly scoped: Qwen3NextAttention
builds its inner ``Attention(...)`` inline with no injection parameter
(qwen3_next.py:298), and constructing stock-then-replace is impossible
(duplicate-prefix guard, attention.py:437). So for the duration of model
construction we rebind the class name in the two namespaces that resolve it,
and restore in ``finally``.
"""

from __future__ import annotations

import contextlib

import vllm.model_executor.models.qwen3_5 as _q35
import vllm.model_executor.models.qwen3_next as _q3n
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP
from vllm.model_executor.models.utils import extract_layer_index

_ORIG_ATTENTION = _q3n.Attention
_ORIG_Q3N_ATTN = _q3n.Qwen3NextAttention


class VortexQwen3NextAttention(_ORIG_Q3N_ATTN):
    """Parent __init__ runs unchanged; only its inline ``Attention(...)``
    resolves to a shim injecting attn_backend= + the layer index."""

    vortex_force_dense = False       # subclass toggles; read per-construction

    def __init__(self, *args, **kwargs):
        from .backend import VortexFlashInferBackend

        force_dense = self.vortex_force_dense

        def patched_attention(*a, **kw):
            kw["attn_backend"] = VortexFlashInferBackend
            kw["vortex_layer_idx"] = extract_layer_index(kw["prefix"])
            if force_dense:
                kw["vortex_force_dense"] = True
            return _ORIG_ATTENTION(*a, **kw)

        _q3n.Attention = patched_attention
        try:
            super().__init__(*args, **kwargs)
        finally:
            _q3n.Attention = _ORIG_ATTENTION


class VortexQwen3NextAttentionDense(VortexQwen3NextAttention):
    """Vortex backend, sparsity permanently off (dense folded wrapper --
    numerically identical to stock; M2-verified). Used for the MTP draft
    layer, where the point is KV-cache-spec parity, not sparsity."""

    vortex_force_dense = True


@contextlib.contextmanager
def _vortex_attention_layers(dense: bool = False):
    cls = VortexQwen3NextAttentionDense if dense else VortexQwen3NextAttention
    _q35.Qwen3NextAttention = cls
    _q3n.Qwen3NextAttention = cls
    try:
        yield
    finally:
        _q35.Qwen3NextAttention = _ORIG_Q3N_ATTN
        _q3n.Qwen3NextAttention = _ORIG_Q3N_ATTN


class VortexQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # Explicit signature: vLLM inspects __init__ to decide what to pass;
    # *args/**kwargs hides vllm_config and it is silently not forwarded.
    def __init__(self, *, vllm_config, prefix: str = ""):
        vortex_on = bool((vllm_config.additional_config or {}).get("vortex"))
        if vortex_on:
            with _vortex_attention_layers():
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)


class VortexQwen3_5MTP(Qwen3_5MTP):
    """MTP draft override: the draft's full-attention layer takes the vortex
    backend in FORCED-DENSE mode.

    This is a PERFORMANCE fix, not a sparsity feature. vLLM buckets KV-cache
    layers by spec equality; ``indexes_kv_by_block_stride`` (derived from the
    attention backend) is part of the spec. With the target's 16 full-attn
    layers on the vortex backend (False) and the draft's 1 on stock
    flashinfer (True), the lone draft bucket drives the hybrid group size to
    min(...) == 1 -> 65 single-layer KV-cache groups -> 65 metadata builds
    per step (measured: 48x gdn.build + 16x vortex.build, ~15 ms/step host,
    the entire sparse-vs-stock MTP wall gap at batch 1). With the draft layer
    on the vortex backend its spec matches, the full-attn bucket is 17
    layers, and grouping returns to stock shape (4 groups, ~5 builds/step).

    Dense-forced because the draft imitates the DENSE-trained target; there
    is no reason to sparsify one 17th of the layers at k<=64 budgets, and
    dense keeps the draft numerically identical to stock (M2-verified fold).
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        vortex_on = bool((vllm_config.additional_config or {}).get("vortex"))
        if vortex_on:
            with _vortex_attention_layers(dense=True):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)


class VortexQwen3_5MoeMTP(Qwen3_5MoeMTP):
    def __init__(self, *, vllm_config, prefix: str = ""):
        vortex_on = bool((vllm_config.additional_config or {}).get("vortex"))
        if vortex_on:
            with _vortex_attention_layers(dense=True):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
