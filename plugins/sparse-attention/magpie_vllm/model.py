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
from vllm.model_executor.models.utils import extract_layer_index

_ORIG_ATTENTION = _q3n.Attention
_ORIG_Q3N_ATTN = _q3n.Qwen3NextAttention


class VortexQwen3NextAttention(_ORIG_Q3N_ATTN):
    """Parent __init__ runs unchanged; only its inline ``Attention(...)``
    resolves to a shim injecting attn_backend= + the layer index."""

    def __init__(self, *args, **kwargs):
        from .backend import VortexFlashInferBackend

        def patched_attention(*a, **kw):
            kw["attn_backend"] = VortexFlashInferBackend
            kw["vortex_layer_idx"] = extract_layer_index(kw["prefix"])
            return _ORIG_ATTENTION(*a, **kw)

        _q3n.Attention = patched_attention
        try:
            super().__init__(*args, **kwargs)
        finally:
            _q3n.Attention = _ORIG_ATTENTION


@contextlib.contextmanager
def _vortex_attention_layers():
    _q35.Qwen3NextAttention = VortexQwen3NextAttention
    _q3n.Qwen3NextAttention = VortexQwen3NextAttention
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
