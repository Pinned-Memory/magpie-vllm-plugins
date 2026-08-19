"""``vllm.general_plugins`` entry point.

Loaded by vLLM in the frontend, the engine core, and every worker process
(vllm/plugins/__init__.py:87; v1/worker/worker_base.py:249), which is what
makes the registrations reach the process that actually builds the model --
no VLLM_ENABLE_V1_MULTIPROCESSING=0 required.

Everything registered here is inert until ``--additional-config
'{"vortex": {...}}'`` is present: the model class defers to stock without it,
and the CUSTOM backend slot is only consulted when the model injects it.
"""

from __future__ import annotations


def register() -> None:
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum, register_backend,
    )

    from .backend import VortexFlashInferBackend

    register_backend(AttentionBackendEnum.CUSTOM, None, is_mamba=False)(
        VortexFlashInferBackend)
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "magpie_vllm.model:VortexQwen3_5ForConditionalGeneration",
    )
    # MTP draft overrides: same config gate, same inertness without
    # {"vortex": ...}. Needed so the draft full-attn layer shares the
    # vortex KV-cache spec (see VortexQwen3_5MTP docstring).
    ModelRegistry.register_model(
        "Qwen3_5MTP", "magpie_vllm.model:VortexQwen3_5MTP")
    ModelRegistry.register_model(
        "Qwen3_5MoeMTP", "magpie_vllm.model:VortexQwen3_5MoeMTP")
