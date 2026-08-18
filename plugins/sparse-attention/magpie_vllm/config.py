"""Vortex knobs, carried into vLLM.

Same names and semantics as the sglang submission JSON, so a
``submissions/<tag>/batch_<x>_id<y>.json`` stays readable across both engines.

Transport: vLLM's only generic escape hatch that survives to the worker is
``--additional-config`` (``vllm/config/vllm.py:395``); every config dataclass
is ``extra="forbid"``. So:

    vllm serve <model> --additional-config '{"vortex": {"topk_val": 64, ...}}'
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VortexConfig:
    # --- selection budget -------------------------------------------------
    topk_val: int = 30
    topk_ratio: float = 0.0
    block_reserved_bos: int = 1
    block_reserved_eos: int = 1          # >= 1 is LOAD-BEARING: the trailing
                                         # partial block never gets a summary,
                                         # so eos=0 scores it on stale bytes.
                                         # vortex hard-enforces this.

    # --- placement --------------------------------------------------------
    # NOTE the sglang default was list(range(1)) == [0], i.e. "skip layer 0",
    # NOT "skip nothing". Empty list is what disables skipping. On Qwen3.8-27B
    # this indexes the 16 FULL-ATTENTION layers only ([3,7,...,63] in model
    # terms), so decide which numbering you mean and assert it at startup.
    layers_skip: tuple[int, ...] = ()

    # --- geometry ---------------------------------------------------------
    # == flashinfer page_size. Capped at 64 on SM120 (RTX 5090): large pages
    # need the trtllm-gen kernel, which requires device capability family 100,
    # and 120//10 == 12 != 10.
    block_size: int = 64
    workload_chunk_size: int = 16

    # --- scoring ----------------------------------------------------------
    schedule_policy: str | None = None   # spliced into ComputeKvBudget verbatim

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "VortexConfig":
        raw = (vllm_config.additional_config or {}).get("vortex", {})
        known = {f for f in cls.__dataclass_fields__}
        # Vortex-on-sglang silently swallowed misspelled vortex_* keys, which
        # meant a typo'd knob ran at its default with no error anywhere. Do not
        # reproduce that: fail loudly instead.
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown vortex config keys: {sorted(unknown)}; "
                f"known keys are {sorted(known)}"
            )
        if "layers_skip" in raw:
            raw = {**raw, "layers_skip": tuple(raw["layers_skip"])}
        return cls(**raw)
