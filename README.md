# magpie-vllm-plugins

The magpie team's vLLM optimizations for DGX Spark (GB10) serving.

Each optimization is a **patch over the vLLM checkout** plus plain
supporting **scripts** — no plugins, no entry points, no env vars, nothing
to pip-install. Apply the patch once, drive the feature through vLLM's own
arguments. Unactivated, every patched file behaves stock.

## Catalog

| Optimization | What it does | Measured gain | Docs |
|---|---|---|---|
| **ple-ssd** | Parks Qwen3.8-Flash-Next's 102 GB n-gram (PLE / "Engram") embedding table on the SSD: the loader's zero-copy mmap views of the 128 checkpoint shards are kept instead of copied to the device, and rows are gathered on demand through the page cache. Patch 0001 makes the 176B model run on one GB10; patch 0002 stages the rows before the forward so decode runs under full CUDA graphs. Activated with `--hf-overrides '{"ple_mmap": true}'`. | Fits: 97 GB resident of 121, 22 GB free. Decode at c=1: 32.8 tok/s (0001) → 34.1 (0002); c=4 aggregate 77.7 → 82.8. GSM8K 96.5–97.0 % (n=200). SSD gather is 0.1–0.3 ms of a 72 ms step. | [docs/ple-ssd.md](docs/ple-ssd.md) |
| **mtp-pruning** | Prunes the MTP draft vocabulary to a frequency keep-set, slicing the shared draft `lm_head` (1.2–2.5 GB) to a few % of its rows. Lossless output; the target still verifies with its full head. Qwen3.5 and Qwen4Exp (Flash-Next) drafters. | Qwen3.8-27B: draft-head 45 → ~1 ms/step, decode 1.28–1.35× at c=1–8, acceptance −1 %. Flash-Next: c=1 34.1 → **41.8 tok/s (1.23×)**, c=4 aggregate 82.8 → 86.5, GSM8K unchanged. | [docs/mtp-pruning.md](docs/mtp-pruning.md) |

Where a Flash-Next decode step goes (torch profiler, best configuration):
the worker holds 95–97 % of the engine-step loop, scheduler and engine I/O
under 3 %; the GPU is 85–89 % busy over ~2,300 kernel launches per step
(~7 graph replays); of GPU time, BF16 side-layer GEMMs are 59 % at c=1 /
42 % at c=4 and NVFP4 expert GEMMs 29 % / 43 %. Full tables in
[docs/ple-ssd.md](docs/ple-ssd.md#worker-vs-scheduler-and-kernel-counts-2026-09-08-captures);
the delivered report is [docs/flash-next-report.html](docs/flash-next-report.html).

## Installation

Prerequisites: a vLLM source checkout you serve from (patches are maintained
against `vllm-project/vllm` @ `33898f832c`, v0.29.0rc1; nearby commits usually
apply cleanly — `git apply --check` tells you), and its Python environment.

```bash
git clone git@github.com:Pinned-Memory/magpie-vllm-plugins.git
cd magpie-vllm-plugins

# 1) check, then apply the optimization's patch to your vLLM checkout
git -C /path/to/vllm apply --check patches/mtp-pruning/0001-mtp-draft-vocab-pruning.patch
git -C /path/to/vllm apply         patches/mtp-pruning/0001-mtp-draft-vocab-pruning.patch

# 2) restart your server; activate per the optimization's doc, e.g.
vllm serve <model> --speculative-config \
    '{"method":"mtp","num_speculative_tokens":3,"draft_vocab_path":"/abs/keep.pt"}'
```

To remove an optimization: `git -C /path/to/vllm apply -R <patch>` (or
`git checkout` the touched files) and restart.

Scripts run with the vLLM environment's Python and need nothing extra:

```bash
/path/to/vllm-venv/bin/python scripts/mtp_pruning/count_tokens.py --help
```

## Qwen3.8-Flash-Next on one GB10

The full stack for `Inferact/Qwen3.8-Flash-Next-NVFP4` (171 GB checkpoint,
of which 102 GB is the BF16 n-gram table) on a 121 GB Spark:

```bash
# patches, order-independent, all three for the measured configuration
for p in patches/ple-ssd/0001-qwen4-exp-ple-table-on-ssd.patch \
         patches/ple-ssd/0002-stage-rows-before-forward-full-cudagraphs.patch \
         patches/mtp-pruning/0001-mtp-draft-vocab-pruning.patch; do
  git -C /path/to/vllm apply "$p"
done

# serve: memory-conservative defaults (0.76 utilization, 4 seqs, 64k context,
# text tower only); MTP k=3 by default, MTP=0 disables it
DRAFT_VOCAB_PATH=/abs/keep.pt GPU_MEM=0.76 MAX_NUM_SEQS=4 \
  scripts/flash_next/serve.sh server.log --language-model-only
scripts/flash_next/memguard.sh memguard.log &   # kills the server under 6 GB free

# measure: decode tok/s + MTP acceptance, GSM8K, a profiled decode/prefill window
scripts/flash_next/run_suite.sh results/<run>
scripts/flash_next/profile_capture.py results/<run>/trace && \
  scripts/flash_next/trace_split.py results/<run>/trace/decode_c1/*.gz

# keep-set for the pruned draft head, fitted on the model's own generations
scripts/flash_next/build_keepset.sh results/<run> keep.pt
```

Notes: the first launch JIT-builds FlashInfer's sm_120 fused-MoE kernel
(~5 min; `serve.sh` puts the venv's `ninja` and `/usr/local/cuda/bin` on
PATH); every launch loads weights for ~10 min; the profiler with stack
tracing grows the engine by ~10 GB over a few captures, so run it on a
dedicated launch. Measured runs and their logs live under
`scripts/flash_next/results/` (see its README).

## Layout

```
patches/<optimization>/NNNN-*.patch   # the vLLM change, numbered, git-apply-able
scripts/<optimization>/*.py           # plain workflow scripts (stdlib + vLLM-env deps)
scripts/flash_next/                   # serving, benchmark, eval, profiling stack for Flash-Next
docs/<optimization>.md                # mechanism, workflow, measured results, limits
docs/flash-next-report.html           # the delivered report (also published as an artifact)
```

Adding an optimization means adding one row to the catalog and one entry in
each of those three places.
