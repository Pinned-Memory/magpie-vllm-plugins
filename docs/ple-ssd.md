# ple-ssd

Park Qwen3.8-Flash-Next's 51B-parameter n-gram embedding table on the SSD
and serve it through the page cache, so the model fits a single GB10.

## Why

Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) carries a
**PLE / "Engram" n-gram embedding**: 16 hashed bigram/trigram heads × 20M
rows × 160 dims = 320M rows, 51.2B parameters, **102 GB in BF16** in the
NVFP4 checkpoint (`model-00001-of-00004.safetensors`, 128 shards of
`[2500012, 160]`). Stock vLLM copies every shard into a device-resident
`VocabParallelEmbedding`. The rest of the model is ~83 GB resident (NVFP4
experts 68 GB, BF16 side layers 11 GB, MTP 1.6 GB); the GB10's 121 GB
unified pool cannot hold both, and on unified memory a host-side offload
frees nothing.

The table is a lookup, not compute: a token reads exactly 16 rows
(16 × 320 B = 5 KB) at hashed, effectively random addresses. That is a
page-cache workload, not a weight-streaming one.

## Mechanism

Two patches. `0001` makes the model run; `0002` is the tracked
optimization on top (full CUDA graphs) and is the recommended state.

`patches/ple-ssd/0001-qwen4-exp-ple-table-on-ssd.patch`, 3 files:

- `vllm/models/qwen4_exp/nvidia/ple_layer.py`
  - `Qwen4ExpNGramEmbedding(mmap_table=True)` allocates **no** embedding.
    vLLM's default safetensors loader already hands `load_weights` the
    shard tensors as **zero-copy mmap views** of the checkpoint file
    (`safe_open(...).get_tensor`, verified: 800 MB shard in 14 ms, no RSS
    growth, the mapping outlives the file handle). The patch simply keeps
    those 128 views (`self.shards`) instead of copying them into a
    parameter, and calls `madvise(MADV_RANDOM)` on each so a row miss
    costs one 4 KiB page read instead of the default 128 KiB readahead.
  - `gather_rows(ngram_ids)`: `torch.unique` on the GPU (dedup + sort),
    one D2H copy of the unique ids, sorted ids split by shard run and
    chunked across a 16-thread pool (`index_select` releases the GIL, so
    page faults overlap on the NVMe), rows land in a persistent **pinned**
    staging buffer, one async H2D copy, and the inverse index expands back
    to `[T, 16, 160]` on the GPU.
  - `forward` routes through a new custom op
    `vllm::qwen4_exp_ple_mmap_lookup` (hash ids → gather → copy into the
    output buffer). Everything downstream (dequant, key/value projections,
    the short conv, the HC gated residual) is stock.
- `vllm/config/compilation.py` — the op is added to the splitting-op list
  so piecewise CUDA graphs are cut around it: the gather is host work and
  must run between graph segments.
- `vllm/model_executor/models/config.py` — with `ple_mmap` set, the
  cudagraph mode is forced to `PIECEWISE` (a FULL capture would record the
  H2D copy of stale rows).

`patches/ple-ssd/0002-stage-rows-before-forward-full-cudagraphs.patch`
(apply after 0001), 4 files: the runner's `Qwen4ExpModelState.prepare_inputs`
— called by the V2 model runner immediately before the forward — hashes the
step's n-gram ids and gathers their rows into a static device buffer owned
by the embedding (`stage_rows`); the forward only reads a slice of that
buffer. No host work is left inside the forward, so the custom op, the
splitting-op entry and the PIECEWISE forcing are removed: the target's
decode batches and the MTP draft's decode steps run under full CUDA graphs
(stock `FULL_AND_PIECEWISE`).

Activation is a vLLM argument on the HF config namespace, next to the
other PLE layout attributes the layer already reads:

```bash
vllm serve <snapshot> --hf-overrides '{"ple_mmap": true}' ...
```

Unset, the file behaves stock (device-resident table).

## Measured

See `scripts/flash_next/results/` (this box: GB10, 121 GB unified,
NVMe 3.7 TB, vLLM 33898f832c + patches). Filled in below.

Run `mtp3` (patch 0001 only, piecewise graphs, MTP k=3 unpruned,
`--language-model-only`, 0.76 utilization, 4 seqs; KV cache 15.25 GiB =
393k tokens):

| Concurrency | Decode tok/s per request (mean / median) | Aggregate output tok/s | TTFT median | MTP acceptance length |
|---|---|---|---|---|
| 1 | 32.8 / 32.9 | 31.6 | 0.39 s | 3.09 |
| 4 | 20.9 / 20.8 | 77.7 | 0.67 s | 3.13 |

512-token answers to 16 coding/agent prompts, greedy, thinking off, no
`ignore_eos`. GSM8K, 200 test questions, zero-shot, thinking off, greedy,
1024-token cap: **96.5 % accuracy** (4 answers truncated), 86.7 output
tok/s at c=4.

Resident memory after load: 97 GB used of 121 (weights 74.5 GiB + KV
15.25 GiB), 23 GB available for the OS and the page cache that serves the
table.

Run `fullgraph-mtp3` (patches 0001 + 0002, `FULL_AND_PIECEWISE`: full
graphs for target decode, draft prefill and draft decode; everything else
identical):

| Concurrency | Decode tok/s per request | Aggregate output tok/s | TTFT median | MTP acceptance length |
|---|---|---|---|---|
| 1 | 34.1 (+4 %) | 33.0 | 0.36 s | 3.10 |
| 4 | 22.0 (+5 %) | 82.8 (+7 %) | 0.62 s | 3.24 |

GSM8K 200 questions: **97.0 %** (3 truncated), 89.1 output tok/s at c=4.
Greedy outputs differ in wording from the piecewise run on some prompts
(kernel selection changes the low-order logit bits), both correct.

Run `fullgraph-mtp0` (patches 0001 + 0002, **MTP off**): c=1 **16.9 tok/s**
(TTFT 0.24 s), c=4 13.5 per request / **52.6 aggregate** — MTP k=3
with the pruned head is 2.5× at c=1 and 1.6× at c=4 over plain decoding.
GSM8K (100 questions): **97.0 %** (3 truncated) — the same as with MTP, as expected from a lossless verifier.

Run `fullgraph-mtp3-pruned` (+ mtp-pruning, 10,010-id keep-set): c=1
**41.8 tok/s**, c=4 23.2 per request / **86.5 aggregate**, GSM8K **97.0 %**
(2 truncated), greedy outputs identical to the unpruned full-graph run on
the coherence prompts. See `docs/mtp-pruning.md` for the breakdown.

### Where a decode step goes (torch profiler, full-graph run)

`scripts/flash_next/profile_decode.sh` + `trace_summary.py`, one c=1
request, 64 tokens (~21 MTP steps), profiler overhead included:

| | |
|---|---|
| GPU busy | 95 % of the window — the step is GPU-bound |
| `stage_rows` (hash + SSD gather + H2D) | 0.7–1 ms wall per step; the rest of its CPU time is the D2H sync waiting for the previous step |
| BF16 side-layer GEMMs (GDN/QSA projections, shared experts, HC) | ~44 % of GPU time |
| `lm_head` GEMV (target + 3 draft steps, 1.2 GB each) | ~22 % — the mtp-pruning target |
| NVFP4 grouped GEMM (routed experts) | ~21 % |
| H2D of the gathered rows | 0.3 ms total |

Same window on the pruned-head run (`fullgraph-mtp3-pruned/trace_summary.txt`):
the cuBLAS GEMV total falls from 502 ms to 84 ms (the target's own full
head remains), `stage_rows` is unchanged, and the window is now 58 % BF16
side-layer GEMMs and 27 % NVFP4 expert GEMMs.

The SSD table is not on the critical path; the checkpoint's BF16 side
layers are (the reference recipe's "hybrid" FP8 conversion of those is
the next lever after pruning).

## Workflow

```bash
# apply (order-independent with mtp-pruning)
git -C /path/to/vllm apply patches/ple-ssd/0001-qwen4-exp-ple-table-on-ssd.patch

# serve: memory-conservative defaults for a 121 GB GB10
GPU_MEM=0.76 MAX_NUM_SEQS=4 scripts/flash_next/serve.sh server.log --language-model-only
# + MTP:      MTP=3 (default)      MTP=0 disables
# + pruning:  DRAFT_VOCAB_PATH=/abs/keep.pt  (mtp-pruning patch)

# decode throughput + MTP acceptance (streaming timestamps, /metrics delta)
python scripts/flash_next/bench_decode.py --concurrency 1 --num-prompts 8 --out bench_c1.json
# GSM8K accuracy (zero-shot, thinking off, greedy)
python scripts/flash_next/gsm8k_eval.py -n 200 --out gsm8k.json
```

`scripts/flash_next/memguard.sh LOG` kills the server if `MemAvailable`
falls under 6 GB; run it alongside on a shared box.

### Verify in the serve log

```
'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>
```
and `free -g` after load: ~83 GB used, the 102 GB table absent. A cold
`Cached` figure that grows as requests run is the page cache filling with
hot rows.

## Limitations

- BF16 / FP16 tables only (the NVFP4 checkpoint). The official FP8
  checkpoint stores an FP8 table with one global `weight_scale`; the mmap
  path does not keep that scale yet.
- `--safetensors-load-strategy eager` reads whole files into RAM and
  defeats the point; keep the default lazy loader.
- `--load-format dummy` has no shards to map.
- Cold rows cost an NVMe read (~0.1 ms each, overlapped 16-way): the first
  pass over a new prompt region is slower than the second. Measured cold
  gathers on this NVMe: 64 rows 3.7 ms, 20k rows 33–60 ms.
- With patch 0001 alone: PIECEWISE cudagraphs only, and the draft (MTP)
  model runs its decode steps eagerly. Patch 0002 removes both limits.
