# Flash-Next runs on the GB10 (2026-09-03)

All runs: Inferact/Qwen3.8-Flash-Next-NVFP4, vLLM 33898f832c + patches,
`serve.sh` defaults (`--language-model-only`, 0.76 utilization, 4 seqs,
64k context), `--hf-overrides '{"ple_mmap": true}'`, memguard active.

| dir | patches | MTP | draft head | cudagraphs |
|---|---|---|---|---|
| `mtp3` | ple-ssd 0001, mtp-pruning | k=3 | full | PIECEWISE (forced by 0001) |
| `fullgraph-mtp3` | + ple-ssd 0002 | k=3 | full | FULL_AND_PIECEWISE |
| `fullgraph-mtp3-pruned` | + ple-ssd 0002 | k=3 | 10,010-id keep-set (`keep_flashnext_p99.pt`) | FULL_AND_PIECEWISE |
| `fullgraph-mtp0` | + ple-ssd 0002 | off | – | FULL_AND_PIECEWISE |

Per dir: `server.log`, `bench_c1.json` / `bench_c4.json` (bench_decode.py:
per-request decode tok/s from streaming timestamps, MTP acceptance from
/metrics), `gsm8k.json` (200 test questions, zero-shot, thinking off,
greedy, 1024-token cap; per-question records), `metrics-after.prom`,
`trace/` (torch profiler, git-ignored). `mtp3/generations.jsonl`,
`counts.npy`, `keep_fitted.pt` are the keep-set fit inputs.
