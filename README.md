# magpie-vllm-plugins

The magpie team's vLLM optimizations for DGX Spark (GB10) serving.

Each optimization ships in the lightest form that reaches every vLLM
process, and is **inert until activated** through vLLM's own arguments:

- **patch** — a git-apply-able change over the vLLM checkout, for features
  that must live inside vLLM (mtp-pruning). Unactivated, patched files
  behave stock.
- **plugin** — a pip-installable package hooking `vllm.general_plugins`,
  for features vLLM's extension points can carry with zero patched lines
  (sparse-attention). Unactivated, registered classes construct stock.

## Catalog

| Optimization | What it does | Measured gain | Docs |
|---|---|---|---|
| **mtp-pruning** (patch) | Prunes the Qwen3.5 MTP draft vocabulary to a frequency keep-set, slicing the 2.5 GB shared draft `lm_head` to a few % of its rows. Lossless output. | draft-head 45 → ~1 ms/step; decode 1.28–1.35× at c=1–8; acceptance −1% | [docs/mtp-pruning.md](docs/mtp-pruning.md) |
| **sparse-attention** (plugin) | Vortex per-KV-head block-sparse decode for Qwen3.5 full-attention layers: per-block mean-K centroids + fused top-k select 4–27% of KV per step; FULL cudagraph decode. Zero vLLM lines patched. | attention GPU 3.1×; decode step flat vs context (1.14× vs stock @160K resident KV, growing with occupancy); RULER 16K 0.983 vs 1.000 (random-control 0.000) | [docs/sparse-attention.md](docs/sparse-attention.md) |

**Known interaction:** the two are not composable yet — MTP spec decode
makes `decode_query_len > 1`, which routes decode rows past the sparse path
(correct output, sparsity inert). See docs/sparse-attention.md.

## Installation

Prerequisites: a vLLM source checkout you serve from (patches are maintained
against `vllm-project/vllm` @ `6e448d0ea9`; nearby commits usually apply
cleanly — `git apply --check` tells you), and its Python environment.

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

## Issues

Known issues, measured evidence, and fix sketches: [ISSUES.md](ISSUES.md).

## Layout

```
patches/<optimization>/NNNN-*.patch   # patch-form: the vLLM change, git-apply-able
plugins/<optimization>/               # plugin-form: pip package (pyproject + code + tests)
scripts/<optimization>/*.py           # plain workflow scripts (stdlib + vLLM-env deps)
docs/<optimization>.md                # mechanism, workflow, measured results, limits
```

Plugin install:

```bash
uv pip install -e plugins/sparse-attention --python /path/to/vllm-venv/bin/python
```

Adding an optimization means adding one row to the catalog and one entry in
each of those three places.
