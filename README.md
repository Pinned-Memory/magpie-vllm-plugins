# magpie-vllm-plugins

The magpie team's vLLM optimizations for DGX Spark (GB10) serving.

Each optimization is a **patch over the vLLM checkout** plus plain
supporting **scripts** — no plugins, no entry points, no env vars, nothing
to pip-install. Apply the patch once, drive the feature through vLLM's own
arguments. Unactivated, every patched file behaves stock.

## Catalog

| Optimization | What it does | Measured gain | Docs |
|---|---|---|---|
| **mtp-pruning** | Prunes the Qwen3.5 MTP draft vocabulary to a frequency keep-set, slicing the 2.5 GB shared draft `lm_head` to a few % of its rows. Lossless output. | draft-head 45 → ~1 ms/step; decode 1.28–1.35× at c=1–8; acceptance −1% | [docs/mtp-pruning.md](docs/mtp-pruning.md) |

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

## Layout

```
patches/<optimization>/NNNN-*.patch   # the vLLM change, numbered, git-apply-able
scripts/<optimization>/*.py           # plain workflow scripts (stdlib + vLLM-env deps)
docs/<optimization>.md                # mechanism, workflow, measured results, limits
```

Adding an optimization means adding one row to the catalog and one entry in
each of those three places.
