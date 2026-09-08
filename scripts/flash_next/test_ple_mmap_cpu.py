#!/usr/bin/env python3
"""CPU-only check of the ple-ssd gather against a dense reference table.

Builds a Qwen4ExpNGramEmbedding in mmap mode with a small synthetic table
(128 shards of 1000 rows), then verifies gather_rows (dedup, shard runs,
chunking, boundary rows) and stage_rows/forward (device_rows staging)
match dense[ids]. Needs only the vLLM venv, no GPU.
"""
import types

import torch

import vllm  # noqa: F401  (package init before deep imports)
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding

cfg = types.SimpleNamespace(
    ngram_size=3, heads_per_ngram=8, eos_token_id=248044, vocab_size=248320,
    split_ngram_parts=128, seed=1234, ngram_vocab_size_base=20_000_000,
    make_ngram_vocab_size_divisible_by=128,
)
emb = Qwen4ExpNGramEmbedding(
    cfg, 2560, 0, max_total_tokens=64, max_num_reqs=4, prefix="p",
    layer_name="p", params_dtype=torch.bfloat16, mmap_table=True,
)
assert emb.padded_vocab_size == 320_001_536 and emb.shard_size == 2_500_012
emb.shard_size, emb.padded_vocab_size = 1000, 128_000
dense = torch.randn(128_000, emb.head_dim, dtype=torch.bfloat16)
emb.shards = [dense[i * 1000 : (i + 1) * 1000].clone() for i in range(128)]

ids = torch.randint(0, 128_000, (64, emb.ngram_heads))
ids[3] = ids[5]
assert torch.equal(emb.gather_rows(ids), dense[ids])
edge = torch.tensor([[0, 999, 1000, 1999, 127_000, 127_999, 64_000, 5]])
assert torch.equal(emb.gather_rows(edge), dense[edge])

# Shrink the per-head hash layout so real compute_ngram_ids lands in the table.
emb.ngram_heads_vocab_sizes.fill_(7999)
emb.ngram_heads_offsets.copy_(torch.arange(emb.ngram_heads) * 8000)
tokens = torch.randint(0, 248320, (7,))
qsl = torch.tensor([0, 3, 7])
ctx = torch.full((2, 2), 248044)
emb.stage_rows(tokens, qsl, ctx)
ngram_ids = emb.compute_ngram_ids(tokens, qsl, ctx)
assert torch.equal(emb.forward(tokens, qsl, ctx), dense[ngram_ids].flatten(-2))
print("ple-ssd CPU test: ok")
