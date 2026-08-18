# SPDX-License-Identifier: Apache-2.0
"""Build a draft keep-set file from token-frequency counts.

TorchSpec-style selection: the smallest top-K (by count) reaching the
requested coverage of total occurrences, unioned with the tokenizer's added
special tokens (chat/tool markup must never be undraftable), sorted
ascending, saved as a torch int64 tensor — the format
``draft_vocab_path`` (in --speculative-config) expects.

    python scripts/mtp_pruning/build_keepset.py --counts counts.npy --coverage 0.99 \
        --tokenizer /path/to/model --out keep_p99.pt

``--counts`` accepts a .npy 1-D array indexed by token id, or a .npz whose
first (or ``--counts-key``) array is that. Produce it however you like —
e.g. tokenize a capture of real traffic from your serving stack and
bincount the ids the model generated.
"""

import argparse
import json
import os

import numpy as np
import torch


def _load_counts(path: str, key: str | None) -> np.ndarray:
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        key = key or list(d.keys())[0]
        arr = d[key]
    else:
        arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr.sum(0)
    if arr.ndim != 1:
        raise SystemExit(f"counts must be 1-D (or 2-D summed), got {arr.shape}")
    return arr.astype(np.int64)


def _special_ids(tokenizer_dir: str) -> np.ndarray:
    cfg = os.path.join(tokenizer_dir, "tokenizer_config.json")
    with open(cfg) as fh:
        added = json.load(fh).get("added_tokens_decoder", {})
    return np.array(sorted(int(i) for i in added), dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counts", required=True, help=".npy/.npz of per-id counts")
    ap.add_argument("--counts-key", default=None, help="array key inside a .npz")
    ap.add_argument("--coverage", type=float, default=0.99,
                    help="fraction of occurrences the keep-set must cover")
    ap.add_argument("--tokenizer", default=None,
                    help="model/tokenizer dir; its added special tokens are "
                         "always included")
    ap.add_argument("--extra-ids", default=None,
                    help="comma-separated ids to force-include")
    ap.add_argument("--out", required=True, help="output .pt path")
    args = ap.parse_args()

    counts = _load_counts(args.counts, args.counts_key)
    total = counts.sum()
    if total <= 0:
        raise SystemExit("counts are empty")
    if not (0.0 < args.coverage <= 1.0):
        raise SystemExit("--coverage must be in (0, 1]")

    order = np.argsort(-counts, kind="stable")
    cum = np.cumsum(counts[order]) / total
    k = int(np.searchsorted(cum, args.coverage - 1e-12) + 1)
    keep = order[:k]

    forced = []
    if args.tokenizer:
        forced.append(_special_ids(args.tokenizer))
    if args.extra_ids:
        forced.append(np.array([int(x) for x in args.extra_ids.split(",")],
                               dtype=np.int64))
    for f in forced:
        keep = np.union1d(keep, f)
    keep = np.unique(keep).astype(np.int64)

    cov = counts[keep].sum() / total
    torch.save(torch.from_numpy(keep), args.out)
    print(f"top-{k} for {args.coverage:.4%} target; |keep|={keep.size} "
          f"({keep.size / counts.size:.2%} of the {counts.size}-id vocab), "
          f"achieved coverage {cov:.4%} -> {args.out}")


if __name__ == "__main__":
    main()
