#!/usr/bin/env python
"""Build the corpus-derived priors the retargeter needs.

* an asset bank -- real 3D-FUTURE assets when the models are on disk, otherwise
  a statistical bank of per-category sizes harvested from the corpus;
* a co-occurrence model -- which categories belong to which room type and in
  what numbers, used when a larger target room has to be populated (eq. 29).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from reroom.data.asset_bank import FutureBank, StatisticalBank
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.retarget.populate import CooccurrenceModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/priors")
    ap.add_argument("--future-root", default=None,
                    help="extracted 3D-FUTURE-model directory")
    ap.add_argument("--future-bboxes", default=None)
    ap.add_argument("--future-embeddings", default=None)
    ap.add_argument("--future-shapes", default=None)
    ap.add_argument("--sage-root", default=None,
                    help="SAGE-10k layouts: open-vocabulary asset augmentation (17)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    scenes = list(iter_scenes(a.corpus, limit=a.limit or None, min_objects=4))
    train, _, _ = split_scenes(scenes)
    print(f"{len(scenes)} scenes ({len(train)} train)")

    if a.future_root and os.path.isdir(a.future_root):
        bank = FutureBank.from_dir(a.future_root, bbox_cache=a.future_bboxes)
        if a.future_embeddings and os.path.exists(a.future_embeddings):
            bank.attach_embeddings(a.future_embeddings)
        if a.future_shapes and os.path.exists(a.future_shapes):
            bank.attach_shapes(a.future_shapes)
            n = sum(1 for x in bank.assets if x.shape is not None)
            print(f"f^geo descriptors on {n}/{len(bank)} assets")
        print(f"3D-FUTURE bank: {len(bank)} assets over {len(bank.categories())} categories")
    else:
        bank = StatisticalBank.from_scenes(train, per_category=200)
        print(f"statistical bank: {len(bank)} pseudo-assets over "
              f"{len(bank.categories())} categories")
    if a.sage_root and os.path.isdir(a.sage_root):
        from reroom.data.asset_bank import merge_banks
        from reroom.data.sage import iter_sage_scenes, sage_style_text
        sage = list(iter_sage_scenes(a.sage_root, min_objects=5))
        if sage:
            extra = StatisticalBank.from_scenes(
                sage, per_category=64, source="sage", style_from=sage_style_text)
            before = set(bank.categories())
            bank = merge_banks(bank, extra)
            gained = sorted(set(bank.categories()) - before)
            print(f"SAGE augmentation: {len(sage)} rooms -> {len(bank)} assets, "
                  f"{len(gained)} new categories {gained[:8]}")
    bank.save(os.path.join(a.out, "assets.pkl"))

    cooc = CooccurrenceModel.fit(train)
    with open(os.path.join(a.out, "cooc.json"), "w") as fh:
        json.dump({"counts": {k: dict(v) for k, v in cooc.counts.items()},
                   "sizes": {k: v.tolist() for k, v in cooc.sizes.items()},
                   "n_scenes": cooc.n_scenes}, fh)
    for rt in sorted(cooc.counts):
        n = cooc.n_scenes[rt]
        top = sorted(cooc.counts[rt].items(), key=lambda t: -t[1])[:8]
        print(f"  {rt:12s} ({n:5d} scenes): " +
              ", ".join(f"{c}x{k / n:.1f}" for c, k in top))


if __name__ == "__main__":
    main()
