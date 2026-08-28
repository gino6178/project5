#!/usr/bin/env python
"""Train the graph-conditioned flow-matching layout proposal (section 13)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.generative.train import TrainConfig, train_flow
from reroom.intent.elasticity import load_elasticity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/flow")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-objects", type=int, default=24)
    ap.add_argument("--min-objects", type=int, default=5)
    ap.add_argument("--val-scenes", type=int, default=400)
    ap.add_argument("--elasticity", default="outputs/elasticity/neural.pt")
    ap.add_argument("--sage-root", default=None,
                    help="SAGE-10k layouts to mix into training (section 17): "
                         "object diversity and clutter, not room geometry")
    ap.add_argument("--sage-weight", type=float, default=1.0,
                    help="how many times to repeat the SAGE half")
    ap.add_argument("--hybrid", action="store_true",
                    help="use the cross-scene / motif-rigid HybridPairs pipeline "
                         "(real targets) instead of affine-warp RetargetPairs")
    ap.add_argument("--forward-frac", type=float, default=0.7,
                    help="HybridPairs: fraction of forward-deform pairs (rest "
                         "are filtered cross-pairing)")
    ap.add_argument("--init-from", default="",
                    help="warm-start checkpoint (copies shape-matching layers)")
    a = ap.parse_args()

    scenes = [s for s in iter_scenes(a.corpus, limit=a.limit or None,
                                     min_objects=a.min_objects)
              if len(s.objects) <= a.max_objects]
    train, val, _ = split_scenes(scenes)
    val = val[:a.val_scenes]
    print(f"{len(train)} train / {len(val)} val scenes", flush=True)

    if a.sage_root and os.path.isdir(a.sage_root):
        # Section 17, month 5.  SAGE rooms are almost all rectangles, so they
        # teach nothing about irregular geometry -- they are here for the thing
        # the plan actually assigns them: object diversity and clutter that
        # 3D-FRONT does not contain.  Validation stays pure 3D-FRONT so the
        # numbers remain comparable with every earlier run.
        from reroom.data.sage import iter_sage_scenes
        extra = [s for s in iter_sage_scenes(a.sage_root,
                                             min_objects=a.min_objects)
                 if len(s.objects) <= a.max_objects]
        reps = max(int(round(a.sage_weight)), 1) if extra else 0
        train = train + extra * reps
        cats = {o.category for s in extra for o in s.objects}
        print(f"+ {len(extra)} SAGE rooms x{reps} "
              f"({len(cats)} categories) -> {len(train)} train scenes",
              flush=True)

    el = load_elasticity(a.elasticity) if os.path.exists(a.elasticity) else None
    cfg = TrainConfig(epochs=a.epochs, batch=a.batch, lr=a.lr,
                      workers=a.workers, device=a.device, depth=a.depth,
                      d_model=a.d_model, out=a.out,
                      use_hybrid=a.hybrid, hybrid_forward_frac=a.forward_frac,
                      init_from=a.init_from)
    train_flow(train, val, cfg, elasticity=el)
    print("done ->", os.path.join(a.out, "flow.pt"))


if __name__ == "__main__":
    main()
