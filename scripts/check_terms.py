#!/usr/bin/env python
"""Sanity-check the two new objective terms before spending a training run.

A relaxed quantity is only useful if it moves the way the metric it stands in for
moves. The dilated jump-flooding bug -- a severed room reading as 0.048 blocked
instead of 0.43 -- was found this way, so both new terms are checked against
layouts whose answer is known before they are trusted in a loss.
"""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import torch

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.generative.train import RetargetPairs, _collate_fn
from reroom.generative.walkable import boundary_outside, object_reachability

scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)]
tr, _, _ = split_scenes(scenes)
ds = RetargetPairs(tr[:64], (1, 2, 3), None, seed=0, cache=True)
b = _collate_fn([ds[i] for i in range(16)])
dev = "cuda" if torch.cuda.is_available() else "cpu"
b = {k: v.to(dev) for k, v in b.items()}
gt, mk = b["state"], b["mask"].float()
fh = b["frame_h"].mean()


def push(x, metres):
    y = x.clone(); y[..., 0] = y[..., 0] + metres / fh
    return y


print("boundary_outside -- metres the worst corner pokes outside the room")
for name, x in (("ground truth", gt), ("+0.5 m", push(gt, 0.5)),
                ("+1.5 m", push(gt, 1.5)), ("+3.0 m", push(gt, 3.0))):
    v = boundary_outside(x, b)
    print(f"  {name:<14} {float((v * mk).sum() / mk.sum()):.3f}")

print("object_reachability -- 1 = reachable floor touches the object")
print("  real 3D-FRONT layouts score ~0.94 on the metric this stands in for")
print(f"  {'G/sharp/dilate':<18}{'ground truth':>14}{'half scale':>12}{'piled':>9}")
for G in (32, 48, 64):
    for sharp in (8.0, 12.0, 20.0):
        for dil in (1, 2, 3):
            vals = []
            for x in (gt, gt * 0.5, gt * 0.02):
                h = object_reachability(x, b, G=G, robot=0.3, sharp=sharp,
                                        dilate=dil)[0]
                vals.append(float((h * mk).sum() / mk.sum()))
            print(f"  {f'{G}/{sharp:g}/{dil}':<18}" +
                  "".join(f"{v:>13.3f} " for v in vals))
