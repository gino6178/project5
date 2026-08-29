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

scenes = list(iter_scenes())
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
for name, x in (("ground truth", gt), ("half scale (crowded)", gt * 0.5),
                ("piled at centre", gt * 0.02)):
    h = object_reachability(x, b, G=48, robot=0.3)[0]
    print(f"  {name:<22} {float((h * mk).sum() / mk.sum()):.3f}")
