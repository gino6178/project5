#!/usr/bin/env python
"""Smoke-test the free-space nodes end to end before training on them."""
import os, sys, time
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np, torch

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.generative.train import RetargetPairs, _collate_fn
from reroom.generative.refiner import GraphRefinementTransformer, violation_features

scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
tr, _, _ = split_scenes(scenes)

t0 = time.time()
ds = RetargetPairs(tr[:64], (1, 2, 3), None, seed=0, cache=True)
items = [ds[i] for i in range(16)]
print(f"dataset: 16 items in {time.time()-t0:.1f}s  ({(time.time()-t0)/16*1000:.0f} ms/item)")

b = _collate_fn(items)
dev = "cuda" if torch.cuda.is_available() else "cpu"
b = {k: v.to(dev) for k, v in b.items()}
print("floor      ", tuple(b["floor"].shape), "adj", tuple(b["floor_adj"].shape),
      "| edges/room", float(b["floor_adj"].sum(dim=(1, 2)).mean() / 2),
      "| covering r", float(b["floor_r"].mean()))

m = GraphRefinementTransformer(384, 6, 8).to(dev)
t0 = time.time(); x = m(b); torch.cuda.synchronize() if dev == "cuda" else None
print(f"forward    {tuple(x.shape)} in {time.time()-t0:.2f}s, "
      f"params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

# does the "nearest free floor" vector actually point at free floor?
v = violation_features(b["cond"][..., 10:14], b, m.gap)
tf = v[..., 9:11]
mk = b["mask"].float()
print(f"to_free    mean |offset| {float((tf.norm(dim=-1)*mk).sum()/mk.sum()):.3f} m")

fp, pts = b["floor_pts"], b["cond"][..., 10:12] * b["frame_h"][:, None, :]
d = ((pts[:, :, None, :] - fp[:, None, :, :]) ** 2).sum(-1).sqrt()
tgt = pts + tf
d2 = ((tgt[:, :, None, :] - fp[:, None, :, :]) ** 2).sum(-1).sqrt()
print(f"           dist to nearest node: before {float((d.amin(-1)*mk).sum()/mk.sum()):.3f} m"
      f" -> at target {float((d2.amin(-1)*mk).sum()/mk.sum()):.3f} m (should be ~0)")
print("DONE_FLOOR")
