#!/usr/bin/env python
"""Is the training target itself teaching the failure?

Run 6 charges the physical terms only for exceeding the ground-truth layout in
the same room. On the forward path that ground truth is not a real layout: it is
motif_rigid_warp's synthetic warp of the reference into the deformed room. If
that warp leaves objects outside a concave room, then the reference LICENSES
exactly the out-of-floor behaviour we lose on -- the model is being told that
being that far out is fine, in precisely the rooms where it is not.

Reports the reference term per deformation level. Levels 1-3 are convex
(scale, aspect, slant); level 4 is corner_cut and level 5 is multi-cut, i.e. the
concave ones that match the L and T shapes in the head-to-head.
"""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np, torch

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.generative.train import _collate_fn
from reroom.generative.xscene import HybridPairs, build_pair_index_filtered
from reroom.generative.walkable import boundary_outside, object_reachability, walkability

scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
tr, _, _ = split_scenes(scenes)
tr = tr[:400]
idx = build_pair_index_filtered(tr, thresh=0.6, max_deg=30.0, max_partners=16, seed=0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

print(f"  {'level':<22}{'n':>5}{'ref_bnd':>10}{'GT out%':>10}{'ref_rch':>10}{'ref_walk':>10}")
for lv in (1, 2, 3, 4, 5):
    ds = HybridPairs(tr, idx, forward_frac=1.0, levels=(lv,), l1_range=(0.6, 1.7),
                     l1_u_shape=False, max_deg=30.0, seed=0, cache=False)
    items = []
    for i in range(64):
        try: items.append(ds[i])
        except Exception: pass
    if not items: continue
    b = {k: v.to(dev) for k, v in _collate_fn(items).items()}
    gt, mk = b["state"], b["mask"].float()
    with torch.no_grad():
        d = boundary_outside(gt, b)
        bnd = float((d * mk).sum() / mk.sum())
        out = float(((d > 1e-3).float() * mk).sum() / mk.sum())
        hit = object_reachability(gt, b, G=48, robot=0.3, sharp=20.0, query=1.5)[0]
        rch = float(((1.0 - hit) * mk).sum() / mk.sum())
        wlk = float(walkability(gt, b, G=32)[0].mean())
    name = {1: "1 uniform_scale", 2: "2 aspect_deform", 3: "3 slant_wall",
            4: "4 corner_cut (concave)", 5: "5 multi-cut (concave)"}[lv]
    print(f"  {name:<22}{len(items):>5}{bnd:>10.4f}{100*out:>9.1f}%{rch:>10.4f}{wlk:>10.4f}")
print("\nref_bnd is the metres-outside the run 6 objective treats as acceptable")
print("DONE_GT")
