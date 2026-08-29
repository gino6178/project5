#!/usr/bin/env python
"""Attribute the gradient, before blaming the weights.

Run 5 holds relations (rel 0.033) and recovers val_recon, but bnd will not fall
below the affine transplant it starts at and recon sits at 0.34 where run 3
reached 0.04 -- the shape of a gradient problem, not a weighting one. The
suspect is the straight-through hard gate in the reachability path: it is read
as a FEATURE at each of six blocks, so its gradient enters the graph six times,
and with grad_clip=1.0 anything that dominates crushes everything else.

Prints the pre-clip gradient norm of each loss term on its own, and the norm with
the violation features detached, which is the fix if the feature path dominates.
"""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import torch

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.generative.train import RetargetPairs, _collate_fn
from reroom.generative.refiner import GraphRefinementTransformer
from reroom.generative.graphcore import gap_supervision, graph_violation
from reroom.generative.walkable import boundary_outside, object_reachability, walkability

scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
tr, _, _ = split_scenes(scenes)
ds = RetargetPairs(tr[:96], (1, 2, 3), None, seed=0, cache=True)
b = _collate_fn([ds[i] for i in range(32)])
dev = "cuda" if torch.cuda.is_available() else "cpu"
b = {k: v.to(dev) for k, v in b.items()}
mk = b["mask"].float()


def norm_of(model, term_fn):
    model.zero_grad(set_to_none=True)
    term_fn().backward()
    tot = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    return tot ** 0.5


for use_walk_feat in (True, False):
    torch.manual_seed(0)
    m = GraphRefinementTransformer(384, 6, 8, use_walk=use_walk_feat).to(dev)
    label = "features WITH walk/reach grad" if use_walk_feat else "features WITHOUT (use_walk=False)"
    print(f"\n{label}")
    terms = {
        "recon": lambda: (lambda x: ((x - b["state"]) ** 2 * mk[..., None]).sum()
                          / mk.sum().clamp(min=1))(m(b)),
        "term":  lambda: graph_violation(m(b), b, m.gap)[1].mean(),
        "bnd":   lambda: (boundary_outside(m(b), b) * mk).sum() / mk.sum().clamp(min=1),
        "rch":   lambda: ((1.0 - object_reachability(m(b), b, G=48, robot=0.3,
                                                     sharp=20.0, query=1.5)[0]) * mk).sum()
                         / mk.sum().clamp(min=1),
        "walk":  lambda: walkability(m(b), b, G=32)[0].mean(),
    }
    for k, fn in terms.items():
        try:
            print(f"  {k:<7} grad-norm {norm_of(m, fn):12.4f}")
        except RuntimeError as e:
            print(f"  {k:<7} FAILED {str(e)[:60]}")
