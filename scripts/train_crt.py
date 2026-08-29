#!/usr/bin/env python
"""project5 — train the Constraint-Refinement Transformer (end-to-end, no post-processing).

Same corpus and split as project4 so the evaluation tables stay comparable.
"""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.intent.elasticity import load_elasticity
from reroom.generative.train_crt import CRTConfig, train_crt

scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
tr, va, te = split_scenes(scenes)
el = load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
print(f"[crt] scenes={len(scenes)} train={len(tr)} val={len(va)}", flush=True)

# w_terminal lowered from 1.0: real 3D-FRONT layouts themselves score 0.388 on
# this collision metric (touching / nested furniture), so driving the energy to
# zero necessarily pulls the layout away from the ground truth. The first run
# showed exactly that -- val_recon rose 8x while val_energy kept falling. This is
# an objective conflict, not overfitting, so it is fixed by weighting, not by
# regularisation. w_walk adds the rasterised blocked-walkway penalty.
# Run 2 collapsed to the identity: it moved objects 0.048 m from the affine
# transplant it starts at and scored exactly like the affine warp (S_rel 0.934,
# S_motif 1.000, R_col 4.69%). The cause is in the data — RetargetPairs builds the
# reference by deforming a real room, so the affine transplant is already close to
# the target and the reconstruction gradient is weak. The correction signal has to
# come from the feasibility terms, so they are raised sharply. That was unsafe
# before, when a category-blind energy fought reconstruction; with the learned
# per-pair spacing (trained on real layouts) the two objectives now agree.
cfg = CRTConfig(epochs=60, batch=128, lr=3.0e-4, d_model=384, n_blocks=6, heads=8,
                w_recon=1.0, w_terminal=2.0, w_monotone=0.5, w_relation=1.0,
                w_walk=2.0, w_gap=1.0, weight_decay=3.0e-4,
                workers=0, out="outputs/grt3")
train_crt(tr, va, cfg, elasticity=el)
print("[crt] DONE", flush=True)
