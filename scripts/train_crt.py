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
cfg = CRTConfig(epochs=60, batch=128, lr=2.0e-4, d_model=384, n_blocks=6, heads=8,
                w_recon=1.0, w_terminal=0.4, w_monotone=0.2, w_relation=2.0,
                w_walk=1.0, weight_decay=3.0e-4,
                workers=0, out="outputs/crt_walk")
train_crt(tr, va, cfg, elasticity=el)
print("[crt] DONE", flush=True)
