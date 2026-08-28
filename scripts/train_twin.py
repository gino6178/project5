#!/usr/bin/env python
"""Matched-protocol global-coordinate TWIN of flow_bfresh (removes the training
confound the reviewer flagged): identical config to the shipped hierarchical
model — fresh from scratch, 180 epochs, same data/gates/losses — differing ONLY
in parent_relative=False. This isolates the hierarchical reparameterization."""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.intent.elasticity import load_elasticity
from reroom.generative.train import TrainConfig, train_flow

scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
tr, va, te = split_scenes(scenes)
el = load_elasticity("outputs/elasticity/neural.pt")
print(f"[twin] scenes={len(scenes)} train={len(tr)}", flush=True)

cfg = TrainConfig(
    epochs=180, batch=192, lr=3.0e-4, weight_decay=3.0e-4,
    depth=12, d_model=384, heads=8, out="outputs/flow_twin",   # fresh, no init_from
    use_hybrid=True, hybrid_forward_frac=1.0, hybrid_max_deg=30.0, hybrid_jaccard=0.6,
    l1_range=(0.6, 1.7), l1_u_shape=False,
    prior_x0=True, prior_noise=0.3,
    geo_bias=True, wall_tokens=True,
    parent_relative=False,                 # <<< the ONLY difference vs flow_bfresh
    rel_loss=4.0, wall_align_loss=1.0, wall_aux=3.0,
    log_every=100,
)
train_flow(tr, va, cfg, elasticity=el, bank=None)
print("[twin] DONE", flush=True)
