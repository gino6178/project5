#!/usr/bin/env python
"""B / real deep contribution: differentiable entropic GROMOV-WASSERSTEIN
relational loss folded into training.  Warm-start from flow_bfresh and fine-tune
with gw_loss>0 (identical protocol otherwise).  Trains global relational
structure via a true OT coupling; evaluated on INDEPENDENT metrics (S_motif, OOD
cross-pairing) so the gain is not the training objective restated."""
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
print(f"[gw] scenes={len(scenes)} train={len(tr)}", flush=True)

cfg = TrainConfig(
    epochs=60, batch=192, lr=1.0e-4, weight_decay=3.0e-4,
    init_from="outputs/flow_bfresh/flow_best.pt", out="outputs/flow_gw",
    depth=12, d_model=384, heads=8,
    use_hybrid=True, hybrid_forward_frac=1.0, hybrid_max_deg=30.0, hybrid_jaccard=0.6,
    l1_range=(0.6, 1.7), l1_u_shape=False,
    prior_x0=True, prior_noise=0.3,
    geo_bias=True, wall_tokens=True, parent_relative=True,
    child_loss_weight=10.0, rel_loss=4.0, wall_align_loss=1.0, wall_aux=3.0,
    ema=0.999, grad_clip=1.0, log_every=50, workers=0,
    # --- THE change: true GW relational loss ---
    gw_loss=2.0, gw_loss_tau=0.4, gw_eps=0.05, gw_iters=10,
)
train_flow(tr, va, cfg, elasticity=el, bank=None)
print("[gw] DONE", flush=True)
