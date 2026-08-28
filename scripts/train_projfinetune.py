#!/usr/bin/env python
"""A / Prop 3 — TRAIN-THROUGH differentiable projection.

Warm-start from the shipped flow_bfresh and fine-tune with the differentiable
projection folded INTO training (proj_loss>0): gradients flow through Pi_theta's
unrolled steps, so the flow learns to emit feasible-by-construction endpoints on
which the deployed projection is a near-no-op.  Identical to flow_bfresh in every
other respect, so the comparison isolates the projection-training effect. This
realises Prop 3 (previously test-time-only) and gives Pi_theta a measurable,
non-confounded benefit over raw flow (reviewer's #1 concern)."""
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
print(f"[proj] scenes={len(scenes)} train={len(tr)}", flush=True)

cfg = TrainConfig(
    # --- fine-tune schedule (warm-started, so shorter + gentler lr) ---
    epochs=60, batch=192, lr=1.0e-4, weight_decay=3.0e-4,
    init_from="outputs/flow_bfresh/flow_best.pt", out="outputs/flow_proj",
    # --- identical to flow_bfresh ---
    depth=12, d_model=384, heads=8,
    use_hybrid=True, hybrid_forward_frac=1.0, hybrid_max_deg=30.0, hybrid_jaccard=0.6,
    l1_range=(0.6, 1.7), l1_u_shape=False,
    prior_x0=True, prior_noise=0.3,
    geo_bias=True, wall_tokens=True, parent_relative=True,
    child_loss_weight=10.0, rel_loss=4.0, wall_align_loss=1.0, wall_aux=3.0,
    ema=0.999, grad_clip=1.0, log_every=50, workers=0,
    # --- THE change: train-through differentiable projection (Prop 3) ---
    # loss = post-projection geometric energy through a 15-step anchored
    # (topology-preserving) differentiable projection.  Weight kept moderate so
    # it shapes feasibility without swamping the flow-matching objective
    # (fm~0.04; postE starts ~0.1-0.2 → contribution ~0.2-0.4).
    proj_loss=2.0, proj_iters=15, proj_step=0.2, proj_anchor=2.0, proj_loss_tau=0.5,
)
train_flow(tr, va, cfg, elasticity=el, bank=None)
print("[proj] DONE", flush=True)
