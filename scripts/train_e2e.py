#!/usr/bin/env python
"""project5 — END-TO-END joint training (DESIGN.md).

One differentiable function, one objective: the flow proposes, the learned
existence head selects, and the unrolled projection Pi_theta closes the layout —
with the reconstruction loss applied to the PROJECTED result x* = Pi(x1_hat, keep)
rather than the raw endpoint. Selection, placement and feasibility are therefore
optimised against the objective we actually deploy.

Trained FROM SCRATCH on purpose. project4's two bolt-on attempts (train-through
proj_loss, GW relational loss) fine-tuned a converged staged model while still
supervising the raw endpoint, and both measured a clean null; a converged model
has no reason to move. Everything except the e2e block matches flow_bfresh, so
project4's evaluation scripts apply unchanged and the numbers stay comparable.

First-run check: print the epoch-0 row and confirm e2e_recon is the same order as
train_loss. If it dominates, lower e2e_recon (the staged experiments showed an
over-weighted geometric term degrades the flow-matching objective).
"""
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
print(f"[e2e] scenes={len(scenes)} train={len(tr)}", flush=True)

cfg = TrainConfig(
    # --- identical protocol to project4's flow_bfresh (fresh, no init_from) ---
    epochs=180, batch=192, lr=3.0e-4, weight_decay=3.0e-4,
    depth=12, d_model=384, heads=8, out="outputs/flow_e2e",
    use_hybrid=True, hybrid_forward_frac=1.0, hybrid_max_deg=30.0, hybrid_jaccard=0.6,
    l1_range=(0.6, 1.7), l1_u_shape=False,
    prior_x0=True, prior_noise=0.3,
    geo_bias=True, wall_tokens=True, parent_relative=True,
    child_loss_weight=10.0, rel_loss=4.0, wall_align_loss=1.0, wall_aux=3.0,
    ema=0.999, grad_clip=1.0, log_every=50,
    workers=0,          # remote /dev/shm is small and shared; in-memory cache instead

    # --- the end-to-end change ---
    e2e=True,
    e2e_recon=1.0,      # reconstruction measured THROUGH the projection
    e2e_resid=0.5,      # residual infeasibility charged back to the flow
    e2e_tau=0.5,        # only where the endpoint estimate is usable
    mask_flow=True,     # existence head becomes the selection mechanism
    mask_loss=1.0,
    proj_iters=15, proj_step=0.2, proj_anchor=2.0,
)
train_flow(tr, va, cfg, elasticity=el, bank=None)
print("[e2e] DONE", flush=True)
