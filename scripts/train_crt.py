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
# Run 4 targets the two axes run 3 lost on. Run 3 beat PhyScene on collision in
# all five shapes (0.207-0.518 vs 0.429-0.556, four of them under the real-layout
# baseline of 0.400) but lost reach 4-1 and out-of-floor on the concave shapes.
# Both losses have the same cause: containment and reachability entered the model
# only as features, never as an objective, and the walkability loss optimised
# blocked *area* while PhyScene reports a *per-object* reach rate. w_bound and
# w_reach charge both at the final block, against the metric's own definition.
#
# Run 2 collapsed to the identity: it moved objects 0.048 m from the affine
# transplant it starts at and scored exactly like the affine warp (S_rel 0.934,
# S_motif 1.000, R_col 4.69%). The cause is in the data — RetargetPairs builds the
# reference by deforming a real room, so the affine transplant is already close to
# the target and the reconstruction gradient is weak. The correction signal has to
# come from the feasibility terms, so they are raised sharply. That was unsafe
# before, when a category-blind energy fought reconstruction; with the learned
# per-pair spacing (trained on real layouts) the two objectives now agree.
# Run 5 rebalances run 4, which failed by epoch 4 and never recovered: it drove
# collision down (term 0.98 -> 0.13) by scattering objects 1.18 m, which made
# BOTH target axes worse (bnd 0.074 -> 0.355) and tore the design apart
# (rel 0.027 -> 0.273, recon 0.017 -> 0.404). Adding w_bound and w_reach without
# touching the other side had raised "spread out" pressure to 7.0 against 2.0 of
# "stay faithful", where run 3's working balance was 4.0 against 2.0.
#
# Two corrections. Collision needs no more pressure -- run 3 already beat
# PhyScene on it in all five shapes (0.207-0.518 vs 0.429-0.556, four of them
# below the real-layout baseline of 0.400) -- so w_terminal drops back to 1.0 and
# w_walk to 0.5, the latter also because blocked area is the wrong quantity and
# w_reach now covers what it was standing in for. And faithfulness has to scale
# with the added feasibility pressure, so w_relation goes to 3.0. That leaves
# 5.0 against 4.0, with the weight concentrated on the two axes actually being
# fixed rather than on the one already won.
cfg = CRTConfig(epochs=60, batch=128, lr=3.0e-4, d_model=384, n_blocks=6, heads=8,
                w_recon=1.0, w_terminal=1.0, w_monotone=0.5, w_relation=3.0,
                w_walk=0.5, w_gap=1.0, weight_decay=3.0e-4,
                w_bound=2.0, w_reach=1.5, walk_G=48, robot=0.3,
                workers=0, out="outputs/grt5")
train_crt(tr, va, cfg, elasticity=el)
print("[crt] DONE", flush=True)
