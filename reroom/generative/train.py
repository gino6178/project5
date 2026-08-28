"""Training the graph-conditioned flow-matching proposal (section 13).

There is no corpus of *paired* (reference room -> retargeted room) layouts, so
the supervision is manufactured from real designs by running the retargeting
problem backwards:

1. take a professionally designed scene ``(L, P)`` from 3D-FRONT;
2. sample a curriculum deformation ``P_r = T_delta(P)`` (section 12) and warp
   the layout into it, giving a *pseudo-reference* design ``(L_r, P_r)``;
3. train the model to recover ``L`` in ``P`` given the design intent extracted
   from ``(L_r, P_r)``.

The target is therefore always a real, human-designed layout, and the input is
always the same design seen in a differently shaped room -- exactly the
retargeting task, with free supervision.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from shapely.geometry import Point as _ShPoint

from ..core.scene import Scene, scene_from_dict
from ..geom.deform import deform_room
from ..geom.polygon import as_polygon
from ..geom.polygon import as_polygon
from ..intent.elasticity import ElasticityModel, PriorElasticity
from ..intent.motifs import build_motifs
from ..intent.relations import build_scene_graph
from ..retarget.optimizer import _map_point, _mrr_frame
from ..retarget.target import build_design_intent
from .model import FlowModel
from .tokens import build_tokens, collate

__all__ = ["RetargetPairs", "TrainConfig", "train_flow", "warp_scene"]


def _nestable_matrix(device):
    """(N_CAT, N_CAT) bool: True where the two categories may legitimately
    overlap (chair under table, nightstand by bed) — exempt from collision."""
    from .tokens import CATS
    from .guidance import NESTABLE_PAIRS
    ix = {c: i for i, c in enumerate(CATS)}
    n = len(CATS)
    m = torch.zeros(n, n, dtype=torch.bool, device=device)
    for pair in NESTABLE_PAIRS:
        a, b = tuple(pair)
        if a in ix and b in ix:
            m[ix[a], ix[b]] = True
            m[ix[b], ix[a]] = True
    return m


def _free_vector(device):
    """(N_CAT,) bool: True for genuinely-free categories (plants, lamps,
    nightstands) that the projection must NOT move — matches diffproj.FREE_CATS
    and the category id order used by batch['cat']."""
    from .tokens import CATS
    from ..retarget.regularity import FREE_CATS
    ix = {c: i for i, c in enumerate(CATS)}
    v = torch.zeros(len(CATS), dtype=torch.bool, device=device)
    for c in FREE_CATS:
        if c in ix:
            v[ix[c]] = True
    return v


def _proj_forward(x1_hat, batch, nest_mat, free_vec, keep=None,
                  iters: int = 15, step: float = 0.2, anchor: float = 2.0):
    """END-TO-END forward projection: return the PROJECTED layout x* itself.

    Same anchored unrolled operator as ``_proj_through_energy``, but it hands back
    the projected state rather than its energy, so the reconstruction loss can be
    computed *through* the projection (project5's central change: the model is
    trained so the layout is right AFTER Pi_theta, making the projection part of
    the model instead of a post-hoc repair).

    ``keep`` (B,N) in [0,1] is a differentiable/straight-through existence mask:
    dropped objects are excluded from the collision term, so the learned selection
    and the geometry are optimised against one another rather than separately.
    Gradients reach both x1_hat and keep.
    """
    mf = batch["mask"].float()
    if keep is not None:
        mf = mf * keep
    bat = dict(batch); bat["mask"] = mf          # energy respects the keep mask
    p0 = x1_hat[..., :2]
    ori = x1_hat[..., 2:4]
    free = free_vec[batch["cat"]].float()
    movable = ((1.0 - free) * mf)[..., None]
    q = p0
    for _ in range(iters):
        E = _geo_energy(torch.cat([q, ori.detach()], dim=-1), bat, nest_mat)
        g = torch.autograd.grad(E, q, create_graph=True)[0]
        q = q - step * (g + anchor * (q - p0)) * movable
    return torch.cat([q, ori], dim=-1)


def _proj_through_energy(x1_hat, batch, nest_mat, free_vec,
                         iters: int = 15, step: float = 0.2, anchor: float = 2.0):
    """Train-through differentiable projection (A / Prop 3).

    Runs Pi_theta as `iters` unrolled ANCHORED gradient-descent steps on the
    geometric energy (collision + containment, ``_geo_energy``) starting from the
    flow's predicted endpoint x1_hat.  Each step also pulls toward the proposal
    (anchor term), exactly like the deployed ``project_batch`` — so the operator
    moves objects the *minimal* amount to reach feasibility and freezes genuinely
    free objects; that anchor is what makes this a topology-preserving projection
    rather than a raw energy push.  Every step is autograd-differentiable
    (create_graph), so the returned POST-projection energy back-propagates through
    the whole unrolled operator into the flow (this realises Prop 3, previously
    test-time-only): minimising it trains the flow to emit endpoints on which a
    minimal, topology-preserving projection is already near-feasible, i.e.
    feasible-by-construction.

    Returns (post_energy_loss, pre_energy) — the differentiable geometric energy
    of the projected layout (the training target), and the raw endpoint energy
    (detached, for logging the improvement).
    """
    mf = batch["mask"].float()                              # (B, N)
    p0 = x1_hat[..., :2]                                    # (B,N,2) has grad→model
    ori = x1_hat[..., 2:4].detach()                         # energy ignores ori
    free = free_vec[batch["cat"]].float()                  # (B, N)
    movable = ((1.0 - free) * mf)[..., None]                # (B,N,1) gate
    pre_E = float(_geo_energy(x1_hat.detach(), batch, nest_mat))
    q = p0
    for _ in range(iters):
        xh = torch.cat([q, ori], dim=-1)
        E = _geo_energy(xh, batch, nest_mat)               # scalar, metric, masked
        g = torch.autograd.grad(E, q, create_graph=True)[0]
        # anchored, gated step: descend energy but stay near the proposal;
        # free objects frozen.  Keeps gradient path to p0 through (q - p0).
        q = q - step * (g + anchor * (q - p0)) * movable
    post_E = _geo_energy(torch.cat([q, ori], dim=-1), batch, nest_mat)
    return post_E, pre_E


def _sinkhorn(M, p, q, eps, iters=30):
    """Batched entropic OT (balanced).  M (B,N,N) cost, p/q (B,N) marginals."""
    K = torch.exp(-(M - M.amin(dim=(1, 2), keepdim=True)) / eps)      # (B,N,N)
    u = torch.ones_like(p); v = torch.ones_like(q)
    for _ in range(iters):
        u = p / (torch.bmm(K, v[..., None]).squeeze(-1) + 1e-30)
        v = q / (torch.bmm(K.transpose(1, 2), u[..., None]).squeeze(-1) + 1e-30)
    return u[..., None] * K * v[:, None, :]


def _gw_relational_loss(x1_hat, batch, eps: float = 0.05, gw_iters: int = 10,
                        sink_iters: int = 30):
    """Differentiable entropic GROMOV-WASSERSTEIN relational loss (B / true OT).

    Couples two metric-measure spaces -- the REFERENCE layout (cond ref
    positions, pairwise-distance structure C_ref) and the GENERATED layout
    (x1_hat, structure C_gen) -- with an entropic GW coupling T (soft
    correspondence, learned by Sinkhorn iterations), then penalises the
    relational distortion transported by T.  Structures are per-scene
    scale-normalised so the loss captures *relative* relational shape (design
    identity), invariant to room size.  The coupling is detached (envelope
    theorem): gradients flow through C_gen -> x1_hat, so the flow is trained to
    emit layouts whose relational structure matches the reference's UNDER THE
    OPTIMAL COUPLING -- which, unlike a fixed-index distance loss, degrades
    gracefully under pruning/substitution (soft re-matching).  First-order only.
    """
    mask = batch["mask"].float()                                     # (B,N)
    gen = x1_hat[..., :2]
    ref = batch["cond"][..., 10:12]
    # pairwise distance structures (manual; double-backward safe if ever needed)
    Cg = ((gen[:, :, None, :] - gen[:, None, :, :]) ** 2).sum(-1).clamp(min=1e-12).sqrt()
    Cr = ((ref[:, :, None, :] - ref[:, None, :, :]) ** 2).sum(-1).clamp(min=1e-12).sqrt()
    pair = mask[:, :, None] * mask[:, None, :]                        # (B,N,N) valid
    npair = pair.sum(dim=(1, 2)).clamp(min=1)
    # per-scene scale normalisation (GW compares RELATIVE structure)
    sg = (Cg * pair).sum(dim=(1, 2)) / npair
    sr = (Cr * pair).sum(dim=(1, 2)) / npair
    Cg = Cg / sg[:, None, None].clamp(min=1e-6)
    Cr = Cr / sr[:, None, None].clamp(min=1e-6)
    # marginals: uniform over present objects
    p = mask / mask.sum(1, keepdim=True).clamp(min=1)
    q = p
    # ---- entropic GW coupling (detached) ----
    with torch.no_grad():
        Crd, Cgd = Cr.detach(), Cg.detach()
        T = p[:, :, None] * q[:, None, :]
        for _ in range(gw_iters):
            M = -2.0 * torch.bmm(torch.bmm(Crd, T), Cgd)             # interaction cost
            T = _sinkhorn(M, p, q, eps, sink_iters)
        T = T.detach()
    # ---- distortion under the fixed coupling, differentiable in C_gen ----
    Tc = T.sum(1)                                                    # (B,N) gen marginal
    TtCrT = torch.bmm(torch.bmm(T.transpose(1, 2), Cr), T)          # (B,N,N)
    L = (Cg ** 2 * (Tc[:, :, None] * Tc[:, None, :])).sum(dim=(1, 2)) \
        - 2.0 * (Cg * TtCrT).sum(dim=(1, 2))
    # shift by the C_ref self-term so the loss is >=0 and ~0 at perfect match
    Tr = T.sum(2)
    L = L + (Cr ** 2 * (Tr[:, :, None] * Tr[:, None, :])).sum(dim=(1, 2))
    return L.clamp(min=0).mean()


def _geo_energy(x1_hat, batch, nest_mat):
    """Batchable differentiable geometric energy on the predicted endpoint.

    Two terms in METRIC space (positions × frame half-sizes):
      * collision — squared circle overlap of non-nestable object pairs;
      * boundary  — footprint poking outside the room (object circumradius
        exceeds its inward distance to the nearest boundary point).
    Returns a scalar loss (already mask-normalised)."""
    mask = batch["mask"]
    fh = batch["frame_h"]                                  # (B, 2) metric half
    p = x1_hat[..., :2] * fh[:, None, :]                   # (B, N, 2) metric
    # circumradius from log sizes in cond[...,0:2]
    w = torch.exp(batch["cond"][..., 0]); d = torch.exp(batch["cond"][..., 1])
    r = 0.5 * torch.sqrt(w * w + d * d)                    # (B, N) metric
    B, N = r.shape
    mf = mask.float()
    # ---- collision ----
    # manual pairwise distance (NOT torch.cdist): cdist has no second-derivative
    # implementation, which breaks the train-through projection's double-backward
    # (create_graph).  This form is double-differentiable.
    diff = p[:, :, None, :] - p[:, None, :, :]            # (B, N, N, 2)
    dist = (diff * diff).sum(-1).clamp(min=1e-12).sqrt() + 1e-6   # (B, N, N)
    overlap = (r[:, :, None] + r[:, None, :] - dist).clamp(min=0.0)
    nest = nest_mat[batch["cat"][:, :, None], batch["cat"][:, None, :]]  # (B,N,N)
    pair = (mf[:, :, None] * mf[:, None, :])
    tri = torch.triu(torch.ones(N, N, device=p.device), diagonal=1)[None]
    wcol = pair * tri * (~nest).float()
    col = (overlap ** 2 * wcol).sum() / wcol.sum().clamp(min=1)
    # ---- boundary ----
    bp = batch["boundary"][..., :2] * fh[:, None, :]       # (B, Nb, 2) metric
    bn = batch["boundary"][..., 4:6]                       # (B, Nb, 2) normals
    # squared distance is enough to pick the nearest boundary sample (argmin is
    # non-differentiable anyway); manual form avoids cdist's missing 2nd deriv.
    d2 = ((p[:, :, None, :] - bp[:, None, :, :]) ** 2).sum(-1)   # (B, N, Nb)
    kmin = d2.argmin(-1)                                   # (B, N)
    bp_near = torch.gather(bp, 1, kmin[..., None].expand(-1, -1, 2))
    bn_near = torch.gather(bn, 1, kmin[..., None].expand(-1, -1, 2))
    inward = ((p - bp_near) * bn_near).sum(-1)             # (B, N) +inside
    bnd = (r - inward).clamp(min=0.0) ** 2                 # poke-out penalty
    bound = (bnd * mf).sum() / mf.sum().clamp(min=1)
    return col + bound


def warp_scene(scene: Scene, new_room, clip_inside: bool = True) -> Scene:
    """Move a layout into a differently shaped room by the MRR-frame map.

    Used only to *manufacture* a pseudo-reference: it is the affine baseline,
    which is precisely what a reference design looks like when it has been
    naively transplanted, so the model learns to undo that transplant.

    ``clip_inside`` (default True) projects any object centre that ends up
    outside the new_room polygon back to the nearest point *inside* the room,
    then nudges it a bit further in.  This closes a train/test distribution
    gap that measured 96x on a 1500-sample probe: real reference inputs at
    inference have 0.1 % center-OOB, but raw warp produces 9.6 % because
    MRR-affine can push objects past a shrunk / corner-cut boundary.  The
    model shouldn't spend capacity learning to fix OOB inputs it never sees.
    """
    from shapely.geometry import Point as _P
    out = scene.copy()
    src = _mrr_frame(as_polygon(scene.room))
    tgt = _mrr_frame(as_polygon(new_room))
    dang = tgt[5] - src[5]
    out.room = new_room.copy()
    poly = as_polygon(new_room) if clip_inside else None
    for o in out.objects:
        o.xy = _map_point(o.xy, src, tgt)
        o.yaw = o.yaw + dang
        if clip_inside and not poly.contains(_P(*o.xy)):
            # project centre to the nearest point on the polygon exterior,
            # then step slightly inwards along the exterior's inward normal so
            # the centre is safely inside (not exactly on the boundary).
            ex = poly.exterior
            d = ex.project(_P(*o.xy))
            p_on = ex.interpolate(d)
            # inward direction: from the outside point toward the polygon centroid
            centroid = np.array(poly.centroid.coords[0])
            to_c = centroid - np.array([p_on.x, p_on.y])
            n = np.linalg.norm(to_c)
            if n > 1e-6:
                inward = to_c / n
                o.xy = np.array([p_on.x, p_on.y]) + inward * 0.05
            else:
                o.xy = np.array([p_on.x, p_on.y])
    return out


@dataclass
class TrainConfig:
    epochs: int = 30
    batch: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 8
    device: str = "cuda:0"
    d_model: int = 256
    depth: int = 6
    heads: int = 8
    levels: tuple = (1, 2, 3, 4, 5)
    ema: float = 0.999
    grad_clip: float = 1.0
    out: str = "outputs/flow"
    log_every: int = 50
    seed: int = 0
    wall_aux: float = 3.0        # weight of the reference-conditioned
                                 # wall-hugging auxiliary loss (0 disables)
    yaw_norm: float = 0.0        # weight of the unit-norm yaw-confidence
                                 # regulariser on the predicted (cos, sin)
    wall_align_loss: float = 1.0 # LEGO-Net-style: learn to align wall objects
                                 # parallel to their nearest wall (0 disables)
    wall_pos_loss: float = 0.0   # upweight the flow-matching loss on the
                                 # *position* channels for wall objects, so the
                                 # model seats them tight to the wall.  A small
                                 # normalised-coordinate error becomes a larger
                                 # metric float in a bigger room, so this is
                                 # what removes the residual float when the
                                 # target room grows (0 disables)
    prior_x0: bool = False           # ReRoom 2.0 Module 1 (LEGO-Net paradigm):
                                     # start the flow from the REFERENCE layout
                                     # projected into the target frame (the
                                     # ref_state already carried in `cond`) plus
                                     # noise, instead of Gaussian N(0,I).  The
                                     # flow then learns *rectification* (snap to
                                     # walls, fix penetration, keep motifs rigid)
                                     # rather than blind generation.  Same change
                                     # is mirrored at sampling time; the flag is
                                     # saved in the checkpoint so inference auto-
                                     # matches how the model was trained.
    prior_noise: float = 0.3         # std of the noise added to the projected
                                     # prior (in normalised frame units).
    geo_bias: bool = False           # ReRoom 2.0 Module 2: dense pairwise
                                     # relational-geometry attention bias + per-
                                     # object wall-alignment feature (LEGO-Net).
    wall_tokens: bool = False        # Full Module 2: object<->wall-segment
                                     # cross-attention (boundary points as wall
                                     # tokens); model can pick which wall to snap.
    parent_relative: bool = False    # ReRoom 2.0 Step 2: predict motif children
                                     # as a scale-invariant offset from their
                                     # parent (head), so furniture is not torn
                                     # apart when the room is scaled up.
    energy_loss: float = 0.0         # Direction-1 (bake polish into weights):
                                     # a differentiable geometric energy on the
                                     # predicted endpoint x1_hat — collision
                                     # (non-nestable overlap) + boundary (footprint
                                     # outside room).  Trains the flow to emit
                                     # low-energy layouts so test-time polish can
                                     # be shortened/removed.  Recommended 2-8.
    energy_loss_tau: float = 0.5     # only fire above this tau (low-tau x1_hat
                                     # is unreliable).
    proj_loss: float = 0.0           # A: TRAIN-THROUGH differentiable projection
                                     # (realises Prop 3).  Runs the projection
                                     # Pi_theta as `proj_iters` unrolled gradient
                                     # steps on the geometric energy WITH
                                     # create_graph, so gradients flow through the
                                     # projection back into the flow, and
                                     # penalises the projection *displacement*
                                     # ||Pi(x1_hat) - x1_hat||.  The flow thus
                                     # learns to land on the projection's fixed
                                     # point (feasible-by-construction): at test
                                     # time the deployed Pi_theta becomes a
                                     # near-no-op that does not disturb topology.
                                     # Distinct from energy_loss (raw energy):
                                     # this penalises the movement the *deployed*
                                     # operator induces.  Recommended 4-12.
    proj_iters: int = 15             # K unrolled inner steps for the train-through
                                     # projection (test-time uses the full 40).
    proj_step: float = 0.2           # inner GD step size for the unroll.
    proj_anchor: float = 2.0         # anchor pull toward the proposal inside the
                                     # unroll (topology-preserving; mirrors the
                                     # deployed project_batch's w_anchor).
    proj_loss_tau: float = 0.5       # only fire the projection loss above this tau.
    # --- project5: END-TO-END joint training -------------------------------
    e2e: bool = False                # master switch. Applies the reconstruction
                                     # loss to the PROJECTED layout x*=Pi(x1_hat)
                                     # instead of the raw endpoint, so selection,
                                     # placement and feasibility are optimised
                                     # against one deployed objective (DESIGN.md).
    e2e_recon: float = 1.0           # weight of the through-projection recon loss
    e2e_resid: float = 1.0           # weight of E_geo(x*): residual infeasibility
                                     # the projection could NOT fix is charged to
                                     # the flow -> feasible-by-construction.
    e2e_tau: float = 0.5             # fire above this tau (endpoint must be usable)
    gw_loss: float = 0.0             # B: differentiable entropic GROMOV-WASSERSTEIN
                                     # relational loss (true OT).  Couples the
                                     # reference and generated relational
                                     # structures with a learned soft coupling and
                                     # penalises transported relational distortion;
                                     # scale-invariant, degrades gracefully under
                                     # pruning/substitution.  Trains global
                                     # relational structure (independent of the
                                     # motif metric).  Recommended 1-5.
    gw_loss_tau: float = 0.4         # only fire the GW loss above this tau.
    gw_eps: float = 0.05             # entropic regularisation for the GW coupling.
    gw_iters: int = 10               # GW outer (Sinkhorn-of-Sinkhorn) iterations.
    mask_flow: bool = False          # D1: joint discrete-continuous flow.  Learn
                                     # a per-object existence logit jointly with
                                     # pose, replacing the greedy Summarise prune.
    mask_loss: float = 1.0           # weight of the existence velocity loss.
    mask_logit: float = 4.0          # +/- target logit for keep/drop; prior
                                     # starts at +logit (assume keep, learn drop).
    drop_frac: float = 0.0           # fraction of forward pairs built by the
                                     # shrink-hard path that physically drops
                                     # infeasible objects (D1 drop supervision).
    child_loss_weight: float = 10.0  # Route B scale compensation: child relative
                                     # velocity lives at table-local scale (~0.3)
                                     # while the parent is at room scale (~1), so
                                     # without upweighting the child fm-loss the
                                     # gradient is drowned by the parent.  ~1/0.3^2.
    rel_loss: float = 0.0            # Pairwise relational consistency loss.
                                     # For every pair in the same motif,
                                     # penalise |‖x̂_i − x̂_j‖ − ‖x_i^ref − x_j^ref‖|
                                     # so the flow's velocity field learns
                                     # intra-motif rigidity end-to-end.
                                     # 0 disables.  Recommended 4-8.
    wall_metric_loss: float = 0.0    # NEW: penalise |pred x1 - true x1| in
                                     # *metric* metres (multiplied by target
                                     # frame half-size) for wall-affinity
                                     # objects, only at tau > wall_metric_tau
                                     # to avoid low-tau noise.  This is what
                                     # wall_pos_loss should have been -- a
                                     # 1 % normalised error in a 6 m room is
                                     # 6 cm; this makes the loss say so.
    wall_metric_tau: float = 0.5     # only fire above this tau
    wall_metric_wd: float = 0.02     # extra weight decay on wall-branch params
    jitter_pos: float = 0.08     # LEGO-Net-style perturbation of the pseudo-
    jitter_yaw: float = 0.10     # reference, so the model learns messy -> regular
    scramble_prob: float = 0.0   # LEGO-Net-strong: with this probability, the
                                 # pseudo-reference's positions are *fully*
                                 # randomised inside the deformed room (yaws
                                 # uniform too), so the model has to recover
                                 # the correct layout from category / motif /
                                 # boundary alone -- teaching it that a wall-
                                 # affinity object goes to a wall, not just to
                                 # "wherever it started"
    subst_prob: float = 0.0      # per-object probability of substituting an
                                 # asset for a same-category, DIFFERENT-size
                                 # asset from the bank on BOTH input and GT
                                 # sides.  This is the missing signal for the
                                 # per-pair elasticity alpha_ij -- with the
                                 # same relation type at various object sizes
                                 # the model learns which distances are rigid
                                 # and which stretch.  0 disables.
    level_probs: tuple = (0.4, 0.2, 0.1, 0.15, 0.15)   # sampling weights for
                                 # deform levels 1..5.  Default biases toward
                                 # L1 uniform_scale (the direct scale-only
                                 # variation that matches the 3-sizes test),
                                 # keeps some L2 aspect, tones down L3 slant
                                 # (near-identity, teaches little); L4 and L5
                                 # remain for shape diversity.
    l1_range: tuple = (0.5, 2.0)         # widened from (0.7, 1.4) so the
                                 # 3-sizes test (s=0.75 / 1.35) lands inside
                                 # the training body rather than at the tail.
    l1_u_shape: bool = True      # sample L1's s from Beta(0.5, 0.5) shape --
                                 # more mass at the tails, less near identity
    use_hybrid: bool = False     # use the cross-scene / motif-rigid HybridPairs
                                 # pipeline (reroom.generative.xscene) instead of
                                 # the affine-warp RetargetPairs: real targets,
                                 # 70% forward-deform + 30% cross-pairing.
    hybrid_forward_frac: float = 0.7
    hybrid_max_deg: float = 30.0     # anchor-orientation filter threshold
    hybrid_jaccard: float = 0.6      # cross-pairing category-overlap threshold
    init_from: str = ""          # warm-start checkpoint; layers whose shape
                                 # still matches are copied, the rest are left
                                 # at their fresh init (e.g. the input
                                 # projection when TOKEN_COND_DIM changed)


class RetargetPairs(Dataset):
    """On-the-fly (pseudo-reference, true layout) pairs."""

    def __init__(self, scenes: list[Scene], levels=(1, 2, 3, 4, 5),
                 elasticity: ElasticityModel | None = None,
                 samples_per_scene: int = 1, seed: int = 0,
                 jitter_pos: float = 0.0, jitter_yaw: float = 0.0,
                 scramble_prob: float = 0.0,
                 subst_prob: float = 0.0,
                 bank=None,
                 level_probs=None,
                 l1_range: tuple = (0.5, 2.0),
                 l1_u_shape: bool = True,
                 cache: bool = False):
        self.dicts = [s.to_dict() for s in scenes]
        self.levels = tuple(levels)
        self.spp = samples_per_scene
        self.seed = seed
        self._elast = elasticity or PriorElasticity()
        # LEGO-Net-style input perturbation: jitter the pseudo-reference before
        # the intent graph is read, so the model is trained to recover a regular
        # target from a slightly messy reference rather than only from an
        # affine-warped one -- it learns to *regularise*, not just to copy.
        self.jitter_pos = jitter_pos
        self.jitter_yaw = jitter_yaw
        self.scramble_prob = scramble_prob
        self.subst_prob = subst_prob
        self.bank = bank
        # normalise level probabilities
        if level_probs is not None:
            self.level_probs = np.array(level_probs, dtype=float)
            self.level_probs /= self.level_probs.sum()
            self.levels_available = np.array([1, 2, 3, 4, 5])
        else:
            self.level_probs = None
            self.levels_available = None
        self.l1_range = l1_range
        self.l1_u_shape = l1_u_shape
        # Optional lazy in-memory cache.  __getitem__ is deterministic per idx
        # (rng seed = self.seed*1000003+idx), so a first-epoch populate lets
        # every later epoch skip the expensive elasticity/graph work.  For
        # 11k scenes at ~30 KB/sample this uses ~300-500 MB of RAM; enable
        # only when the container has DataLoader worker constraints (shm) that
        # force workers=0.  Off by default so training with many workers is
        # unaffected.
        self._cache = {} if cache else None

    def __len__(self) -> int:
        return len(self.dicts) * self.spp

    def __getitem__(self, idx: int):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        base = idx // self.spp
        scene = scene_from_dict(self.dicts[base])
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2 ** 32))
        # weighted-level sampling: bias toward L1 (shape-preserving scale) so
        # the shape-preserving fraction of training pairs (~1.8 % at default
        # weights before this change) rises to ~15-20 %, matching the density
        # the 3-sizes test needs.
        if self.level_probs is not None:
            level = int(rng.choice(self.levels_available, p=self.level_probs))
        else:
            level = int(rng.choice(self.levels))
        for _ in range(4):
            ref_room = deform_room(scene.room, level, rng,
                                   l1_range=self.l1_range,
                                   l1_u_shape=self.l1_u_shape).room
            if ref_room.area > 3.0:
                break
        pseudo = warp_scene(scene, ref_room, clip_inside=True)
        # Object substitution (per-object): with probability subst_prob swap a
        # keeper for a same-category but DIFFERENT-size asset from the bank on
        # BOTH the input pseudo-ref and the GT scene simultaneously.  This is
        # what teaches the model alpha_ij: the *relation* stays the same, the
        # object *sizes* change, so the model learns which distances are
        # rigid and which should stretch as sizes vary.
        if self.bank is not None and self.subst_prob > 0.0:
            gt_scene = scene
            for oi, ot in zip(pseudo.objects, gt_scene.objects):
                if not ot.keep or not self.bank.has(oi.category): continue
                if rng.random() > self.subst_prob: continue
                idxs = list(self.bank.by_category[oi.category])
                if not idxs: continue
                asset = self.bank.assets[int(rng.choice(idxs))]
                oi.size = asset.size.copy()
                ot.size = asset.size.copy()
        # Scramble augmentation (LEGO-Net-strong): with probability
        # scramble_prob, replace the pseudo-reference positions with uniform
        # random samples inside the deformed room and yaws with U(-pi, pi).
        # This forces the model to recover the layout from category /
        # motif / wall-affinity / boundary tokens rather than from proximity
        # to the pseudo-ref -- exactly the "TV should be against the wall"
        # signal that the affine-warp pairs don't carry strongly enough.
        # Object footprints (size, cat, meta) are preserved so the intent
        # graph still identifies the same objects.
        if self.scramble_prob > 0.0 and rng.random() < self.scramble_prob:
            # Rigid-Group Scramble: same-motif members are scrambled as ONE
            # rigid body (same rotation + translation), so the model never
            # sees "dining chairs floating away from their table" as valid
            # training input.  Prior implementation scrambled each object
            # independently and taught the flow exactly the failure mode we
            # want to prevent (chairs decoupled from parent).
            #
            # Group members: from pseudo's scene graph motifs.  Objects not
            # in any motif get individual scramble (old behavior).
            _grp_graph = build_motifs(build_scene_graph(pseudo))
            in_motif = {}
            for _m in _grp_graph.motifs:
                for _i in _m.members:
                    in_motif[_i] = _m
            poly = as_polygon(ref_room)
            minx, miny, maxx, maxy = poly.bounds

            def _rand_inside():
                for _ in range(20):
                    px = rng.uniform(minx, maxx)
                    py = rng.uniform(miny, maxy)
                    if poly.contains(_ShPoint(px, py)):
                        return np.array([px, py])
                return np.array([(minx + maxx) / 2, (miny + maxy) / 2])

            done = set()
            for i, o in enumerate(pseudo.objects):
                if i in done: continue
                m = in_motif.get(i)
                if m is None:
                    # non-motif object: individual scramble as before
                    o.xy = _rand_inside()
                    o.yaw = float(rng.uniform(-math.pi, math.pi))
                    done.add(i)
                    continue
                # motif: pick a head (members[0]) and apply the same
                # (translation, rotation) to every member preserving their
                # relative offsets & orientations
                head_i = m.members[0]
                head_orig_xy = pseudo.objects[head_i].xy.copy()
                head_orig_yaw = float(pseudo.objects[head_i].yaw)
                new_head_xy = _rand_inside()
                new_head_yaw = float(rng.uniform(-math.pi, math.pi))
                dth = new_head_yaw - head_orig_yaw
                c, s = math.cos(dth), math.sin(dth)
                rot = np.array([[c, -s], [s, c]])
                for j in m.members:
                    if j >= len(pseudo.objects): continue
                    obj = pseudo.objects[j]
                    off = obj.xy - head_orig_xy
                    obj.xy = new_head_xy + rot @ off
                    obj.yaw = float(obj.yaw + dth)
                    done.add(j)
        elif self.jitter_pos > 0.0 or self.jitter_yaw > 0.0:
            for o in pseudo.objects:
                o.xy = o.xy + rng.normal(0.0, self.jitter_pos, size=2)
                o.yaw = float(o.yaw + rng.normal(0.0, self.jitter_yaw))
        graph = build_motifs(build_scene_graph(pseudo))
        intent = build_design_intent(graph, scene.room, elasticity=self._elast)
        item = build_tokens(intent, scene.room, scene)
        if self._cache is not None:
            self._cache[idx] = item
        return item


def _collate_fn(items):
    return collate([b for b in items if b is not None])


def train_flow(scenes: list[Scene], val_scenes: list[Scene] | None = None,
               cfg: TrainConfig | None = None,
               elasticity: ElasticityModel | None = None,
               bank=None) -> FlowModel:
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    dev = torch.device(cfg.device)

    # cache the dataset when workers=0 (containers with too-small /dev/shm):
    # __getitem__ is deterministic per idx, so first-epoch populate turns every
    # later epoch into a dict lookup instead of a 100 ms elasticity+graph call.
    if cfg.use_hybrid:
        from .xscene import HybridPairs, build_pair_index_filtered
        print("[flow] building filtered cross-scene pair index "
              f"(jaccard>{cfg.hybrid_jaccard}, orient<={cfg.hybrid_max_deg} deg)...",
              flush=True)
        pair_index = build_pair_index_filtered(
            scenes, thresh=cfg.hybrid_jaccard, max_deg=cfg.hybrid_max_deg,
            max_partners=16, seed=cfg.seed)
        cov = len(pair_index) / max(len(scenes), 1)
        print(f"[flow] HybridPairs: {len(pair_index)}/{len(scenes)} scenes have a "
              f"cross partner ({100*cov:.0f}%); forward_frac={cfg.hybrid_forward_frac}",
              flush=True)
        train_ds = HybridPairs(scenes, pair_index,
                               forward_frac=cfg.hybrid_forward_frac,
                               elasticity=elasticity, levels=cfg.levels,
                               l1_range=cfg.l1_range, l1_u_shape=cfg.l1_u_shape,
                               max_deg=cfg.hybrid_max_deg, seed=cfg.seed,
                               cache=(cfg.workers == 0), drop_frac=cfg.drop_frac)
    else:
        train_ds = RetargetPairs(scenes, cfg.levels, elasticity, seed=cfg.seed,
                                 jitter_pos=cfg.jitter_pos, jitter_yaw=cfg.jitter_yaw,
                                 scramble_prob=cfg.scramble_prob,
                                 subst_prob=cfg.subst_prob, bank=bank,
                                 level_probs=cfg.level_probs,
                                 l1_range=cfg.l1_range,
                                 l1_u_shape=cfg.l1_u_shape,
                                 cache=(cfg.workers == 0))
    # Container /dev/shm is 64 MB and another workload holds ~32 MB of it.
    # persistent_workers=True keeps every worker's shm mapping for the whole
    # run and eventually crashes with "no space left on device"; with fresh
    # workers each epoch and prefetch_factor=1 the shm ceiling is ~4 MB per
    # worker × workers, which fits.
    dl_kwargs = dict(batch_size=cfg.batch, shuffle=True,
                     num_workers=cfg.workers, collate_fn=_collate_fn,
                     drop_last=True, persistent_workers=False)
    if cfg.workers > 0:
        dl_kwargs["prefetch_factor"] = 1
    dl = DataLoader(train_ds, **dl_kwargs)
    val_dl = None
    test_dl_075 = None
    test_dl_135 = None
    if val_scenes:
        val_ds = RetargetPairs(val_scenes, cfg.levels, elasticity, seed=cfg.seed + 7,
                               cache=(cfg.workers == 0))
        val_dl = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                            num_workers=(cfg.workers // 2 if cfg.workers > 0 else 0),
                            collate_fn=_collate_fn)
        # Fixed test set: exactly the shipped 3-sizes test's inference shape.
        # For each val scene, use it as the reference (target = real scene,
        # ref = uniformly-scaled version).  This maps to the training pair
        # structure but with s pinned to 0.75 or 1.35, so metrics are stable
        # across epochs on the same scenes.  Small (n=40) so the extra
        # per-epoch cost is negligible.
        n_test = min(40, len(val_scenes))
        test_scenes = val_scenes[:n_test]
        test_ds_075 = RetargetPairs(
            test_scenes, levels=(1,), elasticity=elasticity,
            seed=cfg.seed + 9997,
            l1_range=(0.75, 0.75001), l1_u_shape=False,
            cache=(cfg.workers == 0))
        test_ds_135 = RetargetPairs(
            test_scenes, levels=(1,), elasticity=elasticity,
            seed=cfg.seed + 9998,
            l1_range=(1.35, 1.35001), l1_u_shape=False,
            cache=(cfg.workers == 0))
        test_dl_075 = DataLoader(test_ds_075, batch_size=cfg.batch, shuffle=False,
                                  num_workers=0, collate_fn=_collate_fn)
        test_dl_135 = DataLoader(test_ds_135, batch_size=cfg.batch, shuffle=False,
                                  num_workers=0, collate_fn=_collate_fn)
        print(f"[flow] fixed test set: {n_test} scenes x 2 sizes "
              f"(s=0.75 and s=1.35)", flush=True)

    model = FlowModel(cfg.d_model, cfg.depth, cfg.heads,
                      geo_bias=cfg.geo_bias, wall_tokens=cfg.wall_tokens,
                      parent_relative=cfg.parent_relative,
                      mask_flow=cfg.mask_flow).to(dev)
    if cfg.init_from and os.path.exists(cfg.init_from):
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        src_sd = ck.get("ema", ck.get("model", ck))
        # strip DataParallel's "module." prefix if the checkpoint has it
        src_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in src_sd.items()}
        msd = model.state_dict()
        copied, skipped = 0, []
        for k, v in src_sd.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k] = v
                copied += 1
            else:
                skipped.append(k)
        model.load_state_dict(msd)
        print(f"[flow] warm-start from {cfg.init_from}: copied {copied} "
              f"tensors, reinit {skipped}", flush=True)
    ema = FlowModel(cfg.d_model, cfg.depth, cfg.heads,
                    geo_bias=cfg.geo_bias, wall_tokens=cfg.wall_tokens,
                    parent_relative=cfg.parent_relative,
                    mask_flow=cfg.mask_flow).to(dev)
    ema.load_state_dict(model.state_dict())
    # DataParallel across all visible CUDA devices.  train.py doesn't need to
    # know which GPUs are used -- CUDA_VISIBLE_DEVICES / cfg.device gates that.
    # Wrapped *after* warm-start and after ema copy so both stay unwrapped.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[flow] DataParallel across {torch.cuda.device_count()} GPUs",
              flush=True)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(len(dl), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(steps, 1), pct_start=0.06)

    nest_mat = _nestable_matrix(dev) if (cfg.energy_loss > 0.0 or cfg.proj_loss > 0.0) else None
    free_vec = _free_vector(dev) if cfg.proj_loss > 0.0 else None
    step = 0
    history = []
    for ep in range(cfg.epochs):
        model.train()
        tot, cnt = 0.0, 0
        tot_w = 0.0
        tot_a = 0.0
        tot_m = 0.0
        tot_mk = 0.0
        tot_pj = 0.0
        tot_pe = 0.0
        tot_gw = 0.0
        tot_e2e = 0.0
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            x1_world = batch["state"]
            if cfg.prior_x0:
                # Module 1: informative prior.  cond[...,10:14] is the reference
                # layout already expressed in the target's normalised frame --
                # i.e. the affine projection T(S_ref, P_tgt).  Start there + noise
                # so the flow learns to rectify, not to generate from scratch.
                x0_world = batch["cond"][..., 10:14] + cfg.prior_noise * torch.randn_like(x1_world)
            else:
                x0_world = torch.randn_like(x1_world)
            if cfg.parent_relative:
                # Step 2: flow lives in the scale-invariant parent-relative space
                from .model import to_relative, to_world
                _par = batch["parent"]; _fh = batch["frame_h"]
                x1 = to_relative(x1_world, _par, _fh)
                x0 = to_relative(x0_world, _par, _fh)
            else:
                x1 = x1_world; x0 = x0_world
            tau = torch.rand(x1.shape[0], device=dev)
            xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * x1
            v_target = x1 - x0
            if cfg.mask_flow:
                # D1: existence logit ell flows alongside pose.  Informative
                # prior ell0 = +logit (assume keep), target ell1 = +/- logit.
                present1 = batch["present1"][..., None]              # (B,N,1)
                ell1 = (present1 * 2.0 - 1.0) * cfg.mask_logit
                # uninformative existence prior (mean 0): the head must actively
                # predict EVERY object's existence, giving dense gradient rather
                # than only the rare droppers (an "assume keep" prior collapses).
                ell0 = torch.randn_like(ell1)
                batch["ell"] = (1 - tau)[:, None, None] * ell0 + tau[:, None, None] * ell1
                ellv_target = ell1 - ell0
            v = model(xt, tau, batch)
            if cfg.mask_flow:
                ell_v = v[..., 4:5]
                v = v[..., :4]
            m = batch["mask"][..., None].float()
            # D1: supervise pose only on survivors (dropped objects have no GT
            # position); existence loss below still spans all valid tokens.
            m_pose = m * present1 if cfg.mask_flow else m
            if cfg.parent_relative:
                # Route B: upweight children so their table-local-scale velocity
                # is not drowned by the room-scale parent gradient.
                is_child = (batch["parent"] >= 0).float()[..., None]
                w = m_pose * (1.0 + (cfg.child_loss_weight - 1.0) * is_child)
                fm_loss = ((v - v_target) ** 2 * w).sum() / w.sum().clamp(min=1)
            else:
                fm_loss = ((v - v_target) ** 2 * m_pose).sum() / m_pose.sum().clamp(min=1)
            if cfg.mask_flow:
                mask_l = ((ell_v - ellv_target) ** 2 * m).sum() / m.sum().clamp(min=1)
            else:
                mask_l = torch.zeros((), device=dev)

            # reference-conditioned wall-hugging auxiliary loss.  The predicted
            # endpoint is recoverable from the velocity: x1_hat = xt + (1-tau)v.
            # Distance is measured against the *actual* room boundary, which the
            # batch already carries as sampled points (concave walls included),
            # not against the normalised MRR edge -- the flow already matches
            # the MRR coordinates, yet wall objects still land ~24 cm off,
            # because the MRR rectangle is not the real wall.  So: for each
            # object, take its distance to the nearest boundary sample, and
            # penalise the prediction for sitting *farther* from the boundary
            # than the true layout does, weighted by the reference-wall
            # affinity (cond feature -1).  True wall objects sit on the
            # boundary, so this pulls exactly those objects tight to the wall
            # they hugged in the reference; free objects (affinity ~0) are
            # untouched.
            # predicted clean endpoint, shared by the auxiliary losses
            x1_hat = xt + (1 - tau)[:, None, None] * v
            if cfg.parent_relative:
                # fm_loss / v are in the relative space; the geometry-based aux
                # losses below need world coordinates, so decode the endpoint
                # and switch the target to the world GT.
                x1_hat = to_world(x1_hat, _par, _fh)
                x1 = x1_world
            # --- project5: end-to-end supervision THROUGH the projection ---
            if cfg.e2e and (cfg.e2e_recon > 0.0 or cfg.e2e_resid > 0.0):
                fire = (tau > cfg.e2e_tau)
                if fire.any():
                    sub = {k: batch[k][fire] for k in
                           ("mask", "frame_h", "cond", "cat", "boundary")}
                    keep = None
                    if cfg.mask_flow:
                        # straight-through keep mask from the existence endpoint:
                        # hard in the forward pass, soft gradient backward, so the
                        # learned selection shapes the projected geometry.
                        ell1_hat = batch["ell"][fire] + (1 - tau[fire])[:, None, None] * ell_v[fire]
                        soft = torch.sigmoid(ell1_hat.squeeze(-1))
                        keep = (soft > 0.5).float() + soft - soft.detach()
                    xs = _proj_forward(x1_hat[fire], sub, nest_mat, free_vec, keep=keep,
                                       iters=cfg.proj_iters, step=cfg.proj_step,
                                       anchor=cfg.proj_anchor)
                    wm = sub["mask"].float()
                    if cfg.mask_flow:
                        wm = wm * present1[fire].squeeze(-1)
                    wm = wm[..., None]
                    e2e_recon = (((xs - x1[fire]) ** 2) * wm).sum() / wm.sum().clamp(min=1)
                    e2e_resid = _geo_energy(xs, sub, nest_mat)
                else:
                    e2e_recon = torch.zeros((), device=dev); e2e_resid = torch.zeros((), device=dev)
            else:
                e2e_recon = torch.zeros((), device=dev); e2e_resid = torch.zeros((), device=dev)

            if cfg.wall_aux > 0.0:
                pp = x1_hat[..., :2]                             # (B, N, 2)
                pt = x1[..., :2]
                bnd = batch["boundary"][..., :2]                # (B, Nb, 2)
                aff = batch["cond"][..., -1]                    # (B, N)
                dp = torch.cdist(pp, bnd).min(-1).values        # (B, N) pred
                dt = torch.cdist(pt, bnd).min(-1).values        # (B, N) true
                pen = aff * (dp - dt).clamp(min=0.0) ** 2
                mm = batch["mask"].float()
                # normalise by participating (reference-wall-hugging) objects
                part = ((aff > 0.3).float() * mm).sum().clamp(min=1)
                wall_loss = (pen * mm).sum() / part
            else:
                wall_loss = torch.zeros((), device=dev)

            # yaw-confidence regularisation: the predicted (cos, sin) endpoint
            # should sit on the unit circle.  On hard, ambiguous orientations
            # the flow otherwise hedges and outputs a *shrunk* vector (norm ~0.5
            # for ~2 % of objects); atan2 still yields an angle, but the shrink
            # doubles the angular noise, so a bookcase that was perfectly
            # parallel in the reference lands ~11 deg skewed.  Penalising the
            # endpoint norm away from 1 forces a confident orientation.
            if cfg.yaw_norm > 0.0:
                mm = batch["mask"].float()
                yn = x1_hat[..., 2:4].norm(dim=-1)                  # (B, N)
                yaw_norm_loss = (((yn - 1.0) ** 2) * mm).sum() / mm.sum().clamp(min=1)
            else:
                yaw_norm_loss = torch.zeros((), device=dev)

            # Wall-orientation emphasis (the effective form of the LEGO-Net
            # alignment idea for a conditioned flow): rather than penalise the
            # noisy predicted endpoint against the wall -- which is unsatisfiable
            # at low tau and did not train -- upweight the flow-matching loss on
            # the *orientation* channels for the objects that hugged a wall in
            # the reference.  The velocity target already encodes the true,
            # wall-parallel orientation, so this simply makes the model spend
            # more capacity getting wall-object orientation exactly right, which
            # is where the residual skew lives.
            if cfg.wall_align_loss > 0.0:
                mm = batch["mask"].float()
                aff = batch["cond"][..., -1]                            # (B, N)
                ori_err = ((v[..., 2:4] - v_target[..., 2:4]) ** 2).sum(-1)  # (B,N)
                part = ((aff > 0.3).float() * mm).sum().clamp(min=1)
                wall_align_loss = (aff * ori_err * mm).sum() / part
            else:
                wall_align_loss = torch.zeros((), device=dev)

            # Metric-space wall position loss (new).  The prior wall_pos_loss
            # used velocity-space L2 in normalised coords -- val dropped 14x
            # but real wall float got worse (see site sec. 8.3), because a
            # small normalised error becomes a large metric error in a big
            # room, and the loss weighted every room equally.  Fix in three
            # parts:
            #   (a) use x1_hat's *position* channels directly, not velocity, so
            #       the loss is on where the object *lands*;
            #   (b) multiply the (u, v) error by the target frame's metric
            #       half-sizes (h1, h2), so the penalty is in metres;
            #   (c) filter tau > wall_metric_tau, because at low tau x1_hat is
            #       just denoised noise -- the target is unreachable there and
            #       the gradient is misleading.
            if cfg.wall_metric_loss > 0.0:
                mm = batch["mask"].float()
                aff = batch["cond"][..., -1]                     # (B, N)
                fh = batch["frame_h"]                            # (B, 2)
                pp = x1_hat[..., :2]                             # (B, N, 2)
                pt = x1[..., :2]
                # (u, v) diff scaled to metric metres per component
                diff = (pp - pt) * fh[:, None, :]                # (B, N, 2)
                d2 = (diff ** 2).sum(-1)                         # (B, N) sq metres
                tau_gate = (tau > cfg.wall_metric_tau).float()   # (B,)
                gate = aff * mm * tau_gate[:, None]              # (B, N)
                part = ((aff > 0.3).float() * mm
                        * tau_gate[:, None]).sum().clamp(min=1)
                wall_metric = (gate * d2).sum() / part
            else:
                wall_metric = torch.zeros((), device=dev)

            # Module 4: motif-rigidity loss.  For every pair (i, j) in the same
            # motif, penalise the change of their pairwise distance relative to
            # the reference offsets (cond ref_state) -- so the predicted layout
            # keeps sofa+coffee-table / bed+nightstand groups rigid.
            if cfg.rel_loss > 0.0:
                mgrp = batch["mgrp"]                            # (B, N) int, -1 none
                ref_xy = batch["cond"][..., 10:12]             # (B, N, 2) ref frame
                p_hat = x1_hat[..., :2]                         # (B, N, 2)
                d_hat = (p_hat[:, :, None, :] - p_hat[:, None, :, :]).norm(dim=-1)
                d_ref = (ref_xy[:, :, None, :] - ref_xy[:, None, :, :]).norm(dim=-1)
                same = (mgrp[:, :, None] == mgrp[:, None, :]) & (mgrp[:, :, None] >= 0)
                mm2 = (batch["mask"][:, :, None].float()
                       * batch["mask"][:, None, :].float())
                w2 = same.float() * mm2
                rel_loss = ((d_hat - d_ref).abs() * w2).sum() / w2.sum().clamp(min=1)
            else:
                rel_loss = torch.zeros((), device=dev)

            # Direction-1: differentiable geometric energy on x1_hat (bake
            # collision-freedom + containment into the flow so test-time polish
            # can be shortened / removed).  Only fired at higher tau.
            if cfg.energy_loss > 0.0:
                tau_gate = (tau > cfg.energy_loss_tau).float().mean().clamp(min=1e-3)
                fire = (tau > cfg.energy_loss_tau)
                if fire.any():
                    energy_geo = _geo_energy(x1_hat[fire], {k: batch[k][fire] for k in
                                             ("mask","frame_h","cond","cat","boundary")},
                                             nest_mat)
                else:
                    energy_geo = torch.zeros((), device=dev)
            else:
                energy_geo = torch.zeros((), device=dev)

            if cfg.wall_pos_loss > 0.0:
                mm = batch["mask"].float()
                aff = batch["cond"][..., -1]                            # (B, N)
                pos_err = ((v[..., :2] - v_target[..., :2]) ** 2).sum(-1)  # (B,N)
                part = ((aff > 0.3).float() * mm).sum().clamp(min=1)
                wall_pos_loss = (aff * pos_err * mm).sum() / part
            else:
                wall_pos_loss = torch.zeros((), device=dev)

            # A: train-through differentiable projection (Prop 3).  Gradients flow
            # through Pi_theta's unrolled steps so the flow learns feasible-by-
            # construction endpoints; fired only at high tau where x1_hat is
            # reliable.  Uses the world-frame endpoint already decoded above.
            if cfg.proj_loss > 0.0:
                fire = (tau > cfg.proj_loss_tau)
                if fire.any():
                    proj_disp, proj_pre_E = _proj_through_energy(
                        x1_hat[fire],
                        {k: batch[k][fire] for k in
                         ("mask", "frame_h", "cond", "cat", "boundary")},
                        nest_mat, free_vec,
                        iters=cfg.proj_iters, step=cfg.proj_step,
                        anchor=cfg.proj_anchor)
                else:
                    proj_disp, proj_pre_E = torch.zeros((), device=dev), 0.0
            else:
                proj_disp, proj_pre_E = torch.zeros((), device=dev), 0.0

            # B: entropic Gromov-Wasserstein relational loss (true OT), fired at
            # higher tau where x1_hat is reliable.  Uses the world endpoint.
            if cfg.gw_loss > 0.0:
                fire = (tau > cfg.gw_loss_tau)
                if fire.any():
                    gw_l = _gw_relational_loss(
                        x1_hat[fire],
                        {"mask": batch["mask"][fire], "cond": batch["cond"][fire]},
                        eps=cfg.gw_eps, gw_iters=cfg.gw_iters)
                else:
                    gw_l = torch.zeros((), device=dev)
            else:
                gw_l = torch.zeros((), device=dev)

            loss = fm_loss + cfg.wall_aux * wall_loss \
                + cfg.gw_loss * gw_l \
                + cfg.yaw_norm * yaw_norm_loss \
                + cfg.wall_align_loss * wall_align_loss \
                + cfg.wall_pos_loss * wall_pos_loss \
                + cfg.wall_metric_loss * wall_metric \
                + cfg.rel_loss * rel_loss \
                + cfg.energy_loss * energy_geo \
                + cfg.proj_loss * proj_disp \
                + cfg.e2e_recon * e2e_recon + cfg.e2e_resid * e2e_resid \
                + cfg.mask_loss * mask_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                # ema tracks the *underlying* module, not the DataParallel wrapper
                base = model.module if hasattr(model, "module") else model
                for pe, pm in zip(ema.parameters(), base.parameters()):
                    pe.mul_(cfg.ema).add_(pm, alpha=1 - cfg.ema)
                for be, bm in zip(ema.buffers(), base.buffers()):
                    be.copy_(bm)
            tot += float(fm_loss) * int(m.sum())
            tot_w += float(wall_loss) * int(m.sum())
            tot_a += float(wall_align_loss) * int(m.sum())
            tot_m += float(wall_metric) * int(m.sum())
            tot_mk += float(mask_l) * int(m.sum())
            tot_pj += float(proj_disp) * int(m.sum())
            tot_pe += float(proj_pre_E) * int(m.sum())
            tot_gw += float(gw_l) * int(m.sum())
            tot_e2e += float(e2e_recon) * int(m.sum())
            cnt += int(m.sum())
            step += 1
            if step % cfg.log_every == 0:
                print(f"  ep {ep} step {step}/{steps} "
                      f"fm {tot / max(cnt, 1):.4f} wall {tot_w / max(cnt, 1):.4f} align {tot_a / max(cnt, 1):.4f} metric {tot_m / max(cnt, 1):.4f} mask {tot_mk / max(cnt, 1):.4f} projE {tot_pj / max(cnt, 1):.5f} preE {tot_pe / max(cnt, 1):.5f}",
                      flush=True)
        row = {"epoch": ep, "train_loss": tot / max(cnt, 1),
               "wall_loss": tot_w / max(cnt, 1),
               "align_loss": tot_a / max(cnt, 1),
               "metric_loss": tot_m / max(cnt, 1)}
        if cfg.proj_loss > 0.0:
            row["proj_postE"] = tot_pj / max(cnt, 1)
            row["proj_preE"] = tot_pe / max(cnt, 1)
        if cfg.gw_loss > 0.0:
            row["gw_loss"] = tot_gw / max(cnt, 1)
        if cfg.e2e:
            row["e2e_recon"] = tot_e2e / max(cnt, 1)
        if val_dl is not None:
            val_fm, val_metric_m = _validate(ema, val_dl, dev,
                                             prior_x0=cfg.prior_x0, prior_noise=cfg.prior_noise,
                                             parent_relative=cfg.parent_relative,
                                             mask_flow=cfg.mask_flow, mask_logit=cfg.mask_logit)
            row["val_loss"] = val_fm
            row["val_wall_m"] = val_metric_m
        # fixed-test bench (a small held-out set of scenes deformed at the
        # exact test sizes 0.75x and 1.35x) -- gives an epoch-to-epoch stable
        # number for the wall-hugging metric on the *inference-shaped* task.
        if test_dl_075 is not None:
            _, m075 = _validate(ema, test_dl_075, dev, tau_fixed=0.9,
                                 prior_x0=cfg.prior_x0, prior_noise=cfg.prior_noise,
                                 parent_relative=cfg.parent_relative,
                                 mask_flow=cfg.mask_flow, mask_logit=cfg.mask_logit)
            row["test075_wall_m"] = m075
        if test_dl_135 is not None:
            _, m135 = _validate(ema, test_dl_135, dev, tau_fixed=0.9,
                                 prior_x0=cfg.prior_x0, prior_noise=cfg.prior_noise,
                                 parent_relative=cfg.parent_relative,
                                 mask_flow=cfg.mask_flow, mask_logit=cfg.mask_logit)
            row["test135_wall_m"] = m135
        history.append(row)
        print(f"[flow] epoch {ep}: " +
              "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"),
              flush=True)
        # unwrap DataParallel so saved state_dict has no "module." prefix --
        # inference / warm-start code can then load without special handling
        raw_model = model.module if hasattr(model, "module") else model
        ck = {"model": raw_model.state_dict(), "ema": ema.state_dict(),
              "cfg": cfg.__dict__, "history": history,
              "epoch": ep, "val_loss": row.get("val_loss")}
        torch.save(ck, os.path.join(cfg.out, "flow.pt"))
        # Also keep a snapshot of the best-val epoch, so a long training run
        # that overshoots the sweet spot does not lose it.  train.py used to
        # overwrite flow.pt every epoch, and flow_lego's best val at epoch 20
        # would have been lost by epoch 29.
        vl = row.get("val_loss")
        if vl is not None:
            prior = [r.get("val_loss") for r in history[:-1]
                     if r.get("val_loss") is not None]
            if not prior or vl < min(prior):
                torch.save(ck, os.path.join(cfg.out, "flow_best.pt"))
                print(f"[flow] epoch {ep}: new best val={vl:.4f}, saved flow_best.pt",
                      flush=True)
    return ema


@torch.no_grad()
def _validate(model: FlowModel, dl, dev,
              tau_fixed: float | None = None,
              prior_x0: bool = False, prior_noise: float = 0.3,
              parent_relative: bool = False,
              mask_flow: bool = False, mask_logit: float = 4.0) -> tuple[float, float]:
    """Return (val_fm_loss, val_wall_metric_m).

    val_fm_loss: the standard flow-matching MSE, unchanged.

    val_wall_metric_m: the wall-affinity subset's *metric-space* position
    error between x1_hat and the true target, in metres.  Complements val_loss
    with a number that directly tracks the real wall-hugging metric we care
    about, at zero extra data cost -- it is computed on the same val batches.
    When ``tau_fixed`` is set (e.g. 0.9), tau is fixed instead of sampled so
    the number is comparable across epochs on the same fixed batches.
    """
    model.eval()
    tot, cnt = 0.0, 0
    err_m2, part_m = 0.0, 0
    for batch in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        x1_world = batch["state"]
        if prior_x0:
            x0_world = batch["cond"][..., 10:14] + prior_noise * torch.randn_like(x1_world)
        else:
            x0_world = torch.randn_like(x1_world)
        if parent_relative:
            from .model import to_relative, to_world
            _par = batch["parent"]; _fh2 = batch["frame_h"]
            x1 = to_relative(x1_world, _par, _fh2); x0 = to_relative(x0_world, _par, _fh2)
        else:
            x1 = x1_world; x0 = x0_world
        if tau_fixed is not None:
            tau = torch.full((x1.shape[0],), tau_fixed, device=dev)
        else:
            tau = torch.rand(x1.shape[0], device=dev)
        xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * x1
        if mask_flow:
            present1 = batch["present1"][..., None]
            ell1 = (present1 * 2.0 - 1.0) * mask_logit
            ell0 = torch.randn_like(ell1)
            batch["ell"] = (1 - tau)[:, None, None] * ell0 + tau[:, None, None] * ell1
        v = model(xt, tau, batch)
        if mask_flow:
            v = v[..., :4]
        m = batch["mask"][..., None].float()
        m_pose = m * batch["present1"][..., None] if mask_flow else m
        tot += float((((v - (x1 - x0)) ** 2) * m_pose).sum())
        cnt += int(m_pose.sum())
        # metric-space wall position error on x1_hat
        x1_hat = xt + (1 - tau)[:, None, None] * v
        if parent_relative:
            x1_hat = to_world(x1_hat, _par, _fh2); x1 = x1_world
        aff = batch["cond"][..., -1]                               # (B, N)
        fh = batch["frame_h"]                                      # (B, 2)
        diff = (x1_hat[..., :2] - x1[..., :2]) * fh[:, None, :]    # metres
        d = torch.sqrt((diff ** 2).sum(-1) + 1e-9)                 # (B, N)
        mask = batch["mask"].float()
        gate = aff * mask * (aff > 0.3).float()
        err_m2 += float((gate * d).sum())
        part_m += float(gate.sum())
    model.train()
    val_fm = tot / max(cnt, 1)
    val_wall_metric = err_m2 / max(part_m, 1)
    return val_fm, val_wall_metric
