"""Sampling from the flow proposal, then projecting onto the constraints.

    Generative Proposal -> Constraint Projection -> Final Scene           (37)

The generative model supplies diversity and a global layout prior; the
optimizer supplies collision-freedom, containment, clearance and functional
validity.  Neither is asked to do the other's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..core.scene import Room, Scene
from ..retarget.energy import EnergyWeights, TorchProblem, exact_energy
from ..retarget.optimizer import (RetargetConfig, RetargetResult, _apply_supports,
                                  _write_back, refine_continuous)
from ..retarget.populate import CooccurrenceModel, plan_population
from ..retarget.summarize import plan_summarization
from ..retarget.target import DesignIntent, build_design_intent
from ..intent.relations import SceneGraph
from .model import FlowModel
from .tokens import build_tokens, collate, from_frame

__all__ = ["sample_layouts", "generative_retarget", "load_flow"]


def _polish_priors(scene, intent):
    """Per-object arrays used by anisotropic + parent-child polish anchor.

    Returns (wall_aff, wall_normal, parent_idx) as numpy arrays of length N,
    where N = len(scene.objects) matches the polish's xy tensor layout.

      wall_aff[i]   in [0, 1]: how strongly object i hugged a wall in the
                    reference (from tokens.ref_wall_affinity, same source as
                    the flow's cond[-1] channel).  Anisotropic anchor kicks
                    in above ~0.3.

      wall_normal[i]: (2,) unit inward normal of the *target-room* wall
                    nearest object i's current xy.  Zeros for objects with
                    low wall_aff.

      parent_idx[i]: index of the motif's head object if i is a non-head
                    motif member, else -1.  Used to compute parent-relative
                    anchor: children ride with parent instead of pinning to
                    absolute world coords.
    """
    from ..geom.polygon import as_polygon
    from .tokens import ref_wall_affinity
    from shapely.geometry import Point as _P
    import numpy as _np

    N = len(scene.objects)
    aff = _np.zeros(N, dtype=_np.float32)
    wn = _np.zeros((N, 2), dtype=_np.float32)
    par = _np.full(N, -1, dtype=_np.int64)
    wd_init = _np.zeros(N, dtype=_np.float32)   # initial inward distance to wall

    # wall affinity from the reference scene
    src = intent.source
    aff_by_oid = {o.oid: ref_wall_affinity(o, src.room) for o in src.objects}
    for i, o in enumerate(scene.objects):
        aff[i] = float(aff_by_oid.get(o.oid, 0.0))

    # Wall inward normal per object.  Ground-truth-driven: find which wall the
    # object hugged in the *reference* scene, then use the same wall INDEX in
    # the target scene's polygon.  Works cleanly for uniform-scale deforms
    # (wall count preserved) and stays robust when the flow's sampled xy is
    # far off the true wall.  Falls back to forward direction if the wall
    # index can't be matched.
    import math as _m
    ref_poly = as_polygon(src.room)
    ref_ring = _np.asarray(ref_poly.exterior.coords)[:-1]
    ref_edges = [(ref_ring[k], ref_ring[(k + 1) % len(ref_ring)])
                 for k in range(len(ref_ring))]
    tgt_poly = as_polygon(scene.room)
    tgt_ring = _np.asarray(tgt_poly.exterior.coords)[:-1]
    tgt_edges = [(tgt_ring[k], tgt_ring[(k + 1) % len(tgt_ring)])
                 for k in range(len(tgt_ring))]

    # Signed area of a ring: positive = CCW, negative = CW.  For robust
    # "inward normal" we need the winding; a wrong sign flips the anisotropic
    # penalty and pushes objects OUT of the room instead of toward the wall.
    def _signed_area(ring):
        r = ring
        x = r[:, 0]; y = r[:, 1]
        return 0.5 * float(_np.sum(x * _np.roll(y, -1) - _np.roll(x, -1) * y))
    ring_orient = 1.0 if _signed_area(tgt_ring) > 0 else -1.0

    def _wall_normal(edge):
        a, b = edge
        d = b - a; L = _np.linalg.norm(d) + 1e-9
        t = d / L
        # CCW ring -> inward is (-ty, tx); CW ring -> flip
        return _np.array([-t[1] * ring_orient, t[0] * ring_orient],
                         dtype=_np.float32)

    ref_orient = 1.0 if _signed_area(ref_ring) > 0 else -1.0

    def _nearest_wall_index(edges_list, xy, orient):
        best_k, best_perp = -1, float("inf")
        for k, (a, b) in enumerate(edges_list):
            d = b - a; L = _np.linalg.norm(d) + 1e-9
            t = d / L
            n = _np.array([-t[1] * orient, t[0] * orient])
            rel = _np.array([float(xy[0]), float(xy[1])]) - a
            proj = float(_np.dot(rel, t))
            if proj < -0.05 or proj > L + 0.05:
                continue
            perp = float(abs(_np.dot(rel, n)))
            if perp < best_perp:
                best_perp = perp; best_k = k
        return best_k

    src_by_oid = {o.oid: o for o in src.objects}
    n_ref_walls = len(ref_edges)
    n_tgt_walls = len(tgt_edges)
    for i, o in enumerate(scene.objects):
        if aff[i] < 0.3:
            continue
        ref_o = src_by_oid.get(o.oid)
        if ref_o is None:
            continue
        k_ref = _nearest_wall_index(ref_edges, ref_o.xy, ref_orient)
        # map: if wall counts match, use same index; else fall back to forward
        if 0 <= k_ref and n_ref_walls == n_tgt_walls:
            wn[i] = _wall_normal(tgt_edges[k_ref])
        else:
            fx = -_m.sin(float(o.yaw)); fy = _m.cos(float(o.yaw))
            wn[i] = _np.array([fx, fy], dtype=_np.float32)
        # Safety check: the "inward" normal must actually point INTO the
        # polygon.  A small step from the wall midpoint along the normal
        # should land inside; if not, flip.  This catches any residual
        # winding-order or fallback-direction mistake.
        if 0 <= k_ref and n_ref_walls == n_tgt_walls:
            a, b = tgt_edges[k_ref]
            mid = (a + b) * 0.5
            probe = mid + wn[i] * 0.05
            if not tgt_poly.contains(_P(float(probe[0]), float(probe[1]))):
                wn[i] = -wn[i]
            # initial inward distance = (o.xy - a) · inward_normal
            wd_init[i] = float((o.xy - a) @ wn[i])

    # motif parent index: head is members[0]
    for m in intent.motifs:
        if len(m.members) < 2:
            continue
        head = m.members[0]
        for j in m.members[1:]:
            par[j] = head

    return aff, wn, par, wd_init


def load_flow(path: str, device: str = "cpu", use_ema: bool = True) -> FlowModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    m = FlowModel(cfg.get("d_model", 256), cfg.get("depth", 6),
                  cfg.get("heads", 8), geo_bias=cfg.get("geo_bias", False),
                  wall_tokens=cfg.get("wall_tokens", False),
                  parent_relative=cfg.get("parent_relative", False),
                  mask_flow=cfg.get("mask_flow", False)).to(device)
    # strict=False so older checkpoints (without the graph-stream params
    # added later) still load: the missing g_* tensors keep their init, and
    # g_fuse is zero-init so the graph branch contributes nothing until it is
    # trained.  Warm-start into a newer arch stays a no-op behaviourally.
    m.load_state_dict(ck["ema" if use_ema and "ema" in ck else "model"],
                      strict=False)
    # ReRoom 2.0 Module 1: carry the informative-prior flag so sampling starts
    # from the reference projection (not Gaussian) exactly as trained.
    m._prior_x0 = bool(cfg.get("prior_x0", False))
    m._prior_noise = float(cfg.get("prior_noise", 0.3))
    m._mask_flow = bool(cfg.get("mask_flow", False))
    m._mask_logit = float(cfg.get("mask_logit", 4.0))
    m.eval()
    return m


@torch.no_grad()
def sample_layouts(model: FlowModel, intent: DesignIntent, target_room: Room,
                   k: int = 8, steps: int = 50, device: str = "cpu",
                   temperature: float = 1.0, seed: int = 0,
                   guidance=None, return_keep: bool = False) -> np.ndarray:
    """Integrate the probability-flow ODE to draw ``k`` candidate layouts.

    Returns ``(k, N, 2)`` positions and ``(k, N)`` yaws, already mapped back
    from the room frame into metres.

    ``guidance`` (a :class:`GuidanceConfig`) turns on PhyScene-style in-sampling
    physical guidance: at every ODE step the predicted clean endpoint is scored
    for feasibility (out-of-floor, non-nestable collision) and its gradient
    nudges the trajectory.  This replaces the post-hoc constraint projection --
    feasibility stays inside the generative loop, and legitimate overlaps
    (a chair under its table) are left intact.
    """
    from .guidance import feasibility_grad
    item = build_tokens(intent, target_room, None)
    batch = collate([item] * k, device=device)
    frames = [item.meta["frame_tgt"]] * k
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(batch["state"].shape, generator=g).to(device)
    if getattr(model, "_prior_x0", False):
        # Module 1: start the ODE from the reference layout projected into the
        # target frame (cond ref_state) + noise, so the flow rectifies rather
        # than generates from scratch.
        prior = batch["cond"][..., 10:14]
        x_world = prior + getattr(model, "_prior_noise", 0.3) * noise
    else:
        x_world = noise * temperature
    # Step 2: the flow runs in the scale-invariant parent-relative space; the
    # world copy is only needed for guidance and the final readout.
    prel = getattr(model, "parent_relative", False)
    if prel:
        from .model import to_relative, to_world
        par = batch["parent"]; fh = batch["frame_h"]
        x = to_relative(x_world, par, fh)
    else:
        x = x_world
    dt = 1.0 / steps
    # D1: existence logit ell flows in parallel, from the informative "assume
    # keep" prior (+logit) toward +/-logit.  Sign at t=1 is the keep decision.
    mask_flow = getattr(model, "_mask_flow", False) or getattr(model, "mask_flow", False)
    if mask_flow:
        # uninformative existence prior (mean 0), matching training
        ell = torch.randn(
            x.shape[0], x.shape[1], 1, generator=torch.Generator(device="cpu").manual_seed(seed + 7)
        ).to(device)
    for s in range(steps):
        t = s * dt
        tau = torch.full((x.shape[0],), t, device=device)
        with torch.no_grad():
            if mask_flow:
                batch["ell"] = ell
            v = model(x, tau, batch)
        if mask_flow:
            ell = ell + dt * v[..., 4:5]
            v = v[..., :4]
        x = x + dt * v
        if guidance is not None and t >= guidance.start:
            # ramp the step in over the tail of the trajectory
            ramp = (t - guidance.start) / max(1.0 - guidance.start, 1e-6)
            x1_hat = x + (1.0 - t) * v
            x1_hat_world = to_world(x1_hat, par, fh) if prel else x1_hat
            # this function runs under @torch.no_grad(); the guidance needs the
            # graph, so re-enable it just for the feasibility gradient
            with torch.enable_grad():
                grad = feasibility_grad(x1_hat_world, batch, frames, guidance)
            if prel:
                xw = to_world(x, par, fh) - dt * ramp * grad
                x = to_relative(xw, par, fh)
            else:
                x = x - dt * ramp * grad
    if prel:
        x = to_world(x, par, fh)
    x = x.cpu().numpy()
    fr = item.meta["frame_tgt"]
    n = len(item.cat)
    xy = np.zeros((k, n, 2))
    yaw = np.zeros((k, n))
    for c in range(k):
        for i in range(n):
            p, a = from_frame(x[c, i], fr)
            xy[c, i] = p
            yaw[c, i] = a
    if return_keep:
        if mask_flow:
            # average the existence logit across restarts; keep iff > 0
            keep = (ell.mean(0).squeeze(-1).cpu().numpy() > 0.0)
        else:
            keep = np.ones(n, dtype=bool)
        return xy, yaw, keep
    return xy, yaw


def generative_retarget(model: FlowModel, graph: SceneGraph, target_room: Room,
                        elasticity=None, bank=None,
                        cooc: CooccurrenceModel | None = None,
                        cfg: RetargetConfig | None = None,
                        k: int = 12, steps: int = 50,
                        temperature: float = 1.0,
                        project: bool = False,
                        polish: bool = True,
                        guidance="default") -> RetargetResult:
    """Stage-two pipeline: propose with ``p_theta``, guided in-sampling (37).

    Three feasibility mechanisms coexist, following PDF eq (37) intent:

    * ``guidance`` -- in-sampling classifier-guidance during the ODE (PhyScene
      style, kept coherent by the flow's own prior; on by default)
    * ``polish``   -- a *light*, position-only constraint projection AFTER the
      guided sample, using all 7 energies of section 8 with anchor L2 to the
      sampled point so the surrogate's E_col cannot spread motifs apart.
      This is the eq (37) projection restored, but tuned to *coexist* with
      guidance instead of replacing it (on by default)
    * ``project``  -- the older, aggressive escalating projection.  Kept for
      ablation only; off by default.

    Pass ``guidance=None`` to disable in-sampling correction entirely.
    """
    from .guidance import GuidanceConfig
    _default_guidance = guidance == "default"
    cfg = cfg or RetargetConfig()
    rng = np.random.default_rng(cfg.seed)
    src = graph.scene
    intent = build_design_intent(graph, target_room, elasticity=elasticity)
    if _default_guidance:
        # The wall-pull is scaled by how the room changed.  A room that *shrank*
        # leaves its wall objects sitting near the walls, so a firm pull seats
        # them cleanly (measured: mean float 10->2 cm at 0.75x).  A room that
        # *grew* has large gaps and added furniture; the same firm pull there
        # fights the collision/clearance terms and shoves objects out of the
        # room (OOB 0->7.5%), so it is gentled -- growth coherence is the flow's
        # and population's job, not the pull's.
        pull = 2.5 if intent.area_ratio <= 1.0 else 1.5
        guidance = GuidanceConfig(wall_pull=pull)
    # intent.source aliases the caller's scene; work on a private copy so
    # shrink-to-fit substitution can resize the reference *before* layout
    # without mutating the graph the caller may reuse (e.g. across room sizes).
    intent.source = Scene(scene_id=src.scene_id, room=src.room,
                          objects=[o.copy() for o in src.objects],
                          source=src.source, meta=dict(src.meta or {}))

    sm = plan_summarization(intent, target_room, allow_drop=cfg.allow_removal)
    for i, o in enumerate(intent.source.objects):
        o.keep = bool(sm.keep[i])
    # PhyScene-style capacity gate (Summarise): a shrunk room cannot hold the
    # reference furniture AND stay walkable, so demote dining sets / shed
    # secondary items before the flow sees them.
    if getattr(cfg, "walkable", False):
        from ..retarget.walkable import capacity_prune
        capacity_prune(intent.source, target_room, min_walkable=getattr(cfg, "walkable_min", 0.55))

    # ---- substitution BEFORE layout (sections 11, 37) --------------------
    # A room that shrank cannot hold the reference furniture at its original
    # size (measured: object footprint up to 1.8x the density budget), so the
    # flow -- asked to place objects that do not fit -- overlaps them and the
    # collision guidance shoves the design apart, destroying the very relations
    # we retarget (S_rel 1.00 -> 0.64 at 0.75x).  Resolving fit *first* -- swap
    # each asset for the closest-fitting smaller real one -- lets the flow lay
    # out furniture that actually fits, so coherence survives.  Substitution
    # only acts when the room shrank; at >=1.0x it is a no-op, and growth is
    # population's job (kept after sampling, below).
    from ..retarget.retrieval import substitute_assets
    if cfg.allow_substitution and bank is not None and len(bank):
        substitute_assets(intent.source, intent, bank, rng=rng, **cfg.retrieval)

    out = Scene(scene_id=f"{src.scene_id}__flow", room=target_room.copy(),
                objects=[o.copy() for o in intent.source.objects],
                source="reroom_flow",
                meta={"source_scene": src.scene_id, "method": "flow"})
    for i, o in enumerate(out.objects):
        o.keep = bool(intent.source.objects[i].keep)   # includes capacity prune

    _mask_flow = getattr(model, "_mask_flow", False) or getattr(model, "mask_flow", False)
    _sl = sample_layouts(model, intent, target_room, k=k, steps=steps,
                         device=cfg.device, temperature=temperature,
                         seed=cfg.seed, guidance=guidance, return_keep=_mask_flow)
    if _mask_flow:
        xy, yaw, learned_keep = _sl
        # D1: the joint flow's existence head REPLACES the greedy Summarise
        # prune.  Override keep with the learned mask (anchors always survive).
        from ..retarget.summarize import _anchor_indices
        _anch = _anchor_indices(intent.source)
        for i, o in enumerate(intent.source.objects):
            o.keep = bool(learned_keep[i]) or (i in _anch)
        for i, o in enumerate(out.objects):
            o.keep = bool(intent.source.objects[i].keep)
        info_mask = {"learned_drop": int((~learned_keep).sum()),
                     "greedy_drop": int((~sm.keep).sum())}
    else:
        xy, yaw = _sl
        info_mask = {}
    # ---- partial-relational-transport SELECTION (design-identity pruning) ----
    # The flow places every object; keep-flags decide which survive.  When the
    # room forced pruning, override WHICH objects are kept so the surviving set
    # maximises retained design-graph relational mass (gwselect.relational_keep),
    # instead of the greedy Summarise/importance mask -- keeping the objects that
    # carry the reference's relational identity.  Anchors never dropped.
    if getattr(cfg, "relational_select", False):
        from ..retarget.gwselect import relational_keep
        from ..retarget.summarize import _anchor_indices
        K = int(sum(1 for o in out.objects if o.keep))
        if 0 < K < len(out.objects):
            anch = _anchor_indices(intent.source)
            km = relational_keep(graph, K, must_keep=anch)
            for i, o in enumerate(out.objects):
                o.keep = bool(km[i])
            for i, o in enumerate(intent.source.objects):
                o.keep = bool(km[i])

    info: dict = {"k": k, "steps": steps, "summarization": sm.log, **info_mask,
                  "projected": bool(project),
                  "polished": bool(polish),
                  "guided": guidance is not None,
                  "relational_select": bool(getattr(cfg, "relational_select", False))}

    # ---- polish (eq (37) constraint projection, coexisting with guidance) ----
    # A single light Adam pass on the surrogate energy, all 7 terms of
    # section 8 active via ``TorchProblem``.  Key differences from the older
    # ``project=True`` path that broke coherence:
    #   * anchor_w > 0 keeps every projected point close to its sampled value,
    #     so E_col cannot spread a same-motif pair apart to zero out overlap;
    #   * a single pass, not an escalating series -- feasibility is already
    #     mostly satisfied by the in-sampling guidance, so this only polishes
    #     the residual (wall-hug gaps, small OOB corners, cross-motif clearance);
    #   * freeze_yaw kept (the flow's orientation prior beats the surrogate's
    #     local minima).
    # This is where E_rel, E_style enter the layout (not just the ranker):
    # E_rel weighted by ``cfg.weights.rel`` pulls objects back toward their
    # reference relative positions, which is what preserves motifs; E_style
    # gently favours reference-consistent orientations.
    if polish:
        problem = TorchProblem(out, intent, cfg.weights, device=cfg.device)
        # Anisotropic + parent-child anchor: fix the "wall_pull vs anchor L2"
        # deadlock so wall-affinity objects can slide inward without a stiff
        # spring, while motif children ride with their parent when it moves.
        aff, wnorm, par, wd_init = _polish_priors(out, intent)
        xy, yaw, e_poll, _ = refine_continuous(
            problem, xy, yaw, steps=cfg.polish_steps, lr=cfg.polish_lr,
            freeze_yaw=True, anchor_w=cfg.polish_anchor,
            wall_aff=aff, wall_normal=wnorm, parent_idx=par,
            wall_dist_init=wd_init,
            anchor_w_tangent=getattr(cfg, "polish_anchor_tangent", 100.0),
            anchor_w_normal=getattr(cfg, "polish_anchor_normal", 8.0),
            phase1_frac=getattr(cfg, "polish_phase1_frac", 0.0),
            phase1_col_scale=getattr(cfg, "polish_phase1_col", 1.0),
            phase1_tan_scale=getattr(cfg, "polish_phase1_tan", 1.0),
            phase1_wall_scale=getattr(cfg, "polish_phase1_wall", 1.0))
        info["polish_steps"] = int(cfg.polish_steps)

    if project:
        problem = TorchProblem(out, intent, cfg.weights, device=cfg.device)
        # freeze_yaw: keep the proposal's orientation and correct only
        # positions.  The flow prior orients furniture the way real rooms do
        # (measured: 89 % of wall objects within 0.5 deg of parallel, median
        # skew 0.58 deg); letting the surrogate re-optimise yaw during
        # projection collapses that back to the optimiser's few-degree local
        # minimum.  This is the eq. (37) division of labour made literal --
        # the transformer owns orientation, the projection owns feasibility.
        xy, yaw, e, _ = refine_continuous(problem, xy, yaw, cfg.grad_steps,
                                          cfg.lr, freeze_yaw=True)
        # escalating projection: the proposal is a good layout prior but knows
        # nothing about hard constraints, so feasibility weights are raised
        # until the best candidate is actually legal (37)
        for scale in (cfg.projection_scale, cfg.projection_scale * 4.0):
            xy, yaw, e, _ = refine_continuous(problem, xy, yaw, cfg.proj_steps,
                                              cfg.lr * 0.35, scale,
                                              freeze_yaw=True)
            best_c = int(np.argmin(e))
            _write_back(out, xy[best_c], yaw[best_c])
            _apply_supports(out, intent)
            ex = exact_energy(out, intent, cfg.weights)
            info["last_projection_scale"] = scale
            if ex["E_bound"] <= cfg.bound_tol and ex["E_col"] <= cfg.col_tol:
                break
        order = np.argsort(e)[:cfg.exact_topk]
    else:
        order = range(len(xy))

    best, best_ex, best_score = None, None, None
    _nav = getattr(cfg, "walkable", False)
    for c in order:
        _write_back(out, xy[c], yaw[c])
        _apply_supports(out, intent)
        ex = exact_energy(out, intent, cfg.weights)
        # Ranking (PhyScene nav): penalise candidates that seal a corridor even
        # when collision-free, so a walkable layout wins the k-way selection.
        score = ex["E"]
        if _nav:
            from ..retarget.walkable import nav_penalty
            score = score + nav_penalty(out)
        if best_score is None or score < best_score:
            best, best_ex, best_score = int(c), ex, score
    _write_back(out, xy[best], yaw[best])
    _apply_supports(out, intent)

    # ---- growth (section 10) and substitution (section 11) --------------
    # The transformer can only place the objects it has tokens for -- the
    # reference set.  The remaining two-thirds of eq. (18), "selection +
    # substitution", are handled around it, exactly as in the optimiser path:
    #   * population proposes furniture to fill a room that grew, placed by the
    #     planner relative to the surviving motifs (the flow cannot invent
    #     tokens for objects with no reference relations);
    #   * substitution swaps each asset for the closest-fitting real one in the
    #     bank, so a reference sofa too large for the target becomes a smaller
    #     sofa of similar style rather than a non-physically shrunk one;
    #   * a feasibility gate then admits each addition only while it costs the
    #     layout nothing -- filling a bigger room is the right instinct, doing
    #     it for free is not.
    # growth/population runs *after* layout: it places new furniture relative
    # to the sampled scene to fill a room that grew.  Substitution (shrink to
    # fit) has already run before sampling, above.
    from ..retarget.optimizer import _add_wall_targets, _vet_additions
    if cfg.allow_addition:
        pop = plan_population(intent, target_room,
                              np.array([o.keep for o in out.objects], dtype=bool),
                              cooc, bank, rng)
        if pop is not None and pop.additions:
            n_ref = len(out.objects)
            for o in pop.additions:
                a = o.copy()
                a.meta = dict(a.meta or {}); a.meta["added"] = True
                out.objects.append(a)
            _add_wall_targets(intent, out, n_ref, target_room)
    if cfg.allow_addition and any(o.meta.get("added") for o in out.objects):
        _vet_additions(out, intent, cfg)

    out.objects = [o for o in out.objects if o.keep]
    if getattr(cfg, "regularity_snap", False):
        # ReRoom 2.0 Step 1: snap to orthogonal / wall-flush / slot structure.
        from ..retarget.regularity import regularity_snap as _reg_snap
        _reg_snap(out, intent)
    if getattr(cfg, "walkable", False):
        # PhyScene-style door-box + affordance push (Polish): clear a metre
        # inside every door and open human-activity buffers.
        from ..retarget.walkable import walkable_push
        walkable_push(out)
    return RetargetResult(scene=out, intent=intent,
                          energy=exact_energy(out, intent, cfg.weights),
                          info=info)
