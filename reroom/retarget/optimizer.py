"""Geometry-aware scene retargeting (plan section 8).

Retargeting is *placement + selection + substitution* (18), not an affine
rescale of the reference coordinates.  The solver is a three-level loop:

1. **Capacity planning** -- summarization (9) if the target room is smaller,
   population (10) if it is larger, so the number of objects is right before
   any geometry is fitted.
2. **Intent-aware initialisation** -- motifs are placed as rigid units, wall
   objects are snapped to the *matched* target wall, and intra-motif offsets are
   stretched by relation elasticity (9).  This is the layout the paper calls
   "the design moved, not the coordinates".
3. **Continuous refinement + repair** -- Adam on the differentiable surrogate,
   batched over random restarts, followed by a feasibility projection at raised
   constraint weights; anything still infeasible is repaired by substitution and,
   only as a last resort, by removal.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import prior
from ..core.scene import ObjectInstance, Room, Scene
from ..data.asset_bank import AssetBank
from ..geom.polygon import (as_polygon, erode, min_rotated_rect_params,
                            object_polygon, sample_interior)
from ..intent.motifs import Motif
from ..intent.relations import SceneGraph, build_scene_graph
from .energy import EnergyWeights, TorchProblem, exact_energy
from .populate import CooccurrenceModel, plan_population
from .retrieval import substitute_assets
from .summarize import plan_summarization
from .target import DesignIntent, build_design_intent

__all__ = ["RetargetConfig", "RetargetResult", "retarget", "initial_layout",
           "refine_continuous"]


@dataclass
class RetargetConfig:
    restarts: int = 24
    grad_steps: int = 200
    proj_steps: int = 90
    lr: float = 0.06
    outer_iters: int = 3
    device: str = "cpu"
    seed: int = 0
    allow_removal: bool = True
    allow_resize: bool = True        # s_i of eq. (17) as a continuous variable
    vet_additions: bool = True       # an added object must cost no feasibility
    addition_tol: float = 0.02       # how much feasibility an addition may cost
    allow_addition: bool = True
    allow_substitution: bool = True
    use_motif_init: bool = True
    regularity_snap: bool = True     # ReRoom 2.0 (shipped main method): 1-step
                                     # layout to orthogonal / wall-flush / slot
                                     # structure (regularity projector).
    walkable: bool = False           # PhyScene-style walkability: capacity prune
                                     # (Summarise) + door-box/affordance push
                                     # (Polish) + A* nav penalty (Ranking).
    walkable_min: float = 0.55       # free-floor ratio below which capacity
                                     # prune kicks in.
    relational_select: bool = False  # partial-relational-transport SELECTION:
                                     # when the room forces pruning, choose WHICH
                                     # objects to keep by maximising retained
                                     # design-graph relational mass (gwselect.
                                     # relational_keep) rather than the greedy
                                     # importance/Summarise mask.  Preserves more
                                     # design identity under capacity (probe:
                                     # +0.12 S_rel over the flow's mask).
    use_elasticity: bool = True
    weights: EnergyWeights = field(default_factory=EnergyWeights)
    projection_scale: float = 6.0
    exact_topk: int = 4
    # polish step used by the generative pipeline: a light, position-only
    # constraint projection AFTER in-sampling guidance (eq (37) coexisting).
    # anchor_w > 0 keeps projected positions close to the sampled ones so the
    # surrogate's E_col cannot spread motifs apart while it fixes wall-hug
    # gaps and residual OOB.  Steps kept short (guidance did the heavy lift).
    polish_steps: int = 25            # 50-step gave better col/OOB but *worse*
                                      # 1.35x wall float (28 vs 22 cm): more
                                      # polish time lets E_col push wall pieces
                                      # further inward to make space for others,
                                      # overpowering the asymmetric anchor.
                                      # 25 steps is the shipped balance.
    polish_lr: float = 0.02
    polish_anchor: float = 100.0
    # Two-phase polish schedule (user-proposed, Plan A).  Phase 1 = first
    # ~half of polish_steps: weaken E_col and the tangent anchor so wall-
    # affinity objects can slide diagonally to the wall without being
    # blocked by transient collisions or a stiff tangential spring.  Phase 2
    # restores full weights for local cleanup.  Defaults chosen so this is
    # ON by default -- Plan A analysis suggests it directly addresses the
    # "multi-energy tug-of-war + diagonal locking" root cause behind the
    # 1.35x residual float.
    # Plan A + Phase 2.2 tuning: two-phase schedule with LONGER Phase 1
    # (65% instead of 50%) and 3× wall-pull boost during Phase 1 so wall
    # objects reach the wall before E_col comes back on.
    polish_phase1_frac: float = 0.65
    polish_phase1_col: float = 0.15
    polish_phase1_tan: float = 0.10
    polish_phase1_wall: float = 3.0    # E_func (wall_flush) scale during Phase 1
    # feasibility tolerances are *fractions of the total furniture footprint*
    # (plus a small absolute floor), because 0.04 m^2 of overhang means one
    # thing in a studio and another in a 40 m^2 living room -- and because an
    # absolute threshold made the repair loop delete a 2.4 m^2 sofa to fix a
    # 0.04 m^2 overlap
    bound_tol: float = 0.004
    col_tol: float = 0.004
    tol_floor: float = 0.01          # m^2
    tol_at_least_reference: bool = True
    # eq. (30) weights: appearance, size, and the f^geo shape term
    retrieval: dict = field(default_factory=dict)
    protect_anchor: float = 0.9      # never repair-delete these unless they
                                     # are themselves badly out of bounds
    verbose: bool = False


@dataclass
class RetargetResult:
    scene: Scene
    intent: DesignIntent
    energy: dict
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------
def _mrr_frame(poly):
    long_, short_, ang = min_rotated_rect_params(poly)
    c = np.asarray(poly.centroid.coords[0])
    a1 = np.array([math.cos(ang), math.sin(ang)])
    a2 = np.array([-a1[1], a1[0]])
    return c, a1, a2, long_ / 2.0, short_ / 2.0, ang


def _rigid_point(p, src, tgt):
    """Move a point between rooms by rotation and translation only.

    The room-frame affine map scales every offset by how much the room grew --
    that is ``alpha = 1`` applied unconditionally.  This is its opposite:
    the layout is carried over at its original metric size, which is
    ``alpha = 0``.  Real placements live between the two.
    """
    cs, a1s, a2s, hls, hss, angs = src
    ct, a1t, a2t, hlt, hst, angt = tgt
    d = np.asarray(p, dtype=float) - cs
    u = np.array([float(np.dot(d, a1s)), float(np.dot(d, a2s))])
    return ct + u[0] * a1t + u[1] * a2t


def _external_alpha(intent, n: int) -> np.ndarray:
    """Per-object elasticity, from the relations that reach *outside* its motif.

    An object's own motif holds it together at body scale; what decides how far
    it drifts as the room changes is its relations to the rest of the room.  So
    the blend between a rigid and a scaled placement is driven by the mean
    alpha over an object's external relations, weighted by relation weight.
    """
    motif_of = {}
    for m in intent.motifs:
        for i in m.members:
            motif_of[i] = m.mid
    num = np.zeros(n)
    den = np.zeros(n)
    for r in intent.relations:
        if r.i >= n or r.j >= n:
            continue
        if motif_of.get(r.i, f"_{r.i}") == motif_of.get(r.j, f"_{r.j}"):
            continue                      # internal to a motif: not the question
        w = float(r.weight)
        num[r.i] += w * r.alpha
        num[r.j] += w * r.alpha
        den[r.i] += w
        den[r.j] += w
    out = np.full(n, np.nan)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def _map_point(p, src, tgt):
    cs, a1s, a2s, hls, hss, angs = src
    ct, a1t, a2t, hlt, hst, angt = tgt
    d = np.asarray(p, dtype=float) - cs
    u = np.array([float(np.dot(d, a1s)) / max(hls, 1e-6),
                  float(np.dot(d, a2s)) / max(hss, 1e-6)])
    u = np.clip(u, -1.2, 1.2)
    return ct + u[0] * hlt * a1t + u[1] * hst * a2t


def _clamp_into(xy: np.ndarray, objs, room: Room, margin: float = 0.05) -> None:
    """Pull an initial position back inside the room, and no further.

    An earlier version projected onto the polygon *eroded* by the object's
    radius.  In a concave room -- an L or a cross -- eroding by a metre leaves
    a small central blob, and every object in an arm of the room was yanked
    into it: the rigid initialisation came out with E_rel = 84.6 where the
    reference scores 0, and the optimiser then "improved" on it by collapsing
    the layout.  Clamp against the room itself, and only for points that are
    actually outside it.
    """
    from shapely.geometry import Point
    poly = as_polygon(room)
    inner = poly.buffer(-margin)
    if inner.is_empty:
        inner = poly
    if inner.geom_type == "MultiPolygon":
        inner = max(inner.geoms, key=lambda g: g.area)
    for i, o in enumerate(objs):
        if o.locked:
            continue                       # C_t: pinned poses are not clamped
        pt = Point(float(xy[i, 0]), float(xy[i, 1]))
        if poly.contains(pt):
            continue
        try:
            q = inner.exterior.interpolate(inner.exterior.project(pt))
            xy[i] = [q.x, q.y]
        except Exception:
            xy[i] = np.asarray(poly.centroid.coords[0])


def _wall_pose(target_room: Room, wall: int, t: float, gap: float,
               half_depth: float):
    walls = target_room.walls()
    a, b = walls[wall % len(walls)]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    tt = d / L
    n = np.array([-tt[1], tt[0]])
    t = float(np.clip(t, 0.05, 0.95))
    xy = a + tt * (t * L) + n * (gap + half_depth)
    yaw = math.atan2(n[1], n[0]) - math.pi / 2
    return xy, yaw


def initial_layout(intent: DesignIntent, target_room: Room, keep: np.ndarray,
                   restarts: int, rng: np.random.Generator,
                   use_motif_init: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Motif-aware, wall-snapped, elasticity-stretched initial layouts."""
    scene = intent.source
    objs = scene.objects
    n = len(objs)
    src = _mrr_frame(as_polygon(scene.room))
    tgt = _mrr_frame(as_polygon(target_room))
    dang = tgt[5] - src[5]

    # Where an object starts is decided by its own elasticity, not by a blanket
    # affine map.  Using the map for everything silently applied alpha = 1 to
    # exactly the long, cross-room relations alpha exists to govern, which left
    # the ablation testing nothing.
    ext_alpha = _external_alpha(intent, n)
    base_xy = np.zeros((n, 2))
    base_yaw = np.zeros(n)
    for i, o in enumerate(objs):
        if o.locked:
            base_xy[i], base_yaw[i] = o.xy, o.yaw
            continue
        p_aff = _map_point(o.xy, src, tgt)
        a = ext_alpha[i]
        if np.isnan(a):
            base_xy[i] = p_aff            # nothing to go on: keep the old map
        else:
            p_rig = _rigid_point(o.xy, src, tgt)
            base_xy[i] = (1.0 - a) * p_rig + a * p_aff
        base_yaw[i] = o.yaw + dang
    # a rigid placement can land outside a shrunken room; pull it back inside
    _clamp_into(base_xy, objs, target_room)

    wall_lut = {w.i: w for w in intent.walls}
    if use_motif_init:
        # relation scale factor between head and member, from elasticity
        srel = {}
        for r in intent.relations:
            s = (1.0 - r.alpha) + r.alpha * r.gamma
            srel[(r.i, r.j)] = s
            srel[(r.j, r.i)] = s
        for m in intent.motifs:
            h = m.head
            if h >= n:
                continue
            head = objs[h]
            wt = wall_lut.get(h)
            if wt is not None:
                pose = _wall_pose(target_room, wt.wall, wt.t, wt.gap,
                                  float(head.half[1]))
                if pose is not None:
                    base_xy[h], base_yaw[h] = pose
            hx, hyaw = base_xy[h], base_yaw[h]
            c, s = math.cos(hyaw - head.yaw), math.sin(hyaw - head.yaw)
            rot = np.array([[c, -s], [s, c]])
            for j in m.members:
                if j == h or j >= n:
                    continue
                off = objs[j].xy - head.xy
                sc = float(np.clip(srel.get((h, j), 1.0), 0.5, 2.0))
                # rigid motifs barely stretch; loose ones follow the room
                sc = 1.0 + (sc - 1.0) * (1.0 - m.rigidity)
                base_xy[j] = hx + rot @ (off * sc)
                base_yaw[j] = objs[j].yaw + (hyaw - head.yaw)
    # non-motif wall objects still snap
    in_motif = {i for m in intent.motifs for i in m.members}
    for i, wt in wall_lut.items():
        if i in in_motif or i >= n:
            continue
        pose = _wall_pose(target_room, wt.wall, wt.t, wt.gap, float(objs[i].half[1]))
        if pose is not None:
            base_xy[i], base_yaw[i] = pose

    # Wall snapping and motif placement run after the base pass and would
    # happily overwrite a pinned object, so its pose is restored once at the
    # end -- one place to be right, instead of a guard in every branch.
    for i, o in enumerate(objs):
        if o.locked:
            base_xy[i], base_yaw[i] = o.xy, o.yaw

    poly = as_polygon(target_room)
    xy = np.repeat(base_xy[None], restarts, axis=0)
    yaw = np.repeat(base_yaw[None], restarts, axis=0)

    # Restart 1 is the plain room-frame affine map, with no elasticity blend.
    # A rigid placement is the right *relative* geometry but it can start
    # objects on top of each other in a reshaped room, and the refinement does
    # not always dig out: collisions rose from 0.23 % to 0.38 % when the blend
    # was introduced.  Keeping the affine layout as its own candidate means the
    # blend competes on exact energy and can only help.
    if restarts > 1:
        for i, o in enumerate(objs):
            xy[1, i] = _map_point(o.xy, src, tgt)
            yaw[1, i] = o.yaw + dang
        for i, wt in wall_lut.items():
            if i >= n:
                continue
            pose = _wall_pose(target_room, wt.wall, wt.t, wt.gap,
                              float(objs[i].half[1]))
            if pose is not None:
                xy[1, i], yaw[1, i] = pose
        # restart 1 is rebuilt from scratch, so it needs the same pin restore
        # the base pass got -- otherwise a pinned object silently moves
        # whenever this candidate happens to win the exact-energy ranking
        for i, o in enumerate(objs):
            if o.locked:
                xy[1, i], yaw[1, i] = o.xy, o.yaw
        _clamp_into(xy[1], objs, target_room)

    # `free` below already names the eroded interior polygon, so the pin mask
    # gets its own name rather than shadowing it
    unpinned = np.array([0.0 if o.locked else 1.0 for o in objs])[:, None]
    for r in range(2, restarts):
        amp = 0.10 + 0.55 * (r / max(restarts - 1, 1))
        xy[r] += rng.normal(0, amp, size=(n, 2)) * unpinned
        yaw[r] += rng.normal(0, 0.10 + 0.55 * amp, size=n) * unpinned[:, 0]
        if r >= restarts * 0.75:
            # a few fully random restarts for the loose objects
            free = erode(poly, 0.25)
            if not free.is_empty:
                pts = sample_interior(free, n, rng)
                loose = [i for i in range(n)
                         if keep[i] and prior(objs[i].category).wall < 0.5
                         and i not in in_motif and not objs[i].locked]
                for i in loose:
                    xy[r, i] = pts[i]
                    yaw[r, i] = rng.uniform(0, 2 * math.pi)
    return xy, yaw


# --------------------------------------------------------------------------
# continuous refinement
# --------------------------------------------------------------------------
def refine_continuous(problem: TorchProblem, xy0: np.ndarray, yaw0: np.ndarray,
                      steps: int, lr: float, weights_scale: float = 1.0,
                      allow_resize: bool = False, s0: np.ndarray | None = None,
                      freeze_yaw: bool = False,
                      anchor_w: float = 0.0,
                      wall_aff: np.ndarray | None = None,
                      wall_normal: np.ndarray | None = None,
                      anchor_w_tangent: float = 100.0,
                      anchor_w_normal: float = 8.0,
                      parent_idx: np.ndarray | None = None,
                      wall_dist_init: np.ndarray | None = None,
                      phase1_frac: float = 0.0,
                      phase1_col_scale: float = 1.0,
                      phase1_tan_scale: float = 1.0,
                      phase1_wall_scale: float = 1.0):
    """Adam on the differentiable surrogate, batched over restarts.

    Objects the user pinned have their gradients zeroed rather than merely
    penalised: ``C_t`` says they do not move, and a soft penalty would trade
    them away whenever the rest of the room got difficult enough.

    ``freeze_yaw`` holds every object's orientation fixed and lets only the
    positions move.  This is what the constraint-projection stage of the
    generative pipeline (eq. 37) needs: the flow proposal already orients
    furniture the way a trained-on-real-rooms prior does -- 89 % of wall
    objects within half a degree of parallel -- and re-optimising yaw against
    the surrogate energy only drags it back into the same local minimum the
    pure-optimiser path sits in (a sofa a few degrees off square).  Projection
    should *correct collisions and boundary*, not re-derive orientation.
    """
    torch = problem.torch
    dev = problem.device
    xy = torch.tensor(xy0, dtype=torch.float32, device=dev, requires_grad=True)
    yaw = torch.tensor(yaw0, dtype=torch.float32, device=dev,
                       requires_grad=not freeze_yaw)
    log_s = None
    if allow_resize:
        base = (np.zeros_like(yaw0) if s0 is None else np.asarray(s0))
        log_s = torch.tensor(base, dtype=torch.float32, device=dev,
                             requires_grad=True)
    frozen = problem.locked.bool()
    if bool(frozen.any()):
        def _freeze(g):
            g = g.clone()
            g[:, frozen] = 0.0
            return g
        xy.register_hook(_freeze)
        if not freeze_yaw:
            yaw.register_hook(lambda g: g.masked_fill(frozen[None], 0.0))
        if log_s is not None:
            log_s.register_hook(lambda g: g.masked_fill(frozen[None], 0.0))
    if weights_scale != 1.0:
        # Escalate the whole hard-constraint group together.  Listing the
        # fields by hand here silently dropped the C_t weights and reset
        # `func_reach`, so the keep-out term got *relatively weaker* at exactly
        # the moment feasibility was being enforced hardest.
        old = problem.w
        problem.w = problem.w.escalated(weights_scale)
    params = [xy] + ([yaw] if not freeze_yaw else []) \
        + ([log_s] if log_s is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    # ``anchor_w > 0`` keeps the projected positions close to their initial
    # values.  This turns the projection into a *polish* around a proposal:
    # feasibility gets fixed (a wall gap closes, an OOB corner comes in) but
    # the layout cannot rearrange itself, so the coherence of the sampled
    # motifs is preserved.  Without it, the surrogate's E_col gradient will
    # spread even legitimately-close pairs (a sofa and its side_table) apart
    # to drive collision to zero.
    #
    # Anisotropic form (wall_aff + wall_normal provided): for objects with
    # high wall affinity, the anchor is *decomposed* into a strong tangential
    # component (parallel to the nearest wall — keeps the wall-line formation)
    # and a WEAK normal component (toward/away the wall — lets wall_pull /
    # E_func slide the object inward without a stiff opposing spring).  This
    # is the fix for the "20 cm float locked in by uniform anchor" deadlock.
    #
    # Parent-child form (parent_idx provided): a motif child's anchor is
    # against (xy_child - xy_parent) rather than absolute xy_child.  When
    # E_func pulls the parent to the wall, the child rides along by
    # construction, preserving the motif's internal geometry.
    anchor_xy = xy.detach().clone() if anchor_w > 0.0 else None
    aff_t = norm_t = par_t = None
    if anchor_xy is not None:
        dev_a = xy.device
        if wall_aff is not None:
            aff_t = torch.as_tensor(wall_aff, dtype=torch.float32, device=dev_a)
        if wall_normal is not None:
            norm_t = torch.as_tensor(wall_normal, dtype=torch.float32, device=dev_a)
        if parent_idx is not None:
            par_t = torch.as_tensor(parent_idx, dtype=torch.long, device=dev_a)
        wd_init_t = None
        if wall_dist_init is not None:
            wd_init_t = torch.as_tensor(wall_dist_init, dtype=torch.float32,
                                         device=dev_a)
    # Two-phase scheduling: Phase 1 (first `phase1_frac` of steps) weakens
    # E_col and the tangent anchor so wall-hugging pieces (with any motif
    # children they carry) can slide diagonally to their target wall without
    # being blocked by transient collision spikes or a stiff tangential
    # spring.  Phase 2 restores the full weights for local collision cleanup
    # once the coarse "get to the wall" step has already happened.
    p1_steps = int(steps * phase1_frac) if phase1_frac > 0.0 else 0
    _w_col_full = float(problem.w.col)
    _w_func_full = float(problem.w.func)
    for step_i in range(steps):
        in_p1 = step_i < p1_steps
        if p1_steps > 0:
            problem.w.col = _w_col_full * (phase1_col_scale if in_p1 else 1.0)
            problem.w.func = _w_func_full * (phase1_wall_scale if in_p1 else 1.0)
        eff_tan = anchor_w_tangent * (phase1_tan_scale if in_p1 else 1.0)
        e = problem.energy(xy, yaw, log_s)
        loss = e.sum()
        if anchor_xy is not None:
            if par_t is not None:
                # replace child's anchor with parent-relative offset
                # anchor_xy_child_rel = anchor_xy[child] - anchor_xy[parent]
                # current_offset = xy[child] - xy[parent]
                # penalise |current_offset - anchor_offset|
                child_mask = (par_t >= 0)
                if bool(child_mask.any()):
                    p = par_t.clamp(min=0)
                    a_off = anchor_xy - anchor_xy[:, p, :]         # (R, N, 2)
                    c_off = xy - xy[:, p, :]                       # (R, N, 2)
                    delta_rel = (c_off - a_off) * child_mask[None, :, None]
                    delta_abs = (xy - anchor_xy) * (~child_mask)[None, :, None].float()
                else:
                    delta_rel = xy.new_zeros(xy.shape)
                    delta_abs = xy - anchor_xy
            else:
                delta_rel = xy.new_zeros(xy.shape)
                delta_abs = xy - anchor_xy
            delta = delta_abs + delta_rel                          # (R, N, 2)
            # anisotropic split for wall-affinity objects
            if aff_t is not None and norm_t is not None:
                # normalise the wall normal (world-space, inward)
                nn = norm_t / (norm_t.norm(dim=-1, keepdim=True) + 1e-6)   # (N, 2)
                tan = torch.stack([-nn[..., 1], nn[..., 0]], dim=-1)       # (N, 2)
                # scalar components of delta along tangent / normal
                d_n_s = (delta * nn[None, :, :]).sum(-1)                   # (R, N)
                d_t_s = (delta * tan[None, :, :]).sum(-1)                  # (R, N)
                gate = aff_t[None, :]                                       # (1, N)
                # Plan B — distance-gated tangent: the tangent anchor only
                # kicks in when the object is *close* to its wall.  Far from
                # the wall the object needs to slide diagonally around
                # obstacles; a stiff tangent spring blocks that path
                # ("diagonal locking").  Close to the wall we want the
                # tangent formation locked to preserve reference arrangement.
                # Current inward distance = initial dist + (delta · n).
                if wd_init_t is not None:
                    curr_dist = wd_init_t[None, :] + d_n_s.detach()   # (R, N)
                    # sigmoid ramp: near ≤ 5 cm -> 1.0, far ≥ 15 cm -> ~0.1
                    dist_gate = torch.sigmoid((0.10 - curr_dist) / 0.03)
                else:
                    dist_gate = torch.ones_like(d_t_s)
                pen_wall_tan = eff_tan * (d_t_s ** 2) * gate * dist_gate
                # normal: ASYMMETRIC single-sided penalty — only penalise moving
                # *inward* (away from the wall).  d_n_s > 0 means the object
                # drifted toward room centre (bad).  d_n_s < 0 means it moved
                # further outward toward the wall (this is exactly what
                # wall_pull / E_func want, so give it a free pass).
                inward_drift = torch.relu(d_n_s)
                pen_wall_nrm = anchor_w_normal * (inward_drift ** 2) * gate
                pen_free     = anchor_w * (delta ** 2).sum(-1) * (1.0 - gate)
                loss = loss + pen_wall_tan.sum() + pen_wall_nrm.sum() + pen_free.sum()
            else:
                loss = loss + anchor_w * (delta ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 20.0)
        opt.step()
        sched.step()
    if weights_scale != 1.0:
        problem.w = old
    # restore col & func weights if we scaled them in the phase-1 schedule
    if p1_steps > 0:
        problem.w.col = _w_col_full
        problem.w.func = _w_func_full
    with torch.no_grad():
        e = problem.energy(xy, yaw, log_s)
    ls = (log_s.detach().clamp(-problem.max_log_s, problem.max_log_s)
          .cpu().numpy() if log_s is not None else None)
    return (xy.detach().cpu().numpy(), yaw.detach().cpu().numpy(),
            e.detach().cpu().numpy(), ls)


def _vet_additions(scene: Scene, intent: DesignIntent,
                   cfg: "RetargetConfig") -> None:
    """Make every added object earn its place, eq. (29) notwithstanding.

    Population was the one stage that fired unconditionally.  Summarization is
    checked by the repair loop and substitution by its own gain test, but
    anything the population planner proposed simply went in -- and measured
    over 180 (scene, level) pairs that cost 0.020 of the joint score and
    0.014 of legality in exactly the cases where it fired, while buying no
    feasibility at all.  Filling a bigger room is still the right instinct
    (section 10); doing it for free is the part that was wrong.

    So the additions are re-admitted one at a time, best first, and only while
    they cost nothing in feasibility over the layout without them.
    """
    added = [o for o in scene.objects if o.keep and o.meta.get("added")]
    if not added:
        return
    feas = lambda: sum(exact_energy(scene, intent, cfg.weights)[k]
                       for k in ("E_bound", "E_col", "E_clear"))
    for o in added:
        o.keep = False
    base = feas()
    tol = max(cfg.addition_tol * max(base, 1.0), cfg.tol_floor)

    # largest first: a big piece is what actually restores the reference's
    # density, and small clutter is what fragments the floor
    order = sorted(added, key=lambda o: -o.footprint_area)
    kept = 0
    for o in order:
        o.keep = True
        if feas() > base + tol:
            o.keep = False
        else:
            kept += 1
    scene.meta["additions_proposed"] = len(added)
    scene.meta["additions_kept"] = kept


def _write_back(scene: Scene, xy: np.ndarray, yaw: np.ndarray,
                log_s: np.ndarray | None = None,
                base: np.ndarray | None = None) -> Scene:
    for i, o in enumerate(scene.objects):
        o.position[0] = float(xy[i, 0])
        o.position[1] = float(xy[i, 1])
        o.yaw = float(yaw[i])
        if log_s is not None and base is not None:
            # s_i of eq. (17): a footprint trim, always measured against the
            # retrieved asset's own extents so repeated write-backs do not
            # compound into a drift
            f = float(np.exp(log_s[i]))
            o.size[0], o.size[1] = base[i, 0] * f, base[i, 1] * f
    return scene


def _apply_supports(scene: Scene, intent: DesignIntent) -> None:
    """Keep a supported object sitting on its base after the base has moved."""
    for r in intent.relations:
        if r.kind != "support":
            continue
        if r.i >= len(scene.objects) or r.j >= len(scene.objects):
            continue
        base, top = scene.objects[r.i], scene.objects[r.j]
        if not (base.keep and top.keep) or top.locked:
            continue
        off = r.phi_ref[0] * base.right + r.phi_ref[1] * base.forward
        top.position[0] = base.position[0] + float(off[0])
        top.position[1] = base.position[1] + float(off[1])
        top.position[2] = base.top
        top.yaw = base.yaw + math.atan2(float(r.phi_ref[3]), float(r.phi_ref[2]))


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def retarget(graph: SceneGraph, target_room: Room,
             elasticity=None, bank: AssetBank | None = None,
             cooc: CooccurrenceModel | None = None,
             cfg: RetargetConfig | None = None) -> RetargetResult:
    """Retarget the reference design in ``graph`` into ``target_room``."""
    cfg = cfg or RetargetConfig()
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    src = graph.scene

    intent = build_design_intent(
        graph, target_room,
        elasticity=elasticity if cfg.use_elasticity else None)

    # How clean is the reference design in its *own* room?  3D-FRONT rooms
    # routinely have a nightstand clipping the bed and a wardrobe a few
    # centimetres into a wall.  Holding the retargeted scene to a stricter
    # standard than its own reference makes the repair loop delete
    # room-defining furniture to fix defects it did not introduce -- measured:
    # an identity target lost both nightstands to a 0.12 m^2 pre-existing
    # overlap.  So the reference's own violation is the floor of the tolerance.
    ref_bound = ref_col = 0.0
    if cfg.tol_at_least_reference:
        # note: this second call must not re-derive motif rigidity, or it would
        # clobber the target intent's values that `initial_layout` depends on
        ref_e = exact_energy(src, build_design_intent(
            graph, src.room, elasticity=None,
            motif_rigidity_from_alpha=False), cfg.weights)
        ref_bound, ref_col = ref_e["E_bound"], ref_e["E_col"]

    # ---- 1. capacity planning ----
    sm = plan_summarization(intent, target_room, allow_drop=cfg.allow_removal)
    keep = sm.keep.copy()
    pop = plan_population(intent, target_room, keep, cooc, bank, rng) \
        if cfg.allow_addition else None

    out = Scene(scene_id=f"{src.scene_id}__retarget", room=target_room.copy(),
                objects=[o.copy() for o in src.objects], source="reroom",
                meta={"source_scene": src.scene_id})
    for i, o in enumerate(out.objects):
        o.keep = bool(keep[i])
    if pop is not None and pop.additions:
        for o in pop.additions:
            out.objects.append(o.copy())
        # Added objects carry no reference relations, so nothing would hold
        # them against a wall during refinement and they drift into the middle
        # of the room, splitting the free space.  Give the wall-loving ones an
        # explicit wall target at the pose the population planner chose.
        _add_wall_targets(intent, out, len(src.objects), target_room)
    n_all = len(out.objects)
    keep_all = np.array([o.keep for o in out.objects], dtype=bool)

    # ---- 2. asset substitution ----
    if cfg.allow_substitution and bank is not None and len(bank):
        substitute_assets(out, intent, bank, rng=rng, **cfg.retrieval)

    # ---- 3. initialisation ----
    xy0, yaw0 = initial_layout(intent, target_room, keep_all[:len(src.objects)],
                               cfg.restarts, rng, cfg.use_motif_init)
    if n_all > len(src.objects):
        poly = as_polygon(target_room)
        free = erode(poly, 0.3)
        extra = n_all - len(src.objects)
        added = out.objects[len(src.objects):]
        add_xy = np.zeros((cfg.restarts, extra, 2))
        add_yaw = np.zeros((cfg.restarts, extra))
        # restart 0 keeps the wall-seeded poses the population planner chose;
        # the rest jitter around them, with a few fully random for diversity
        base_xy = np.stack([o.xy for o in added])
        base_yaw = np.array([o.yaw for o in added])
        for r in range(cfg.restarts):
            if r == 0:
                add_xy[r], add_yaw[r] = base_xy, base_yaw
            elif r < cfg.restarts * 0.7:
                add_xy[r] = base_xy + rng.normal(0, 0.25, size=(extra, 2))
                add_yaw[r] = base_yaw + rng.normal(0, 0.25, size=extra)
            else:
                add_xy[r] = sample_interior(
                    free if not free.is_empty else poly, extra, rng)
                add_yaw[r] = rng.uniform(0, 2 * math.pi, size=extra)
        xy0 = np.concatenate([xy0, add_xy], axis=1)
        yaw0 = np.concatenate([yaw0, add_yaw], axis=1)

    # ---- 4. refine / repair loop ----
    base_size = np.array([o.size.copy() for o in out.objects])
    history = []
    best_scene, best_e = None, None
    for it in range(cfg.outer_iters):
        problem = TorchProblem(out, intent, cfg.weights, device=cfg.device)
        xy, yaw, e, ls = refine_continuous(problem, xy0, yaw0, cfg.grad_steps,
                                           cfg.lr, allow_resize=cfg.allow_resize)
        xy, yaw, e, ls = refine_continuous(
            problem, xy, yaw, cfg.proj_steps, cfg.lr * 0.35,
            cfg.projection_scale, allow_resize=cfg.allow_resize, s0=ls)
        # the surrogate ranks candidates; the *exact* energy picks the winner,
        # and the intent-aware initialisation competes on equal terms so the
        # solver can never return something worse than where it started
        n_seed = min(2, xy0.shape[0])
        cand_xy = np.concatenate([xy, xy0[:n_seed]], axis=0)
        cand_yaw = np.concatenate([yaw, yaw0[:n_seed]], axis=0)
        # the two hand-built seeds are scored at the asset's retrieved size
        zero = np.zeros((n_seed, xy.shape[1]), dtype=np.float32)
        cand_s = (np.concatenate([ls, zero], axis=0) if ls is not None
                  else np.zeros((len(cand_xy), xy.shape[1]), dtype=np.float32))
        rank = np.argsort(np.concatenate([e, [np.inf] * n_seed]))[:cfg.exact_topk]
        # always score the two hand-built starts exactly, whatever the
        # surrogate thinks of them
        rank = np.concatenate([rank,
                               np.arange(len(cand_xy) - n_seed, len(cand_xy))])
        best, best_ex = None, None
        for c in rank:
            _write_back(out, cand_xy[c], cand_yaw[c], cand_s[c], base_size)
            _apply_supports(out, intent)
            cex = exact_energy(out, intent, cfg.weights)
            if best_ex is None or cex["E"] < best_ex["E"]:
                best, best_ex = int(c), cex
        _write_back(out, cand_xy[best], cand_yaw[best], cand_s[best], base_size)
        _apply_supports(out, intent)
        ex = best_ex
        xy[0], yaw[0] = cand_xy[best], cand_yaw[best]
        best = 0
        history.append({"iter": it, "surrogate": float(e[best]), **{
            k: v for k, v in ex.items() if k.startswith("E")}})
        if best_e is None or ex["E"] < best_e["E"]:
            best_e = ex
            best_scene = out.copy()
        total_fp = max(sum(o.footprint_area for o in out.objects if o.keep), 1e-6)
        # a hair of slack on the reference floor: the identity target
        # reproduces the reference's violation to the last bit, and an exact
        # `<=` comparison on two floats that differ in the 14th decimal was
        # enough to trigger a deletion
        b_tol = max(cfg.bound_tol * total_fp, cfg.tol_floor, ref_bound * 1.02 + 1e-9)
        c_tol = max(cfg.col_tol * total_fp, cfg.tol_floor, ref_col * 1.02 + 1e-9)
        feasible = (ex["E_bound"] <= b_tol and ex["E_col"] <= c_tol)
        if cfg.verbose:
            print(f"  [retarget] iter {it}: E={ex['E']:.3f} bound={ex['E_bound']:.3f} "
                  f"col={ex['E_col']:.3f} rel={ex['E_rel']:.3f} kept={ex['n_kept']}")
        if feasible or it == cfg.outer_iters - 1:
            break
        # repair: remove the single worst offender, keeping importance in mind
        victim = _worst_offender(out, intent, cfg.protect_anchor)
        if victim is None or not cfg.allow_removal:
            break
        out.objects[victim].keep = False
        history.append({"iter": it, "repair": out.objects[victim].category})
        xy0 = np.repeat(xy[best][None], cfg.restarts, axis=0)
        yaw0 = np.repeat(yaw[best][None], cfg.restarts, axis=0)
        # the re-seed after a repair jitters every restart, and it has to skip
        # the pinned rows for the same reason the initial layout does -- the
        # gradient freeze keeps a bad start *fixed*, it does not undo it
        loose = np.array([0.0 if o.locked else 1.0
                          for o in out.objects[:xy0.shape[1]]])
        if len(loose) < xy0.shape[1]:            # objects added by population
            loose = np.concatenate([loose, np.ones(xy0.shape[1] - len(loose))])
        xy0[1:] += rng.normal(0, 0.15, size=xy0[1:].shape) * loose[None, :, None]
        yaw0[1:] += rng.normal(0, 0.15, size=yaw0[1:].shape) * loose[None, :]

    scene = best_scene if best_scene is not None else out
    if cfg.allow_addition and cfg.vet_additions:
        _vet_additions(scene, intent, cfg)
    scene.objects = [o for o in scene.objects if o.keep]
    ex = exact_energy(scene, intent, cfg.weights)
    return RetargetResult(
        scene=scene, intent=intent, energy=ex,
        info={"summarization": sm.log, "dropped_motifs": sm.dropped_motifs,
              "reference_bound": ref_bound, "reference_col": ref_col,
              "population": (pop.log if pop else []),
              "history": history, "seconds": time.time() - t0,
              "n_source": len(src.objects), "n_target": len(scene.objects),
              "area_ratio": intent.area_ratio,
              "density_source": src.density(), "density_target": scene.density()})


def _add_wall_targets(intent: DesignIntent, scene: Scene, offset: int,
                      target_room: Room, min_affinity: float = 0.5) -> None:
    from .target import WallTarget
    walls = target_room.walls()
    if not walls:
        return
    for k in range(offset, len(scene.objects)):
        o = scene.objects[k]
        p = prior(o.category)
        if p.wall < min_affinity:
            continue
        back = o.xy - o.forward * o.half[1]
        best, best_d, best_t = None, 1e18, 0.5
        for wi, (a, b) in enumerate(walls):
            d = b - a
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            t = d / L
            n = np.array([-t[1], t[0]])
            par = float(np.clip(np.dot(back - a, t) / L, 0.0, 1.0))
            dist = abs(float(np.dot(back - a, n)))
            if dist < best_d:
                best, best_d, best_t = wi, dist, par
        if best is None:
            continue
        intent.walls.append(WallTarget(i=k, wall=best, t=best_t, gap=0.04,
                                       strength=float(p.wall) * 0.8, oid=o.oid))


def _worst_offender(scene: Scene, intent: DesignIntent,
                    protect_anchor: float = 0.9,
                    min_relative_violation: float = 0.08) -> int | None:
    """The kept object contributing most infeasibility per unit of importance.

    Removal is a last resort, so two guards apply.  An object is only a
    candidate if its own violation is a meaningful fraction of *its own*
    footprint -- clipping a wall by a centimetre is not a reason to delete
    anything.  And a room-defining anchor (bed, sofa, dining table) is only
    considered when nothing else offends, and even then only if it is badly
    misplaced; otherwise the solver 'fixes' a 0.04 m^2 overhang by deleting the
    one object the reference room is about.
    """
    poly = as_polygon(scene.room)
    objs = scene.objects
    zeta = intent.zeta
    polys = {i: object_polygon(o) for i, o in enumerate(objs) if o.keep}
    ranked: list[tuple[float, int, bool]] = []
    for i, o in enumerate(objs):
        if not o.keep or o.locked:
            continue
        v = float(polys[i].difference(poly).area)
        for j, p in polys.items():
            if j == i:
                continue
            b = objs[j]
            if o.z >= b.top - 1e-3 or b.z >= o.top - 1e-3:
                continue
            v += float(polys[i].intersection(p).area) * 0.5
        if v / max(o.footprint_area, 1e-6) < min_relative_violation:
            continue
        z = float(zeta[i]) if i < len(zeta) else 0.25
        ranked.append((v / max(z, 0.05), i,
                       prior(o.category).anchor >= protect_anchor))
    if not ranked:
        return None
    ranked.sort(key=lambda t: -t[0])
    for _, i, anchored in ranked:
        if not anchored:
            return i
    _, i, _ = ranked[0]
    o = objs[i]
    return i if float(polys[i].difference(poly).area) > 0.3 * o.footprint_area \
        else None
