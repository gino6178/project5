"""Turn a reference design intent into *target-room* constraints.

This is where the plan's central claim becomes concrete.  The reference
relation is not copied:

    phi~^{r->t}_ij  =  reference relation, rescaled by relation elasticity   (21)

so a chair keeps its 0.35 m from the dining table no matter how the room
changes, while a sofa and a TV drift apart as the room gets deeper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import prior
from ..core.scene import ObjectInstance, Room, Scene
from ..geom.polygon import (as_polygon, characteristic_scale, floor_descriptor,
                            min_rotated_rect_params)
from ..intent.elasticity import (ElasticityModel, PriorElasticity,
                                 RelationContext, desired_distance)
from ..intent.importance import motif_importance, object_importance
from ..intent.motifs import Motif, motif_of
from ..intent.relations import SceneGraph, relation_features

__all__ = ["DesiredRelation", "WallTarget", "DesignIntent", "build_design_intent"]


@dataclass
class DesiredRelation:
    """One entry of ``E_rel``: what relation (i, j) should look like in P_t."""

    i: int
    j: int
    kind: str
    weight: float                  # base importance of the relation
    phi_ref: np.ndarray            # reference relation features
    phi_des: np.ndarray            # elasticity-adjusted target features
    alpha: float
    gamma: float
    rigid: bool = False            # alpha ~ 0: never let this one stretch
    stiffness: float = 1.0         # how hard to insist on phi_des
    # for `symmetric` relations, the anchor object index in the reference
    # scene; carried through from the source SceneGraph so E_sym can find the
    # axis of symmetry without re-scanning the graph.
    anchor: int | None = None

    @property
    def effective_weight(self) -> float:
        """``w_ij`` scaled by how *confidently* the target is known.

        Eq. (9) says what the target distance is; it says nothing about how
        hard to insist on it.  But the elasticity already answers that: at
        ``alpha ~ 0`` the distance is fixed by the human body and is known
        precisely, so the relation should be stiff; at ``alpha ~ 1`` it is only
        known up to the room's scale, so it should be soft.  Using alpha only
        to move the target -- and not to set the stiffness -- throws away half
        of what it means, which is why ablating it changed nothing.
        """
        return self.weight * self.stiffness


@dataclass
class WallTarget:
    """A reference `against-wall` fact transferred to a target wall."""

    i: int
    wall: int                      # index into the *target* polygon's walls
    t: float                       # preferred parameter along that wall
    gap: float                     # preferred gap in metres
    strength: float                # from the category's wall affinity
    oid: str | None = None         # set for objects added during population,
                                   # whose index is not a reference index


@dataclass
class DesignIntent:
    """``M_r`` plus everything needed to score a candidate target layout."""

    source: Scene
    graph: SceneGraph
    target_room: Room
    relations: list[DesiredRelation]
    walls: list[WallTarget]
    motifs: list[Motif]
    zeta: np.ndarray                       # object importance, eq. (26)
    motif_zeta: dict[str, float]
    scale_hint: np.ndarray                 # (2,) per-axis room scale ratio
    area_ratio: float
    target_density: float                  # rho(S_r), eq. (28)-(29)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.source.objects)

    def rigid_pairs(self) -> list[tuple[int, int]]:
        return [(r.i, r.j) for r in self.relations if r.rigid]


def _wall_dirs(room: Room) -> np.ndarray:
    """Inward normal of every wall."""
    out = []
    for a, b in room.walls():
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-9:
            out.append(np.array([1.0, 0.0]))
            continue
        t = d / L
        out.append(np.array([-t[1], t[0]]))
    return np.asarray(out)


def _match_wall(src_room: Room, tgt_room: Room, src_wall: int,
                src_t: float) -> tuple[int, float]:
    """Map a source wall onto the most similar target wall.

    Matching is by inward-normal direction (so 'the wall behind the bed' stays
    behind the bed even when the room is re-proportioned or cut), with the
    wall's length used to break ties.  The parameter along the wall is kept.
    """
    sn = _wall_dirs(src_room)
    tn = _wall_dirs(tgt_room)
    if len(tn) == 0:
        return 0, src_t
    src_n = sn[src_wall % len(sn)]
    s_lens = [float(np.linalg.norm(b - a)) for a, b in src_room.walls()]
    t_lens = [float(np.linalg.norm(b - a)) for a, b in tgt_room.walls()]
    s_len = s_lens[src_wall % len(s_lens)]
    best, best_score = 0, -1e18
    for k, n in enumerate(tn):
        align = float(np.dot(src_n, n))
        if align <= 0.15:
            continue
        # prefer walls that are long enough to host the same object
        len_pen = -abs(math.log(max(t_lens[k], 1e-3) / max(s_len, 1e-3))) * 0.25
        score = align + len_pen + 0.15 * (t_lens[k] / max(max(t_lens), 1e-6))
        if score > best_score:
            best, best_score = k, score
    if best_score < -1e17:
        best = int(np.argmax([float(np.dot(src_n, n)) for n in tn]))
    return best, float(np.clip(src_t, 0.05, 0.95))


def build_design_intent(graph: SceneGraph, target_room: Room,
                        elasticity: ElasticityModel | None = None,
                        rigid_alpha: float = 0.12,
                        stiffness_kappa: float = 2.5,
                        motif_rigidity_from_alpha: bool = True) -> DesignIntent:
    """Build ``phi~^{r->t}`` for every reference relation."""
    elasticity = elasticity or PriorElasticity()
    scene = graph.scene
    objs = scene.objects
    src_poly = as_polygon(scene.room)
    tgt_poly = as_polygon(target_room)
    g_src = floor_descriptor(src_poly)
    g_tgt = floor_descriptor(tgt_poly)

    src_ext = scene.room.extent
    tgt_ext = target_room.extent
    scale_hint = np.where(src_ext > 1e-6, tgt_ext / np.maximum(src_ext, 1e-6), 1.0)
    area_ratio = float(tgt_poly.area / max(src_poly.area, 1e-9))

    zeta = object_importance(graph)
    mzeta = motif_importance(graph, zeta)

    # ---- relations ----
    ctxs: list[RelationContext] = []
    keys: list = []
    for r in graph.relations:
        a, b = objs[r.i], objs[r.j]
        d_world = b.xy - a.xy
        dist = float(np.linalg.norm(d_world))
        if dist < 1e-6:
            d_world = np.array([1.0, 0.0])
            dist = 1e-6
        gs = characteristic_scale(src_poly, d_world)
        gt = characteristic_scale(tgt_poly, d_world)
        gamma = float(np.clip(gt / max(gs, 1e-6), 0.25, 4.0))
        mi, mj = motif_of(graph, r.i), motif_of(graph, r.j)
        same = mi is not None and mj is not None and mi.mid == mj.mid
        ctxs.append(RelationContext(
            cat_i=a.category, cat_j=b.category, kind=r.kind,
            motif_i=mi.name if mi else "none", motif_j=mj.name if mj else "none",
            same_motif=same, rigidity=(mi.rigidity if same and mi else 0.4),
            g_src=g_src, g_tgt=g_tgt, gamma=gamma, d_ref=dist,
            gamma_src_abs=float(gs)))
        keys.append((r, gamma, dist))

    alphas = elasticity.alphas(ctxs) if ctxs else np.zeros(0)

    desired: list[DesiredRelation] = []
    for (r, gamma, dist), alpha in zip(keys, alphas):
        if r.kind == "support":
            alpha = 0.0
        phi_ref = r.phi.copy()
        s = (1.0 - alpha) + alpha * gamma          # eq. (9) as a scale factor
        phi_des = phi_ref.copy()
        phi_des[0] *= s                            # dp_x
        phi_des[1] *= s                            # dp_y
        phi_des[4] = max(phi_ref[4] * s, 0.0)      # gap
        phi_des[5] = desired_distance(float(phi_ref[5]), float(alpha), gamma)
        anchor = r.meta.get("anchor") if isinstance(r.meta, dict) else None
        desired.append(DesiredRelation(
            i=r.i, j=r.j, kind=r.kind, weight=r.weight, phi_ref=phi_ref,
            phi_des=phi_des, alpha=float(alpha), gamma=gamma,
            rigid=bool(alpha <= rigid_alpha),
            stiffness=1.0 + stiffness_kappa * (1.0 - float(alpha)),
            anchor=int(anchor) if anchor is not None else None))

    # Normalise the stiffnesses to mean 1.  Elasticity should *redistribute*
    # weight between rigid and elastic relations, not inflate ``E_rel`` as a
    # whole: the un-normalised version made alpha a real lever (the ablation
    # gap grew ~9x) but cost 0.024 of the overall score, because a heavier
    # relation term simply outvoted the feasibility terms.
    if desired:
        m = float(np.mean([d.stiffness for d in desired]))
        if m > 1e-6:
            for d in desired:
                d.stiffness /= m

    # ---- motif rigidity, from the fitted elasticity ----
    if motif_rigidity_from_alpha and graph.motifs:
        # A motif's rigidity was a hand-set constant per schema, which quietly
        # did elasticity's job for it -- tight groups stayed tight whatever
        # alpha said, so removing alpha changed nothing.  Deriving it from the
        # fitted alphas of the motif's own internal relations puts the two on
        # the same footing and lets the ablation actually bite.
        by_pair = {(d.i, d.j): d.alpha for d in desired}
        for m in graph.motifs:
            mem = set(m.members)
            a_int = [a for (i, j), a in by_pair.items()
                     if i in mem and j in mem]
            if not a_int:
                continue
            m.rigidity = float(np.clip(1.0 - float(np.mean(a_int)), 0.05, 0.98))

    # ---- wall targets ----
    walls: list[WallTarget] = []
    for w in graph.walls:
        o = objs[w.i]
        k, t = _match_wall(scene.room, target_room, w.wall, w.t)
        walls.append(WallTarget(i=w.i, wall=k, t=t,
                                gap=float(np.clip(w.gap, 0.0, 0.25)),
                                strength=float(prior(o.category).wall)))

    return DesignIntent(
        source=scene, graph=graph, target_room=target_room,
        relations=desired, walls=walls, motifs=graph.motifs, zeta=zeta,
        motif_zeta=mzeta, scale_hint=scale_hint.astype(float),
        area_ratio=area_ratio, target_density=scene.density(),
        meta={"g_src": g_src, "g_tgt": g_tgt,
              "src_mrr": min_rotated_rect_params(src_poly),
              "tgt_mrr": min_rotated_rect_params(tgt_poly)},
    )
