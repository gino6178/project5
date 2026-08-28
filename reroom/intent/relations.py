"""Design-intent graph: the relation vocabulary of plan section 7.

Every edge carries

    e_ij = [rho_ij, dp_ij, dtheta_ij, d_ij, w_ij]                     (12)

with ``rho`` a relation label, ``dp`` the offset of j expressed in i's own
frame (so it is rotation invariant), ``dtheta`` the relative yaw, ``d`` the
edge-to-edge gap, and ``w`` a confidence/weight used by ``E_rel`` (21) and by
the relation-preservation score (42).

The same module provides ``relation_features`` -- the ``phi(y_i, y_j, P_t)``
of eq. (21) -- so the optimizer and the metric agree on what a relation *is*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import Polygon

from ..core.categories import SEATING_CATEGORIES, prior
from ..core.scene import ObjectInstance, Room, Scene
from ..geom.polygon import (as_polygon, object_polygon,
                            sat_separation, wall_of)

__all__ = [
    "RELATION_TYPES", "Relation", "SceneGraph", "build_scene_graph",
    "relation_features", "relation_distance", "PHI_DIM",
    "pair_gap", "local_offset",
]

RELATION_TYPES = (
    "near",            # generic proximity
    "facing",          # i's front points at j
    "face_to_face",    # both face each other
    "aligned",         # same orientation and collinear
    "left_of",         # j is to i's left
    "right_of",
    "in_front_of",
    "behind",
    "support",         # j rests on i
    "centered_with",   # j sits on i's forward axis
    "symmetric",       # i and j mirror each other about a shared anchor
    "grouped_with",    # same functional motif (filled in by motifs.py)
    "against_wall",    # unary, stored with j == i
    "surrounds",       # j is one of several objects ringing i
)
RELATION_INDEX = {r: k for k, r in enumerate(RELATION_TYPES)}

PHI_DIM = 6            # [dx, dy, cos dtheta, sin dtheta, gap, centre distance]

# how important each relation is to the perceived design (w_ij prior)
RELATION_WEIGHT = {
    "support": 3.0, "symmetric": 2.0, "face_to_face": 2.0, "facing": 1.6,
    "centered_with": 1.4, "surrounds": 1.5, "aligned": 1.0, "grouped_with": 1.2,
    "in_front_of": 0.8, "behind": 0.6, "left_of": 0.6, "right_of": 0.6,
    "near": 0.5, "against_wall": 1.8,
}


def local_offset(a: ObjectInstance, b: ObjectInstance) -> np.ndarray:
    """Offset of ``b`` in ``a``'s frame: +x is a's right, +y is a's front."""
    d = b.xy - a.xy
    return np.array([float(np.dot(d, a.right)), float(np.dot(d, a.forward))])


def pair_gap(a: ObjectInstance, b: ObjectInstance) -> float:
    """Edge-to-edge distance between the two footprints (0 if touching)."""
    pa, pb = object_polygon(a), object_polygon(b)
    if pa.intersects(pb):
        return 0.0
    return float(pa.distance(pb))


def relation_features(a: ObjectInstance, b: ObjectInstance) -> np.ndarray:
    """``phi(y_i, y_j, .)`` -- the geometry a relation is judged on."""
    dp = local_offset(a, b)
    dth = _wrap(b.yaw - a.yaw)
    gap = max(sat_separation(a, b), 0.0)
    dist = float(np.linalg.norm(b.xy - a.xy))
    return np.array([dp[0], dp[1], math.cos(dth), math.sin(dth), gap, dist],
                    dtype=float)


def relation_distance(f_ref: np.ndarray, f_tgt: np.ndarray,
                      scale: float = 1.0, w_dir: float = 1.0,
                      w_ang: float = 1.0, w_dist: float = 1.0,
                      w_contact: float = 0.5, max_ratio: float = 3.0) -> float:
    """``D(e^r_ij, e^t_ij)`` of eq. (42), in [0, 1].

    Three of the four parts are invariant to a uniform rescaling of the whole
    scene; the distance part is not, and cannot be: comparing distances at all
    means choosing whether "the same metres" or "the same proportions" counts
    as preserved.  Rather than hide that choice, ``eval.metrics`` reports the
    same score under all three yardsticks -- ``S_rel`` (metres preserved),
    ``S_rel_scaled`` (proportions preserved) and ``S_rel_elastic`` (the plan's
    elasticity-adjusted target) -- so no method is advantaged by the pick.

    ``dir``      direction of the offset in the subject's own frame -- a rigid
                 copy and a uniform rescale both score 0, a rearrangement does not;
    ``ang``      relative orientation;
    ``dist``     ``|log(d_t / d_r)|`` capped at ``log(max_ratio)``;
    ``contact``  whether a touching pair is still touching (a nightstand against
                 the bed), which no continuous distance term captures well.

    ``scale`` is accepted for signature stability and is unused.
    """
    dr, dt = f_ref[:2], f_tgt[:2]
    nr, nt = np.linalg.norm(dr), np.linalg.norm(dt)
    if nr < 1e-6 or nt < 1e-6:
        dir_err = 0.0 if abs(nr - nt) < 0.1 else 1.0
    else:
        cos = float(np.clip(np.dot(dr / nr, dt / nt), -1.0, 1.0))
        dir_err = (1.0 - cos) / 2.0
    ang_err = (1.0 - float(np.clip(f_ref[2] * f_tgt[2] + f_ref[3] * f_tgt[3],
                                   -1.0, 1.0))) / 2.0
    d_r = max(float(f_ref[5]), 1e-3)
    d_t = max(float(f_tgt[5]), 1e-3)
    dist_err = min(abs(math.log(d_t / d_r)) / math.log(max_ratio), 1.0)
    contact_r = float(f_ref[4]) < 0.05
    contact_t = float(f_tgt[4]) < 0.05
    contact_err = 0.0 if contact_r == contact_t else 1.0
    tot = w_dir + w_ang + w_dist + w_contact
    return float((w_dir * dir_err + w_ang * ang_err + w_dist * dist_err
                  + w_contact * contact_err) / tot)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Relation:
    """One edge of ``G_r = (V, E)``."""

    i: int
    j: int
    kind: str
    weight: float
    phi: np.ndarray                       # relation_features(i, j)
    meta: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        return float(self.phi[4])

    @property
    def dist(self) -> float:
        return float(self.phi[5])


@dataclass
class WallRelation:
    """Unary `against-wall` fact: object ``i`` backs onto wall ``w``."""

    i: int
    wall: int
    gap: float
    angle_err: float
    t: float                              # parameter along the wall in [0, 1]


@dataclass
class SceneGraph:
    scene: Scene
    relations: list[Relation]
    walls: list[WallRelation]
    motifs: list = field(default_factory=list)     # filled by motifs.build_motifs

    @property
    def objects(self) -> list[ObjectInstance]:
        return self.scene.objects

    def edges_of(self, i: int) -> list[Relation]:
        return [r for r in self.relations if r.i == i or r.j == i]

    def wall_of_index(self, i: int) -> WallRelation | None:
        for w in self.walls:
            if w.i == i:
                return w
        return None

    def adjacency(self, kinds: Sequence[str] | None = None) -> np.ndarray:
        n = len(self.objects)
        a = np.zeros((n, n), dtype=float)
        for r in self.relations:
            if kinds and r.kind not in kinds:
                continue
            a[r.i, r.j] += r.weight
            a[r.j, r.i] += r.weight
        return a

    def summary(self) -> dict:
        from collections import Counter
        c = Counter(r.kind for r in self.relations)
        return {"n_objects": len(self.objects), "n_relations": len(self.relations),
                "n_walls": len(self.walls), "kinds": dict(c),
                "n_motifs": len(self.motifs)}


# --------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------
def _near_threshold(a: ObjectInstance, b: ObjectInstance) -> float:
    """Proximity scales with object size -- a sofa's 'near' is bigger."""
    sa = float(np.mean(a.size[:2]))
    sb = float(np.mean(b.size[:2]))
    return max(0.55, 0.65 * max(sa, sb))


def _is_support(a: ObjectInstance, b: ObjectInstance) -> bool:
    """Is ``b`` resting on ``a``?"""
    if b.z < a.top - 0.12 or b.z > a.top + 0.12:
        return False
    if float(np.linalg.norm(b.xy - a.xy)) > float(a.half.max() + b.half.max()):
        return False
    inter = object_polygon(a).intersection(object_polygon(b)).area
    return inter > 0.35 * min(a.footprint_area, b.footprint_area) and inter > 1e-4


# Objects whose local +y is a meaningful "front".  A rug or a pendant lamp has
# no front, so `facing`/`centered_with` are not defined for them as the subject.
HAS_FRONT = frozenset({
    "double_bed", "single_bed", "kids_bed", "bunk_bed", "sofa", "l_sofa",
    "loveseat", "armchair", "lounge_chair", "dining_chair", "office_chair",
    "bench", "desk", "dressing_table", "tv", "tv_stand", "wardrobe", "cabinet",
    "bookcase", "shelf", "sideboard", "drawer_chest", "wine_cabinet",
    "shoe_cabinet", "console_table", "fireplace", "piano", "mirror",
    "nightstand", "barstool", "stool",
})


def _facing(a: ObjectInstance, b: ObjectInstance, tol_deg: float = 32.0,
            max_dist: float = 5.5) -> bool:
    if a.category not in HAS_FRONT:
        return False
    d = b.xy - a.xy
    n = float(np.linalg.norm(d))
    if n < 1e-6 or n > max_dist:
        return False
    cos = float(np.dot(d / n, a.forward))
    if cos < math.cos(math.radians(tol_deg)):
        return False
    # the target must subtend the view, not merely lie in the half plane
    lateral = abs(float(np.dot(d, a.right)))
    return lateral <= max(b.half.max(), 0.45) + 0.12 * n


def _aligned(a: ObjectInstance, b: ObjectInstance, ang_tol: float = 0.20,
             lat_tol: float = 0.30) -> bool:
    if abs(_wrap(b.yaw - a.yaw)) > ang_tol:
        return False
    off = local_offset(a, b)
    return abs(off[1]) <= lat_tol + 0.10 * abs(off[0])


def _centered(a: ObjectInstance, b: ObjectInstance, tol: float = 0.28,
              max_dist: float = 3.0) -> bool:
    if a.category not in HAS_FRONT:
        return False
    off = local_offset(a, b)
    return abs(off[0]) <= tol and 0 < off[1] <= max_dist


def _symmetric(anchor: ObjectInstance, b: ObjectInstance, c: ObjectInstance,
               pos_tol: float = 0.30, ang_tol: float = 0.35) -> bool:
    """Do ``b`` and ``c`` mirror each other about ``anchor``'s forward axis?"""
    if b.category != c.category:
        return False
    ob, oc = local_offset(anchor, b), local_offset(anchor, c)
    if ob[0] * oc[0] >= 0:                       # must straddle the axis
        return False
    if abs(abs(ob[0]) - abs(oc[0])) > pos_tol:
        return False
    if abs(ob[1] - oc[1]) > pos_tol:
        return False
    dth = abs(_wrap(b.yaw - anchor.yaw)) - abs(_wrap(c.yaw - anchor.yaw))
    return abs(dth) < ang_tol


def _surrounds(centre: ObjectInstance, ring: Sequence[ObjectInstance]) -> bool:
    if len(ring) < 3:
        return False
    angs = sorted(math.atan2(*(o.xy - centre.xy)[::-1]) for o in ring)
    spread = max((angs[(k + 1) % len(angs)] - angs[k]) % (2 * math.pi)
                 for k in range(len(angs)))
    return spread < math.radians(200)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def build_scene_graph(scene: Scene, max_pairs_dist: float | None = None,
                      max_degree: int = 10) -> SceneGraph:
    objs = scene.objects
    if max_pairs_dist is None:
        # A flat six-metre cutoff silently deletes every relation that spans a
        # large room -- which is the entire long-range half of the design.
        # Scale it with the room instead.
        ext = scene.room.extent
        max_pairs_dist = float(np.clip(1.05 * float(np.hypot(*ext)), 6.0, 16.0))
    n = len(objs)
    rels: list[Relation] = []
    seen: set[tuple[int, int, str]] = set()

    def add(i: int, j: int, kind: str, w: float | None = None, **meta):
        key = (i, j, kind)
        if key in seen:
            return
        seen.add(key)
        rels.append(Relation(i, j, kind,
                             RELATION_WEIGHT.get(kind, 0.5) if w is None else w,
                             relation_features(objs[i], objs[j]), meta))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = objs[i], objs[j]
            dist = float(np.linalg.norm(b.xy - a.xy))
            if dist > max_pairs_dist:
                continue
            if _is_support(a, b):
                add(i, j, "support")
                continue
            if j > i:
                gap = pair_gap(a, b)
                if gap <= _near_threshold(a, b):
                    # a touching pair is a rigid local motif (bed-nightstand);
                    # a merely nearby pair is weak.  Weight reflects that.
                    add(i, j, "near", w=RELATION_WEIGHT["near"] +
                        1.3 * float(np.exp(-gap / 0.45)))
                if _aligned(a, b):
                    add(i, j, "aligned")
            if _facing(a, b):
                add(i, j, "facing")
                if _facing(b, a):
                    add(min(i, j), max(i, j), "face_to_face")
            if _centered(a, b):
                add(i, j, "centered_with")
            off = local_offset(a, b)
            if j > i and float(np.linalg.norm(off)) <= _near_threshold(a, b) * 2.2:
                if abs(off[0]) > abs(off[1]):
                    add(i, j, "right_of" if off[0] > 0 else "left_of")
                else:
                    add(i, j, "in_front_of" if off[1] > 0 else "behind")

    # symmetric triples -> an edge between the mirrored pair
    for k in range(n):
        anchor = objs[k]
        if prior(anchor.category).anchor < 0.45:
            continue
        cand = [t for t in range(n)
                if t != k and float(np.linalg.norm(objs[t].xy - anchor.xy)) < 3.2]
        for ii in range(len(cand)):
            for jj in range(ii + 1, len(cand)):
                b, c = objs[cand[ii]], objs[cand[jj]]
                if _symmetric(anchor, b, c):
                    add(min(cand[ii], cand[jj]), max(cand[ii], cand[jj]),
                        "symmetric", anchor=k)

    # 'surrounds': a table ringed by >=3 seats
    for k in range(n):
        c = objs[k]
        if c.category not in ("dining_table", "coffee_table"):
            continue
        ring = [t for t in range(n)
                if objs[t].category in SEATING_CATEGORIES
                and pair_gap(c, objs[t]) < 1.1]
        if _surrounds(c, [objs[t] for t in ring]):
            for t in ring:
                add(k, t, "surrounds")

    # unary wall relations
    walls: list[WallRelation] = []
    wsegs = scene.room.walls()
    for i, o in enumerate(objs):
        if prior(o.category).on_support and o.z > 0.3:
            continue
        w = wall_of(o, scene.room)
        if w is None:
            continue
        k, gap, ang = w
        a, b = wsegs[k]
        d = b - a
        L2 = float(np.dot(d, d))
        t = float(np.clip(np.dot(o.xy - a, d) / max(L2, 1e-9), 0.0, 1.0))
        walls.append(WallRelation(i, k, gap, ang, t))

    rels = _prune(rels, n, max_degree)
    return SceneGraph(scene=scene, relations=rels, walls=walls)


def _prune(rels: list[Relation], n: int, max_degree: int) -> list[Relation]:
    """Keep the strongest edges per node so ``E_rel`` stays O(n * max_degree).

    Structural relations (support, symmetric, surrounds, face_to_face) are
    never pruned -- they are what the motif layer is built from.
    """
    if max_degree <= 0:
        return rels
    protected = {"support", "symmetric", "surrounds", "face_to_face"}
    keep = [r for r in rels if r.kind in protected or r.meta.get("motif_link")]
    rest = sorted([r for r in rels if r.kind not in protected],
                  key=lambda r: -r.weight)
    deg = np.zeros(n, dtype=int)
    for r in keep:
        deg[r.i] += 1
        deg[r.j] += 1
    for r in rest:
        if deg[r.i] < max_degree and deg[r.j] < max_degree:
            keep.append(r)
            deg[r.i] += 1
            deg[r.j] += 1
    return keep
