"""Scene summarization: what to delete when the target room is smaller.

Plan section 9.  Selection happens at *motif* level

    k_{m_k} in {0, 1}                                                  (27)

and inside a kept motif it is *structured* pruning (four dining chairs -> two)
rather than arbitrary object dropout, so the result stays semantically legal.
Every candidate edit is scored by area freed per unit of importance lost, which
makes the greedy choice interpretable and reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import prior
from ..core.scene import Scene
from ..geom.polygon import as_polygon, erode, largest_inscribed_circle
from ..intent.importance import removal_order
from ..intent.motifs import Motif, structured_prune
from .target import DesignIntent


# Anchor categories per room type — the semantic core that MUST survive
# summarization no matter how tight the target room is.  A "living room" is
# not a living room without a sofa; a bedroom is not a bedroom without a bed.
# Dropping the anchor drove the shrunk-room qualitative failure ("keeps the
# dining set, throws out the sofa").  Substitution can shrink the anchor to a
# smaller variant, but the category itself stays.
ANCHOR_CATEGORIES: dict[str, tuple] = {
    "living_room": ("sofa", "sofa_bed"),
    "bedroom":     ("bed", "double_bed", "single_bed", "kids_bed", "baby_bed"),
    "dining_room": ("dining_table",),
    "office":      ("desk", "office_desk"),
    "kitchen":     ("kitchen_cabinet",),
}


def _anchor_indices(scene: Scene) -> set:
    """Objects whose category is the room's semantic anchor; never dropped."""
    room_type = getattr(scene.room, "room_type", "") or ""
    anchors = ANCHOR_CATEGORIES.get(room_type, ())
    if not anchors:
        return set()
    return {i for i, o in enumerate(scene.objects) if o.category in anchors}

__all__ = ["SummarizationPlan", "plan_summarization", "area_budget",
           "fits_in_room"]

MAX_DENSITY = 0.80          # beyond this a room is unusable whatever the source


@dataclass
class SummarizationPlan:
    keep: np.ndarray                     # (N,) bool, the k_i of eq. (17)
    dropped_motifs: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    budget: float = 0.0
    demanded: float = 0.0


def area_budget(intent: DesignIntent, target_room, slack: float = 0.0) -> float:
    """Floor area the layout may occupy, from eq. (28)-(29).

    ``slack = 0`` is the density-matching budget of eq. (29) and is what the
    *population* step aims at when the room grows.  Summarization uses a
    positive slack: a smaller room is allowed to be denser than the reference
    before anything is deleted, so objects are removed because they no longer
    *fit*, not merely because the room shrank.
    """
    rho = float(np.clip(intent.target_density + slack, 0.05, MAX_DENSITY))
    return rho * as_polygon(target_room).area


def fits_in_room(size_xy, target_room, margin: float = 0.02) -> bool:
    """Can an object of this footprint be placed inside the polygon at all?"""
    poly = as_polygon(target_room)
    r = 0.5 * float(np.linalg.norm(np.asarray(size_xy)[:2])) * 0.5 + margin
    return not erode(poly, r).is_empty


def _too_big(o, target_room) -> bool:
    poly = as_polygon(target_room)
    _, inr = largest_inscribed_circle(poly, tol=0.04)
    short = float(min(o.size[0], o.size[1]))
    return short * 0.5 > inr + 0.05 or not fits_in_room(o.size, target_room)


def plan_summarization(intent: DesignIntent, target_room,
                       protect_top_motifs: int = 1,
                       allow_drop: bool = True,
                       shrink_slack: float = 0.14) -> SummarizationPlan:
    """Decide ``k_i`` (and ``k_{m_k}``) for the target room."""
    scene = intent.source
    objs = scene.objects
    n = len(objs)
    keep = np.ones(n, dtype=bool)
    budget = area_budget(intent, target_room, slack=shrink_slack)
    demanded = float(sum(o.footprint_area for o in objs))
    plan = SummarizationPlan(keep=keep, budget=budget, demanded=demanded)

    # objects that simply cannot fit are removed regardless of the budget --
    # EXCEPT semantic anchors (sofa in living_room, bed in bedroom, etc.).
    # Anchors that don't fit stay in the keep-set here; the substitution stage
    # (§11) is then responsible for swapping them for a smaller same-category
    # asset from the bank so they fit.
    locked = {i for i, o in enumerate(objs) if o.locked}
    anchors = _anchor_indices(scene)
    for i, o in enumerate(objs):
        if i in locked or i in anchors:
            continue
        if _too_big(o, target_room):
            keep[i] = False
            plan.log.append(f"drop {o.category}[{i}]: does not fit target room")

    if not allow_drop:
        return plan

    mz = dict(intent.motif_zeta)
    order = sorted(mz, key=lambda k: -mz[k])
    protected = set(order[:max(protect_top_motifs, 0)])
    motifs = {m.mid: m for m in intent.motifs}
    zeta = intent.zeta
    # anchor-containing motifs cannot be dropped whole (the anchor object
    # must stay).  Prune inside them is still allowed to save area.
    for mid, m in motifs.items():
        if any(i in anchors for i in m.members):
            protected.add(mid)
    # protect anchor objects from every deletion path (motif-drop and the
    # importance-order fallback both consult `locked`).
    locked = locked | anchors

    def cur_area() -> float:
        return float(sum(objs[i].footprint_area for i in range(n) if keep[i]))

    guard = 0
    while cur_area() > budget and guard < 4 * n + 16:
        guard += 1
        best = None            # (efficiency, kind, payload)
        for mid, m in motifs.items():
            alive = [i for i in m.members if keep[i]]
            if not alive:
                continue
            # (a) drop the whole motif
            if mid not in protected and not (set(alive) & locked):
                a = float(sum(objs[i].footprint_area for i in alive))
                cost = mz.get(mid, 0.3) * (0.6 + 0.4 * len(alive) / max(len(m.members), 1))
                if a > 1e-6:
                    eff = a / max(cost, 1e-3)
                    if best is None or eff > best[0]:
                        best = (eff, "motif", (mid, alive))
            # (b) structured prune inside the motif
            victims = structured_prune(m, scene, drop_budget=2,
                                       symmetric_pairs=_sym_pairs(intent))
            victims = [i for i in victims if keep[i] and i not in locked]
            if victims:
                a = float(sum(objs[i].footprint_area for i in victims))
                cost = float(sum(zeta[i] for i in victims)) * 0.55
                if a > 1e-6:
                    eff = a / max(cost, 1e-3)
                    if best is None or eff > best[0]:
                        best = (eff, "prune", (mid, victims))
        # (c) fall back to plain importance order
        if best is None:
            for i in removal_order(intent.graph, zeta):
                if keep[i] and i not in locked:
                    keep[i] = False
                    plan.log.append(f"drop {objs[i].category}[{i}]: importance order")
                    break
            else:
                break
            continue
        _, kind, payload = best
        mid, idxs = payload
        for i in idxs:
            keep[i] = False
        if kind == "motif":
            plan.dropped_motifs.append(mid)
            plan.log.append(
                f"drop motif {mid} ({[objs[i].category for i in idxs]}) "
                f"zeta={mz.get(mid, 0):.2f}")
        else:
            plan.log.append(
                f"prune {mid}: {[objs[i].category for i in idxs]}")

    plan.keep = keep
    return plan


def _sym_pairs(intent: DesignIntent) -> list[tuple[int, int]]:
    return [(r.i, r.j) for r in intent.relations if r.kind == "symmetric"]
