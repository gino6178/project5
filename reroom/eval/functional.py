"""Absolute functional plausibility, independent of any reference.

S_rel and S_motif ask how faithfully a *reference* was reproduced.  They cannot
catch a room that is faithful to a bad reference, or judge a generated layout
that has no reference at all.  What a person reads at a glance is different and
absolute: a dining chair belongs beside its table, a television and a wardrobe
belong against a wall, a nightstand belongs next to a bed.  This scores exactly
those expectations, with no comparison to a source scene.

Two families of rule, both drawn from the category priors already in the system
rather than hand-tuned per experiment:

* **companionship** -- a satellite category (chair, nightstand, coffee table)
  must have an anchor of its partner category within a functional distance;
* **wall affinity** -- a category whose ``wall`` prior is high must actually be
  against a wall.

Each object contributes a score in [0, 1]; the room's score is their mean,
weighted by how strongly the rule applies (a bed's wall affinity 0.95 counts
for more than a coffee table's 0.05).  ``functional_score`` returns the room
mean and the per-rule breakdown so a weakness is legible, not just a number.
"""
from __future__ import annotations

import numpy as np

from ..core.categories import prior
from ..core.scene import Scene
from ..geom.polygon import as_polygon, object_polygon

__all__ = ["functional_score", "COMPANION_RULES", "WALL_RULES"]

# (satellite, {anchor categories}, max centre distance in metres)
COMPANION_RULES = [
    ("dining_chair", {"dining_table"}, 1.4),
    ("armchair", {"coffee_table", "sofa", "tv_stand"}, 3.2),
    ("lounge_chair", {"coffee_table", "sofa", "tv_stand"}, 3.4),
    ("nightstand", {"double_bed", "single_bed", "kids_bed"}, 1.3),
    ("coffee_table", {"sofa", "l_sofa", "loveseat", "lounge_chair"}, 2.2),
    ("stool", {"dining_table", "coffee_table", "desk", "dressing_table"}, 1.8),
    ("tv_stand", {"sofa", "l_sofa", "double_bed", "bed", "loveseat"}, 5.5),
    ("wardrobe", {"double_bed", "single_bed", "kids_bed"}, 6.0),
    ("desk", {"office_chair", "dining_chair"}, 1.6),
]

# categories whose wall prior is treated as a functional requirement, and the
# gap below which they count as satisfied
WALL_MIN_PRIOR = 0.6
WALL_FLUSH = 0.30          # metres from footprint to nearest wall


def _soft(dist: float, want: float, slack: float = 0.6) -> float:
    """1 at or under ``want``, decaying to 0 by ``want + slack``."""
    if dist <= want:
        return 1.0
    return float(max(0.0, 1.0 - (dist - want) / max(slack, 1e-6)))


def functional_score(scene: Scene) -> dict:
    kept = [o for o in scene.objects if o.keep]
    if not kept:
        return {"functional": float("nan"), "companion": float("nan"),
                "wall": float("nan"), "n_companion": 0, "n_wall": 0}
    by_cat: dict[str, list] = {}
    for o in kept:
        by_cat.setdefault(o.category, []).append(o)

    comp_terms, comp_w = [], []
    for sat, anchors, dmax in COMPANION_RULES:
        for o in by_cat.get(sat, []):
            cand = [a for c in anchors for a in by_cat.get(c, [])]
            if not cand:
                s = 0.0                      # its partner is not even present
            else:
                d = min(float(np.linalg.norm(o.xy - a.xy)) for a in cand)
                s = _soft(d, dmax)
            comp_terms.append(s)
            comp_w.append(max(prior(sat).anchor, 0.3))

    boundary = as_polygon(scene.room).boundary
    wall_terms, wall_w = [], []
    for o in kept:
        pw = prior(o.category).wall
        if pw < WALL_MIN_PRIOR or o.z >= 1.4:
            continue
        gap = float(boundary.distance(object_polygon(o)))
        wall_terms.append(_soft(gap, WALL_FLUSH, slack=0.5))
        wall_w.append(pw)

    def wmean(v, w):
        if not v:
            return float("nan")
        v, w = np.asarray(v), np.asarray(w)
        return float((v * w).sum() / max(w.sum(), 1e-9))

    comp = wmean(comp_terms, comp_w)
    wall = wmean(wall_terms, wall_w)
    parts = [x for x in (comp, wall) if not np.isnan(x)]
    return {"functional": float(np.mean(parts)) if parts else float("nan"),
            "companion": comp, "wall": wall,
            "n_companion": len(comp_terms), "n_wall": len(wall_terms)}
