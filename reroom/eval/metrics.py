"""Evaluation metrics (plan section 15).

Geometric / functional legality
    R_OOB   = sum_i Area(B_i \\ P_t) / sum_i Area(B_i)                  (40)
    R_col   = sum_{i<j} Area(B_i ∩ B_j) / sum_i Area(B_i)               (41)
    plus minimum-clearance violation, door/window blockage, free-space
    connectivity and reachable-area ratio.

Reference design preservation
    S_rel   = 1 - (1/|E|) sum_{(i,j) in E} D(e^r_ij, e^t_ij)            (42)
    S_motif = sum_k w_k [m_k preserved] / sum_k w_k                     (43)
    plus an appearance similarity that is reported as an *auxiliary* number
    only, because a global CLIP-style score is dominated by colour and by the
    largest object and cannot substitute for relation/motif evaluation.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from shapely.ops import unary_union

from ..core.categories import prior
from ..core.scene import Scene
from ..geom.freespace import (build_freespace, clearance_violation,
                              door_blockage, door_seeds)
from ..geom.polygon import as_polygon, object_polygon
from ..intent.motifs import Motif
from ..intent.relations import (SceneGraph, relation_distance,
                                relation_features)

__all__ = ["geometry_metrics", "preservation_metrics", "evaluate",
           "aggregate", "MOTIF_REL_TOL"]

MOTIF_REL_TOL = 0.35        # mean internal relation error above which a motif
                            # counts as broken


def _oid_map(scene: Scene) -> dict:
    return {o.oid: o for o in scene.objects if o.keep}


# --------------------------------------------------------------------------
def geometry_metrics(scene: Scene, res: float = 0.05) -> dict:
    objs = [o for o in scene.objects if o.keep]
    room_poly = as_polygon(scene.room)
    total = float(sum(o.footprint_area for o in objs))
    if total <= 1e-9:
        return {"R_OOB": 0.0, "R_col": 0.0, "clearance_violation_ratio": 0.0,
                "clearance_worst_object": 0.0, "door_blockage": 0.0,
                "window_blockage": 0.0, "free_components": 0,
                "largest_free_ratio": 0.0, "reachable_ratio": 0.0,
                "walkable_ratio": 0.0, "density": 0.0, "n_objects": 0}

    polys = [object_polygon(o) for o in objs]
    oob = float(sum(p.difference(room_poly).area for p in polys))
    col = 0.0
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if a.z >= b.top - 1e-3 or b.z >= a.top - 1e-3:
                continue
            col += float(polys[i].intersection(polys[j]).area)

    fs = build_freespace(scene, res=res)
    seeds = door_seeds(scene.room)
    cl = clearance_violation(scene)
    db = door_blockage(scene)
    return {
        "R_OOB": oob / total,
        "R_col": col / total,
        "clearance_violation_ratio": cl["clearance_violation_ratio"],
        "clearance_worst_object": cl["clearance_worst_object"],
        "door_blockage": db["door_blockage"],
        "window_blockage": db["window_blockage"],
        "free_components": fs.n_components(),
        "largest_free_ratio": fs.largest_component_ratio(),
        "reachable_ratio": fs.reachable_ratio(seeds),
        "walkable_ratio": fs.walkable_ratio(),
        "density": scene.density(),
        "n_objects": len(objs),
    }


# --------------------------------------------------------------------------
def preservation_metrics(ref_graph: SceneGraph, target: Scene,
                         intent=None) -> dict:
    """``S_rel`` (42) and ``S_motif`` (43), plus object/category retention.

    Two variants of ``S_rel`` are reported deliberately:

    ``S_rel``       over *all* reference relations, a deleted endpoint scoring
                    the worst possible distance -- this is eq. (42) as written
                    and it charges summarization for what it removes;
    ``S_rel_kept``  over surviving pairs only -- this isolates how well the
                    *geometry* of what survived was transferred.
    """
    ref = ref_graph.scene
    tmap = _oid_map(target)
    tgt_poly = as_polygon(target.room)
    scale = math.sqrt(max(tgt_poly.area, 1e-6))
    ref_scale = math.sqrt(max(as_polygon(ref.room).area, 1e-6))

    room_scale = scale / max(ref_scale, 1e-6)
    tot_w = 0.0
    err_all = 0.0
    err_kept = 0.0
    err_scaled = 0.0
    kept_w = 0.0
    per_kind: dict[str, list] = {}
    for r in ref_graph.relations:
        a = ref.objects[r.i]
        b = ref.objects[r.j]
        w = r.weight
        tot_w += w
        ta, tb = tmap.get(a.oid), tmap.get(b.oid)
        if ta is None or tb is None:
            err_all += w * 1.0
            err_scaled += w * 1.0
            continue
        f_t = relation_features(ta, tb)
        d = relation_distance(r.phi, f_t)
        # same comparison, but judging distances in units of the target room:
        # this is the yardstick a pure uniform rescale would maximise
        f_scaled = r.phi.copy()
        f_scaled[4] *= room_scale
        f_scaled[5] *= room_scale
        err_scaled += w * relation_distance(f_scaled, f_t)
        err_all += w * d
        err_kept += w * d
        kept_w += w
        per_kind.setdefault(r.kind, []).append(d)

    s_rel = 1.0 - err_all / tot_w if tot_w > 1e-9 else 1.0
    s_rel_kept = 1.0 - err_kept / kept_w if kept_w > 1e-9 else 1.0
    s_rel_scaled = 1.0 - err_scaled / tot_w if tot_w > 1e-9 else 1.0
    s_rel_elastic = _elastic_relation_score(ref_graph, target, tmap)

    # ---- motifs ----
    num = den = 0.0
    preserved = []
    for m in ref_graph.motifs:
        den += m.weight
        head = ref.objects[m.head]
        if head.oid not in tmap:
            continue
        alive = [i for i in m.members if ref.objects[i].oid in tmap]
        frac = len(alive) / max(len(m.members), 1)
        need = _min_member_fraction(m, ref)
        if frac < need:
            continue
        errs = []
        for r in ref_graph.relations:
            if r.i not in m.members or r.j not in m.members:
                continue
            ta = tmap.get(ref.objects[r.i].oid)
            tb = tmap.get(ref.objects[r.j].oid)
            if ta is None or tb is None:
                continue
            errs.append(relation_distance(r.phi, relation_features(ta, tb)))
        if errs and float(np.mean(errs)) > MOTIF_REL_TOL:
            continue
        num += m.weight
        preserved.append(m.mid)
    s_motif = num / den if den > 1e-9 else 1.0

    ref_cats = Counter(o.category for o in ref.objects)
    tgt_cats = Counter(o.category for o in target.objects if o.keep)
    inter = sum((ref_cats & tgt_cats).values())
    union = sum((ref_cats | tgt_cats).values())
    return {
        "S_rel": float(s_rel),
        "S_rel_kept": float(s_rel_kept),
        "S_rel_scaled": float(s_rel_scaled),
        "S_rel_elastic": float(s_rel_elastic),
        "S_motif": float(s_motif),
        "motifs_preserved": len(preserved),
        "motifs_total": len(ref_graph.motifs),
        "object_retention": (sum(1 for o in ref.objects if o.oid in tmap)
                             / max(len(ref.objects), 1)),
        "category_jaccard": inter / max(union, 1),
        "n_added": sum(1 for o in target.objects if o.keep and o.meta.get("added")),
        "n_substituted": sum(1 for o in target.objects
                             if o.keep and o.meta.get("substituted_from")),
        "per_kind_error": {k: float(np.mean(v)) for k, v in per_kind.items()},
    }


def _elastic_relation_score(ref_graph: SceneGraph, target: Scene,
                            tmap: dict) -> float:
    """``S_rel`` judged against the *elasticity-adjusted* reference relation.

    This is the plan's own hypothesis made measurable: a relation is preserved
    if it matches ``phi~^{r->t}``, not the raw reference.  The prior (never the
    learned) elasticity model is used, so every method -- including ReRoom with
    a trained ``f_psi`` -- is scored by the same yardstick.
    """
    from ..intent.elasticity import PriorElasticity
    from ..retarget.target import build_design_intent
    intent = build_design_intent(ref_graph, target.room,
                                 elasticity=PriorElasticity())
    tot = err = 0.0
    for r in intent.relations:
        a = ref_graph.scene.objects[r.i]
        b = ref_graph.scene.objects[r.j]
        tot += r.weight
        ta, tb = tmap.get(a.oid), tmap.get(b.oid)
        if ta is None or tb is None:
            err += r.weight
            continue
        err += r.weight * relation_distance(r.phi_des, relation_features(ta, tb))
    return 1.0 - err / tot if tot > 1e-9 else 1.0


def _min_member_fraction(m: Motif, ref: Scene) -> float:
    """A motif survives only if enough of its members do."""
    if m.schema is None:
        return 0.5
    req = 1
    for cat, k in (m.schema.min_members or {}).items():
        req += k
    return min(0.9, req / max(len(m.members), 1))


# --------------------------------------------------------------------------
def evaluate(ref_graph: SceneGraph, target: Scene, intent=None,
             res: float = 0.05, bank=None, encoder=None,
             global_appearance: bool = False) -> dict:
    """All of section 15 for one retargeting.

    ``bank`` and ``encoder`` enable the *auxiliary* appearance score of
    section 15.2.  It is off by default and never enters ``score``: a
    CLIP-style similarity is dominated by colour and by the largest object, and
    the plan is explicit that it cannot stand in for relation and motif
    evaluation.
    """
    out = {}
    out.update(geometry_metrics(target, res=res))
    out.update(preservation_metrics(ref_graph, target, intent))
    if bank is not None:
        from .appearance import object_matched_similarity
        out.update(object_matched_similarity(ref_graph.scene, target,
                                             bank=bank, encoder=encoder))
    if global_appearance:
        # section 15.2 in its whole-image form: renders both scenes from the
        # canonical views and compares CLIP embeddings.  It costs a render per
        # view, so it is opt-in, and like the object-matched score it stays out
        # of `score`.
        from .appearance import appearance_similarity
        out.update(appearance_similarity(ref_graph.scene, target,
                                         encoder=encoder))
    # a single scalar for sorting runs: legality x preservation
    legal = (1.0 - min(out["R_OOB"], 1.0)) * (1.0 - min(out["R_col"], 1.0)) \
        * (1.0 - min(out["clearance_violation_ratio"], 1.0))
    out["legality"] = float(legal)
    # preservation is the mean over the three yardsticks, so the headline score
    # cannot be gamed by whichever notion of "distance preserved" a method suits
    pres = (out["S_rel"] + out["S_rel_scaled"] + out["S_rel_elastic"]) / 3.0
    out["S_rel_mean"] = float(pres)
    out["score"] = float(math.sqrt(max(legal, 0.0)
                                   * max(0.5 * (pres + out["S_motif"]), 0.0)))
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean of every numeric field, with the count."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    out["n"] = len(rows)
    return out
