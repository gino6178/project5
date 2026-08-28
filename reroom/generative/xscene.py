"""Cross-Scene Pairing (同風格跨場景配對) and Motif-Preserving forward deform.

Two ways to manufacture *real*-target training pairs, replacing the
affine-warp pseudo-reference that stretches motifs:

1. **CrossScenePairs** — find two real rooms of the same type whose object
   *category* sets have Jaccard similarity > tau (default 0.6).  Use one as the
   reference input (S_ref) and the other's polygon as the target boundary, with
   its *real* human layout as ground truth (S_tgt).  Both sides are real,
   human-plausible designs, so the model learns genuine cross-space style
   transfer rather than "undo an affine warp".  Objects are matched across the
   two rooms per-category by footprint size (Hungarian), so the GT position of
   reference object i is the placement of its matched counterpart in S_tgt.

2. **motif_rigid_warp** — the fallback for a reference with no > tau partner:
   keep the reference real, deform the *target* polygon, and transplant the
   layout with each motif moved as one rigid body (head translated, member
   offsets/yaws preserved) so Affine never stretches a sofa+coffee-table or
   bed+nightstand group.  This is the motif-preserving GT generator.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.scene import Scene
from ..geom.polygon import as_polygon
from ..intent.relations import build_scene_graph
from ..intent.motifs import build_motifs

__all__ = ["catset", "jaccard", "build_pair_index", "match_objects",
           "make_cross_pair", "motif_rigid_warp"]


def catset(s: Scene) -> frozenset:
    return frozenset(o.category for o in s.objects if o.keep)


def jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def build_pair_index(scenes: list[Scene], thresh: float = 0.6,
                     max_partners: int = 16, seed: int = 0) -> dict:
    """For each scene index, a list of partner indices (same room_type,
    Jaccard(category-set) > thresh).  Capped at ``max_partners`` random
    partners per scene to keep the epoch diverse and the index small."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    sets = [catset(s) for s in scenes]
    rts = [s.room.room_type for s in scenes]
    by_rt = defaultdict(list)
    for i, rt in enumerate(rts):
        by_rt[rt].append(i)
    index: dict[int, list[int]] = {}
    for rt, idxs in by_rt.items():
        for a in idxs:
            partners = [b for b in idxs
                        if b != a and jaccard(sets[a], sets[b]) > thresh]
            if len(partners) > max_partners:
                partners = list(rng.choice(partners, max_partners, replace=False))
            if partners:
                index[a] = [int(x) for x in partners]
    return index


def match_objects(ref: Scene, tgt: Scene) -> list[tuple[int, int]]:
    """Per-category Hungarian matching of reference objects to target objects
    by footprint-size L2.  Returns (ref_i, tgt_j) pairs; ref objects with no
    same-category counterpart are left unmatched (no GT supervision)."""
    from collections import defaultdict
    ref_by = defaultdict(list); tgt_by = defaultdict(list)
    for i, o in enumerate(ref.objects):
        if o.keep: ref_by[o.category].append(i)
    for j, o in enumerate(tgt.objects):
        if o.keep: tgt_by[o.category].append(j)
    pairs = []
    for cat, ris in ref_by.items():
        tjs = tgt_by.get(cat, [])
        if not tjs: continue
        C = np.zeros((len(ris), len(tjs)))
        for a, ri in enumerate(ris):
            sr = ref.objects[ri].size[:2]
            for b, tj in enumerate(tjs):
                st = tgt.objects[tj].size[:2]
                C[a, b] = float(np.linalg.norm(sr - st))
        ra, cb = linear_sum_assignment(C)
        for a, b in zip(ra, cb):
            pairs.append((ris[a], tjs[b]))
    return pairs


def make_cross_pair(ref: Scene, tgt: Scene):
    """Assemble a training triplet from a cross-scene pair.

    Returns (source_scene, target_room, gt_scene) where:
      * source_scene = ``ref`` (real reference layout + room), the input,
      * target_room  = ``tgt.room`` (real target boundary),
      * gt_scene     = objects carrying *ref* oids at the *matched tgt*
        placements, in ``tgt.room`` — so build_tokens supervises reference
        object i at its counterpart's real position.  Unmatched ref objects are
        absent from gt_scene and hence masked out of the flow loss.
    Returns None if fewer than 2 objects match.
    """
    pairs = match_objects(ref, tgt)
    if len(pairs) < 2:
        return None
    gt_objs = []
    for ri, tj in pairs:
        o = ref.objects[ri].copy()          # keep ref identity / size / category
        t = tgt.objects[tj]
        o.position = np.array([t.xy[0], t.xy[1], o.z], dtype=float)
        o.yaw = float(t.yaw)
        gt_objs.append(o)
    gt_scene = Scene(scene_id=f"{ref.scene_id}__to__{tgt.scene_id}",
                     room=tgt.room.copy(), objects=gt_objs)
    return ref, tgt.room.copy(), gt_scene


def motif_rigid_warp(scene: Scene, new_room, clip_inside: bool = True) -> Scene:
    """Motif-preserving forward deform (Proposal 2).

    Like ``warp_scene`` but each motif is transplanted as one rigid body: the
    head object is mapped by the MRR-frame affine, and every member keeps its
    offset and relative yaw to the head.  Non-motif objects fall back to the
    per-object affine map.  This prevents the affine from stretching intra-motif
    spacing (sofa+coffee_table, bed+nightstand), producing a plausible GT for a
    deformed target room when no real cross-scene partner exists.
    """
    from shapely.geometry import Point as _P
    from ..geom.polygon import object_polygon
    from ..retarget.optimizer import _mrr_frame, _map_point
    out = scene.copy()
    src = _mrr_frame(as_polygon(scene.room))
    tgt = _mrr_frame(as_polygon(new_room))
    dang = tgt[5] - src[5]
    out.room = new_room.copy()
    poly = as_polygon(new_room) if clip_inside else None
    centroid = np.array(poly.centroid.coords[0]) if poly is not None else None

    graph = build_motifs(build_scene_graph(scene))

    def _fit_group_inside(objs, iters: int = 5):
        """Rigidly shift a whole group (one shared translation, so intra-motif
        geometry is preserved) to pull the most-overhanging footprint corner
        back inside the room.  Iterated a few times because pulling one side in
        can expose another.  Objects too big for a concave notch simply stop
        improving -- the overhang guard in make_forward_pair then rejects the
        deform."""
        if poly is None:
            return
        ex = poly.exterior
        for _ in range(iters):
            worst_d = 0.0; worst_shift = None
            for o in objs:
                for cx, cy in o.corners():
                    pt = _P(cx, cy)
                    if poly.contains(pt):
                        continue
                    p_on = ex.interpolate(ex.project(pt))
                    v = np.array([p_on.x - cx, p_on.y - cy])   # corner -> boundary
                    d = float(np.linalg.norm(v))
                    if d > worst_d:
                        worst_d = d; worst_shift = v
            if worst_shift is None or worst_d < 1e-3:
                break
            # shift the whole group inward a hair past the boundary
            nrm = np.linalg.norm(worst_shift)
            step = worst_shift + (worst_shift / nrm) * 0.03 if nrm > 1e-6 else worst_shift
            for o in objs:
                o.xy = o.xy + step

    done = set()
    for gi, m in enumerate(graph.motifs):
        head = m.members[0]
        h_old = out.objects[head].xy.copy()
        h_new = _map_point(h_old, src, tgt)
        # rotate member offsets by the frame rotation delta only (rigid)
        c, s = math.cos(dang), math.sin(dang)
        rot = np.array([[c, -s], [s, c]])
        members = []
        for j in m.members:
            if j >= len(out.objects): continue
            o = out.objects[j]
            off = o.xy - h_old
            o.xy = h_new + rot @ off
            o.yaw = o.yaw + dang
            members.append(o); done.add(j)
        _fit_group_inside(members)               # rigid inward shift for the motif
    for j, o in enumerate(out.objects):
        if j in done: continue
        o.xy = _map_point(o.xy, src, tgt)
        o.yaw = o.yaw + dang
        _fit_group_inside([o])                   # singleton: shift fully inside
    return out


# ---------------------------------------------------------------------------
# Filtering & correction mechanisms (added 2026-08-26)
# ---------------------------------------------------------------------------

# core-anchor categories per room type: the furniture whose orientation defines
# the room's design intent (a sofa's facing, a bed's headboard direction).
CORE_ANCHORS = {
    "living_room": ("sofa", "sofa_bed", "l_shaped_sofa", "chaise"),
    "bedroom":     ("double_bed", "bed", "single_bed", "kids_bed"),
    "dining_room": ("dining_table", "chinese_dining_table"),
    "office":      ("desk", "office_desk"),
}


def _primary_anchor(scene):
    """The single largest-footprint core-anchor object, or None."""
    cats = CORE_ANCHORS.get(scene.room.room_type, ())
    cands = [o for o in scene.objects if o.keep and o.category in cats]
    if not cands:
        return None
    return max(cands, key=lambda o: float(o.size[0] * o.size[1]))


def _norm_yaw(scene, obj):
    """Object yaw relative to the room's MRR frame (canonical orientation)."""
    from .tokens import room_frame
    fr = room_frame(scene.room)
    return float(obj.yaw - fr.angle)


def anchor_orientation_ok(ref: Scene, tgt: Scene, max_deg: float = 30.0) -> bool:
    """Anchor Orientation Filter (Mechanism 1).

    Compares the primary core-anchor's canonical orientation between the two
    rooms.  The MRR long axis is 180-degree ambiguous, so the difference is
    folded into [0, 90] (0 and 180 both count as aligned); a sofa/bed turned
    45 or 90 degrees is a genuine layout change and gets rejected.  Returns True
    if aligned within ``max_deg`` (or if either room lacks an anchor, in which
    case orientation can't discriminate and the pair is kept)."""
    ar = _primary_anchor(ref); at = _primary_anchor(tgt)
    if ar is None or at is None:
        return True
    d = abs(_norm_yaw(ref, ar) - _norm_yaw(tgt, at)) % math.pi   # [0, pi)
    d = min(d, math.pi - d)                                      # fold to [0, pi/2]
    return math.degrees(d) <= max_deg


def match_objects_spatial(ref: Scene, tgt: Scene,
                          w_size: float = 0.2) -> list[tuple[int, int]]:
    """Hungarian matching on a *spatial* cost matrix (Mechanism 2).

    Within each category, the cost between reference object a and target object
    b is the Euclidean distance of their canonical (room-frame-normalised)
    positions, plus ``w_size`` times footprint-size distance as a role
    tiebreaker.  Minimising total normalised displacement gives the shortest,
    least-crossing assignment S_ref -> S_tgt -- better than size-only matching
    when a category has several instances (dining chairs, nightstands)."""
    from collections import defaultdict
    from .tokens import room_frame, to_frame
    fr_r = room_frame(ref.room); fr_t = room_frame(tgt.room)
    ref_by = defaultdict(list); tgt_by = defaultdict(list)
    for i, o in enumerate(ref.objects):
        if o.keep: ref_by[o.category].append(i)
    for j, o in enumerate(tgt.objects):
        if o.keep: tgt_by[o.category].append(j)
    pairs = []
    for cat, ris in ref_by.items():
        tjs = tgt_by.get(cat, [])
        if not tjs: continue
        C = np.zeros((len(ris), len(tjs)))
        for a, ri in enumerate(ris):
            or_ = ref.objects[ri]
            pr = to_frame(or_.xy, or_.yaw, fr_r)[:2]
            for b, tj in enumerate(tjs):
                ot = tgt.objects[tj]
                pt = to_frame(ot.xy, ot.yaw, fr_t)[:2]
                pos = float(np.linalg.norm(pr - pt))
                sz = float(np.linalg.norm(or_.size[:2] - ot.size[:2]))
                C[a, b] = pos + w_size * sz
        ra, cb = linear_sum_assignment(C)
        for a, b in zip(ra, cb):
            pairs.append((ris[a], tjs[b]))
    return pairs


def make_cross_pair_filtered(ref: Scene, tgt: Scene, max_deg: float = 30.0):
    """make_cross_pair but with the orientation filter and spatial Hungarian.
    Returns None if the orientation filter rejects the pair or <2 objects match.
    """
    if not anchor_orientation_ok(ref, tgt, max_deg):
        return None
    pairs = match_objects_spatial(ref, tgt)
    if len(pairs) < 2:
        return None
    gt_objs = []
    for ri, tj in pairs:
        o = ref.objects[ri].copy()
        t = tgt.objects[tj]
        o.position = np.array([t.xy[0], t.xy[1], o.z], dtype=float)
        o.yaw = float(t.yaw)
        gt_objs.append(o)
    troom = tgt.room.copy()
    gt_scene = Scene(scene_id=f"{ref.scene_id}__to__{tgt.scene_id}",
                     room=troom, objects=gt_objs)
    # reject degenerate real pairs whose matched GT lands mostly outside the
    # target polygon (imperfect 3D-FRONT room outlines); caller falls back.
    if _gt_oob_frac(gt_scene, troom) > 0.30:
        return None
    return ref, troom, gt_scene


def _anchor_yaw(scene):
    """Canonical (frame-relative) yaw of the primary anchor, or None."""
    a = _primary_anchor(scene)
    if a is None:
        return None
    return _norm_yaw(scene, a)


def _orient_ok_yaws(ya, yb, max_deg: float = 30.0) -> bool:
    if ya is None or yb is None:
        return True
    d = abs(ya - yb) % math.pi
    d = min(d, math.pi - d)
    return math.degrees(d) <= max_deg


def build_pair_index_filtered(scenes: list[Scene], thresh: float = 0.6,
                              max_deg: float = 30.0, max_partners: int = 16,
                              ratio_range=(0.5, 2.0), seed: int = 0) -> dict:
    """Pair index restricted to partners that pass ALL of: Jaccard>thresh, the
    anchor orientation filter, and a target/source area ratio inside
    ``ratio_range`` (so a bed+wardrobe in a 10 m2 room is not paired with the
    same furniture floating in a 30 m2 room).  Anchor yaws and areas are
    precomputed once so every pair test is O(1)."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    sets = [catset(s) for s in scenes]
    yaws = [_anchor_yaw(s) for s in scenes]
    areas = [as_polygon(s.room).area for s in scenes]
    rts = [s.room.room_type for s in scenes]
    by_rt = defaultdict(list)
    for i, rt in enumerate(rts):
        by_rt[rt].append(i)
    rlo, rhi = ratio_range
    index: dict[int, list[int]] = {}
    for rt, idxs in by_rt.items():
        for a in idxs:
            partners = [b for b in idxs
                        if b != a and jaccard(sets[a], sets[b]) > thresh
                        and _orient_ok_yaws(yaws[a], yaws[b], max_deg)
                        and rlo <= areas[b] / max(areas[a], 1e-6) <= rhi]
            if len(partners) > max_partners:
                partners = list(rng.choice(partners, max_partners, replace=False))
            if partners:
                index[a] = [int(x) for x in partners]
    return index


def _gt_oob_frac(gt: Scene, room) -> float:
    from shapely.geometry import Point as _P
    poly = as_polygon(room); n = 0; o = 0
    for x in gt.objects:
        if not x.keep: continue
        n += 1
        if not poly.contains(_P(*x.xy)): o += 1
    return o / max(n, 1)


def _gt_overhang_frac(gt: Scene, room, tol: float = 0.15) -> float:
    """Fraction of keep-objects whose *footprint* pokes more than ``tol`` of its
    area outside the room polygon.  Catches furniture that clears the centre
    check but still overhangs a slanted / concave / corner-cut wall -- the
    'objects sticking out' the centre-only guard missed."""
    from ..geom.polygon import object_polygon
    poly = as_polygon(room); n = 0; bad = 0
    for x in gt.objects:
        if not x.keep: continue
        n += 1
        fp = object_polygon(x); a = fp.area + 1e-9
        if fp.difference(poly).area / a > tol:
            bad += 1
    return bad / max(n, 1)


def real_collisions(scene: Scene, min_overlap: float = 0.02) -> int:
    """Count genuine object-object collisions, using the same rules the
    optimiser/guidance enforce: a pair collides only if their footprints
    intersect, they overlap in *z* (a lamp on a table or a rug underfoot does
    not), the pair is not in ``NESTABLE_PAIRS`` (chair-under-table etc.), and
    the intersection exceeds ``min_overlap`` fraction of the smaller footprint.
    """
    from ..geom.polygon import boxes_overlap_3d, object_polygon
    from .guidance import NESTABLE_PAIRS
    objs = [o for o in scene.objects if o.keep]
    n = 0
    for i in range(len(objs)):
        pi = object_polygon(objs[i]); ai = pi.area
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if frozenset({a.category, b.category}) in NESTABLE_PAIRS:
                continue
            if not boxes_overlap_3d(a, b):
                continue
            pj = object_polygon(b)
            inter = pi.intersection(pj).area
            if inter <= 1e-4:
                continue
            if inter / (min(ai, pj.area) + 1e-9) > min_overlap:
                n += 1
    return n


def make_forward_pair(scene: Scene, rng, levels=(1, 2, 3, 4, 5),
                      l1_range=(0.5, 2.0), l1_u_shape=True,
                      ratio_range=(0.6, 1.7), max_oob=0.15,
                      max_overhang=0.15, tries=8):
    """Forward-deform pair (Mechanism 3, the 70% majority path).

    Keep ``scene`` as the real reference; deform its room into a target
    boundary; transplant the layout with each motif moved as a rigid body
    (``motif_rigid_warp``) to produce the GT.  Because the reference is real and
    motifs are kept rigid, intent transfer is guaranteed by construction.

    Two data-quality guards (added after the pipeline check found a 4.0x area
    blow-up and a rare all-objects-out-of-bounds collapse):
      * the target/source **area ratio** is constrained to ``ratio_range`` so
        training pairs stay near the deployment range (test is 0.75-1.35x),
        not the unrealistic 2-4x tail a raw corner-cut can produce;
      * the produced GT must have OOB fraction <= ``max_oob``; a degenerate
        deform (where the rigid warp collapses a motif onto the boundary) is
        rejected.
    If no deform passes in ``tries`` attempts, fall back to a bounded
    uniform_scale whose linear factor is sqrt(mid-ratio), guaranteeing a clean
    in-range pair.  Returns (source_scene, target_room, gt_scene).
    """
    from ..geom.deform import deform_room
    src_area = as_polygon(scene.room).area
    src_coll = real_collisions(scene)          # baseline collisions in the ref
    best = None
    for _ in range(tries):
        level = int(rng.choice(levels))
        tgt_room = deform_room(scene.room, level, rng,
                               l1_range=l1_range, l1_u_shape=l1_u_shape).room
        if tgt_room.area <= 3.0:
            continue
        ratio = tgt_room.area / src_area
        if not (ratio_range[0] <= ratio <= ratio_range[1]):
            continue
        gt = motif_rigid_warp(scene, tgt_room, clip_inside=True)
        # reject centre-OOB collapses, footprint overhang (objects sticking
        # through a slanted / concave / corner-cut wall), AND any deform that
        # introduces collisions beyond what the real reference already had --
        # the rigid transplant must not pack furniture into each other.
        score = max(_gt_oob_frac(gt, tgt_room),
                    _gt_overhang_frac(gt, tgt_room, max_overhang))
        new_coll = max(0, real_collisions(gt) - src_coll)
        if score <= max_oob and new_coll == 0:
            return scene, tgt_room, gt
        key = (new_coll, score)              # prefer fewest new collisions
        if best is None or key < best[0]:
            best = (key, tgt_room, gt)
    # fallback: clean bounded uniform scale (area ratio at the range midpoint)
    from ..geom.deform import uniform_scale
    from ..geom.deform import _anchor_openings, _replace_openings
    from ..core.scene import Room
    r_mid = math.sqrt((ratio_range[0] * ratio_range[1]))
    s = math.sqrt(r_mid)                    # linear factor -> area ratio r_mid
    poly = uniform_scale(scene.room.polygon, s)
    a = _anchor_openings(scene.room)
    troom = Room(polygon=poly, height=scene.room.height,
                 openings=_replace_openings(poly, a, len(scene.room.polygon)),
                 room_type=scene.room.room_type)
    gt = motif_rigid_warp(scene, troom, clip_inside=True)
    fb_score = max(_gt_oob_frac(gt, troom), _gt_overhang_frac(gt, troom, max_overhang))
    fb_key = (max(0, real_collisions(gt) - src_coll), fb_score)
    if best is not None and best[0] < fb_key:      # a tried deform was cleaner
        _, troom, gt = best
    return scene, troom, gt


def _obj_overhang(o, poly) -> float:
    """Fraction of an object's footprint lying outside the room polygon."""
    from ..geom.polygon import object_polygon
    fp = object_polygon(o); a = fp.area + 1e-9
    return fp.difference(poly).area / a


def _overlap_score(objs, poly) -> dict:
    """Per-object physical-infeasibility score: max(footprint overhang,
    normalised overlap with any non-nestable neighbour).  Pure geometry — this
    is what makes an object *not fit*, independent of any hand priority."""
    from ..geom.polygon import object_polygon, boxes_overlap_3d
    from .guidance import NESTABLE_PAIRS
    polys = [object_polygon(o) for o in objs]
    score = {}
    for i, oi in enumerate(objs):
        s = _obj_overhang(oi, poly)
        ai = polys[i].area + 1e-9
        for j, oj in enumerate(objs):
            if i == j:
                continue
            if frozenset({oi.category, oj.category}) in NESTABLE_PAIRS:
                continue
            if not boxes_overlap_3d(oi, oj):
                continue
            inter = polys[i].intersection(polys[j]).area
            s = max(s, inter / ai)
        score[oi.oid] = s
    return score


def make_forward_pair_drop(scene: Scene, rng, l1_range=(0.30, 0.62),
                           max_overhang: float = 0.15, tries: int = 6):
    """D1 drop-supervision pair: deform into a room too SMALL for the full set,
    then remove objects by *physical infeasibility* (footprint poke-out /
    non-nestable overlap) until the layout fits.  Removal is geometry-driven,
    NOT the hand-written importance table — so the flow learns to amortise
    feasibility-based pruning, not to imitate the greedy heuristic.  Semantic
    anchors (bed/sofa) are protected from removal.  Returns
    (source_scene, target_room, gt_scene_with_drops)."""
    from ..geom.deform import deform_room
    from ..retarget.summarize import _anchor_indices
    src_area = as_polygon(scene.room).area
    src_coll = real_collisions(scene)
    anchor_oids = {scene.objects[i].oid for i in _anchor_indices(scene)}
    best = None
    for _ in range(tries):
        level = int(rng.choice((2, 3, 4, 5)))
        tgt_room = deform_room(scene.room, level, rng,
                               l1_range=l1_range, l1_u_shape=False).room
        if tgt_room.area <= 3.0:
            continue
        gt = motif_rigid_warp(scene, tgt_room, clip_inside=True)
        poly = as_polygon(tgt_room)
        # greedily drop the single worst-fitting non-anchor object until the
        # layout is feasible.  Drop key = per-object footprint overhang (O(N),
        # cheap) -- a physics criterion (poke-out), not the hand priority table.
        guard = 0
        while guard < len(gt.objects) + 2:
            guard += 1
            overs = [(_obj_overhang(o, poly), o.oid) for o in gt.objects
                     if o.keep and o.oid not in anchor_oids]
            worst = max(overs, default=(0.0, None))
            if worst[0] <= max_overhang or worst[1] is None:
                break
            gt.objects = [o for o in gt.objects if o.oid != worst[1]]
        dropped = len(scene.objects) - len(gt.objects)
        over = _gt_overhang_frac(gt, tgt_room, max_overhang)
        new_coll = max(0, real_collisions(gt) - src_coll)
        key = (over, new_coll, -dropped)
        if best is None or key < best[0]:
            best = (key, tgt_room, gt)
        if over <= max_overhang and new_coll == 0 and dropped >= 1:
            return scene, tgt_room, gt
    _, tgt_room, gt = best
    return scene, tgt_room, gt


# ---------------------------------------------------------------------------
# HybridPairs dataset — the 70/30 pipeline the flow actually trains on.
# ---------------------------------------------------------------------------
try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:                                   # torch optional at import
    _TorchDataset = object


class HybridPairs(_TorchDataset):
    """70% forward-deform + 30% filtered cross-pairing training pairs.

    Every item is a real reference retargeted to a target boundary with a
    plausible GT layout:
      * forward path — real scene as reference, its room deformed into the
        target boundary, GT produced by ``motif_rigid_warp`` (motifs kept
        rigid) so intent transfer is guaranteed;
      * cross path   — a same-type room with Jaccard>0.6 and anchor-orientation
        agreement supplies a *real* target boundary and *real* GT layout, with
        objects matched by the spatial Hungarian.
    A cross draw whose match/orientation filter fails falls back to the forward
    path, so every index always yields a valid item.
    """

    def __init__(self, scenes, pair_index: dict, forward_frac: float = 0.7,
                 elasticity=None, levels=(1, 2, 3, 4, 5),
                 l1_range=(0.5, 2.0), l1_u_shape: bool = True,
                 max_deg: float = 30.0, seed: int = 0, cache: bool = False,
                 drop_frac: float = 0.0):
        from ..intent.elasticity import PriorElasticity
        self.scenes = scenes
        self.pair_index = pair_index
        self.forward_frac = forward_frac
        self.drop_frac = drop_frac       # D1: portion of forward pairs that are
                                         # shrink-hard with physics-based drops
        self._elast = elasticity or PriorElasticity()
        self.levels = tuple(levels)
        self.l1_range = l1_range
        self.l1_u_shape = l1_u_shape
        self.max_deg = max_deg
        self.seed = seed
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.scenes)

    def _build_item(self, source, target_room, gt):
        from ..retarget.target import build_design_intent
        from .tokens import build_tokens
        graph = build_motifs(build_scene_graph(source))
        intent = build_design_intent(graph, target_room, elasticity=self._elast)
        return build_tokens(intent, target_room, gt)

    def sample_triplet(self, idx: int):
        """Return (kind, source_scene, target_room, gt_scene) without tokenising
        — used by the pipeline-verification renderer."""
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2 ** 32))
        base = self.scenes[idx]
        use_cross = (idx in self.pair_index) and (rng.random() >= self.forward_frac)
        if use_cross:
            partners = self.pair_index[idx]
            tgt = self.scenes[int(rng.choice(partners))]
            trip = make_cross_pair_filtered(base, tgt, self.max_deg)
            if trip is not None:
                ref, troom, gt = trip
                return "cross", ref, troom, gt
        # D1 drop path: a portion of forward draws shrink hard and drop
        # physically-infeasible objects, giving the mask-flow its negatives.
        if getattr(self, "drop_frac", 0.0) > 0.0 and rng.random() < self.drop_frac:
            try:
                src, troom, gt = make_forward_pair_drop(base, rng)
                return "drop", src, troom, gt
            except Exception:
                pass                      # fall through to the safe forward path
        # forward path (default / fallback)
        src, troom, gt = make_forward_pair(base, rng, self.levels,
                                           self.l1_range, self.l1_u_shape)
        return "forward", src, troom, gt

    def __getitem__(self, idx: int):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        _, source, troom, gt = self.sample_triplet(idx)
        item = self._build_item(source, troom, gt)
        if self._cache is not None:
            self._cache[idx] = item
        return item
