"""Regularity projector (ReRoom 2.0 Step 1): snap a polished layout to the
orthogonal / flush / slotted structure human eyes expect.

Continuous flow + MSE loss tolerate 5-10 cm non-orthogonal offsets that read as
"messy" to people.  This runs after polish (it is a projection in the eq. (37)
sense) and enforces three hard regularities without retraining:

  * slot attraction  -- dining_chair / office_chair snap to the nearest
    table/desk edge, facing the table (motif tear -> 0, no free drift at 1.35x);
  * wall flush + normal orientation -- wall-hugging items (wall_art, wardrobe,
    tv_stand, cabinets, bed, sofa, ...) get their back edge pushed onto the
    nearest wall and their facing set to that wall's inward normal;
  * Manhattan orientation snap -- free-standing orthogonal items (coffee_table,
    dining_table, side_table) snap yaw to the nearest wall axis when within a
    tolerance, killing the 3-15 deg "tipsy" tilt.
"""
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from ..geom.polygon import as_polygon, object_polygon

# wall-hugging: back flush to wall, forward = inward normal
WALL_CATS = frozenset({
    "wall_art", "mirror", "wardrobe", "tv_stand", "tv", "sideboard", "cabinet",
    "bookcase", "shelf", "console_table", "drawer_chest", "shoe_cabinet",
    "wine_cabinet", "double_bed", "single_bed", "kids_bed", "bunk_bed",
    "sofa", "l_sofa", "desk", "dressing_table", "piano", "fireplace",
})
# free-standing but should stay orthogonal to the room axes
ORTHO_CATS = frozenset({"coffee_table", "dining_table", "side_table", "bench"})
# slot children -> snap to their parent's edge
SLOT_PARENTS = {"dining_chair": ("dining_table",),
                "office_chair": ("desk",),
                "stool": ("dining_table", "console_table", "desk")}
# genuinely free: never touched
FREE_CATS = frozenset({
    "armchair", "lounge_chair", "loveseat", "barstool", "plant", "decoration",
    "misc", "rug", "ceiling_lamp", "floor_lamp", "table_lamp", "pendant_lamp",
    "nightstand",   # nightstand rides its bed via the flow; snapping fights it
})


def _wall_segments(room):
    poly = as_polygon(room)
    ring = np.asarray(poly.exterior.coords)[:-1]
    n = len(ring)
    area = 0.5 * float(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1)
                              - np.roll(ring[:, 0], -1) * ring[:, 1]))
    orient = 1.0 if area > 0 else -1.0
    segs = []
    for i in range(n):
        a = ring[i]; b = ring[(i + 1) % n]
        d = b - a; L = float(np.linalg.norm(d)) + 1e-9; t = d / L
        nrm = np.array([-t[1] * orient, t[0] * orient])   # inward normal
        segs.append((a, b, t, nrm, L))
    return segs


def _nearest_wall(xy, segs):
    best = None; bestd = 1e18
    for a, b, t, nrm, L in segs:
        rel = xy - a; proj = float(np.dot(rel, t))
        pc = min(max(proj, 0.0), L)
        foot = a + pc * t
        d = float(np.linalg.norm(xy - foot))
        if d < bestd:
            bestd = d; best = (a, b, t, nrm, L, foot)
    return best, bestd


def _yaw_from_forward(fwd):
    # forward = (-sin yaw, cos yaw)  ->  yaw = atan2(-fx, fy)
    return math.atan2(-float(fwd[0]), float(fwd[1]))


def regularity_snap(scene, intent=None, ortho_tol_deg: float = 22.0,
                    wall_max: float = 0.9):
    """Snap ``scene`` in place.  ``wall_max`` (m) gates wall flush: only objects
    whose nearest wall is within this distance are treated as wall-hugging."""
    room = scene.room
    segs = _wall_segments(room)
    objs = [o for o in scene.objects if o.keep]
    by_cat = {}
    for o in objs:
        by_cat.setdefault(o.category, []).append(o)

    # ---- 1. slot children -> parent edge ------------------------------------
    for o in objs:
        parents = SLOT_PARENTS.get(o.category)
        if not parents:
            continue
        cands = [p for pc in parents for p in by_cat.get(pc, [])]
        if not cands:
            continue
        par = min(cands, key=lambda p: float(np.linalg.norm(p.xy - o.xy)))
        depth = float(o.half[1])                     # chair half-depth
        tpoly = object_polygon(par, inflate=max(depth, 0.12))
        pt = nearest_points(tpoly.exterior, Point(float(o.xy[0]), float(o.xy[1])))[0]
        o.xy = np.array([pt.x, pt.y])
        # face the table ALONG the table's own axes, not toward its centre --
        # so chairs tuck in squarely (parallel to the table edge) instead of
        # sitting at a diagonal.  Snap the chair->table direction to the nearest
        # of {+/- table.forward, +/- table.right}.
        face = par.xy - o.xy
        nf = np.linalg.norm(face)
        if nf > 1e-6:
            face = face / nf
            axes = [par.forward, -par.forward, par.right, -par.right]
            best_ax = max(axes, key=lambda ax: float(np.dot(face, ax)))
            o.yaw = _yaw_from_forward(best_ax)

    # ---- 2. wall-hugging: normal orientation + back flush -------------------
    for o in objs:
        if o.category not in WALL_CATS:
            continue
        (a, b, t, nrm, L, foot), d = _nearest_wall(o.xy, segs)
        if d > wall_max:
            continue
        # orientation: forward = inward normal (back against wall)
        o.yaw = _yaw_from_forward(nrm)
        # flush: push so the rear-most corner sits on the wall line
        corners = o.corners()
        proj = corners @ nrm
        back = float(proj.min())
        wall_line = float(np.dot(foot, nrm))
        o.xy = o.xy + (wall_line - back) * nrm

    # ---- 3. Manhattan snap for free-standing orthogonal items ---------------
    for o in objs:
        if o.category not in ORTHO_CATS:
            continue
        (a, b, t, nrm, L, foot), d = _nearest_wall(o.xy, segs)
        fwd = o.forward
        cands = [nrm, -nrm, np.array([-nrm[1], nrm[0]]), np.array([nrm[1], -nrm[0]])]
        best_c = max(cands, key=lambda c: float(np.dot(fwd, c)))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(fwd, best_c))))))
        if ang <= ortho_tol_deg:
            o.yaw = _yaw_from_forward(best_c)
    return scene
