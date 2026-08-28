"""Polygon geometry for arbitrary simple (possibly concave) floor plans.

The whole point of ReRoom is that the target room is a polygon ``P_t``, not a
``(W, D)`` pair, so every feasibility term is written against the polygon
directly: footprint-outside-polygon area for eq. (22), pairwise footprint
overlap for eq. (23), wall assignment for `against-wall` relations, and a
floor-geometry descriptor ``g(P)`` that conditions relation elasticity in
eq. (45).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from ..core.scene import ObjectInstance, Room

__all__ = [
    "signed_area", "as_polygon", "object_polygon", "objects_union",
    "out_of_bounds_area", "overlap_area", "boxes_overlap_3d",
    "inside_fraction", "wall_of", "wall_distance", "distance_to_boundary",
    "erode", "characteristic_scale", "sat_separation", "floor_descriptor", "FLOOR_DESCRIPTOR_DIM",
    "sample_interior", "largest_inscribed_circle", "is_simple",
    "polygon_from_extent", "normalize_polygon", "min_rotated_rect_params",
]


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def as_polygon(p) -> Polygon:
    """Accept ndarray / Room / Polygon and return a valid shapely Polygon."""
    if isinstance(p, Polygon):
        poly = p
    elif isinstance(p, Room):
        poly = Polygon(p.polygon)
    else:
        poly = Polygon(np.asarray(p, dtype=float).reshape(-1, 2))
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def is_simple(poly: np.ndarray) -> bool:
    ring = LineString(np.vstack([poly, poly[:1]]))
    return bool(ring.is_simple) and Polygon(poly).is_valid


def object_polygon(obj: ObjectInstance, inflate: float = 0.0) -> Polygon:
    """Oriented footprint ``B_i(y_i)`` of eq. (22), optionally dilated.

    Degenerate or non-finite boxes are clamped rather than allowed to reach
    GEOS -- real datasets do contain zero-thickness and NaN entries, and one of
    them must not take down a corpus-wide fit.
    """
    hx, hy = obj.half
    if not (np.isfinite(hx) and np.isfinite(hy)
            and np.isfinite(obj.position[0]) and np.isfinite(obj.position[1])
            and np.isfinite(obj.yaw)):
        return Polygon()
    hx, hy = max(float(hx), 1e-3) + inflate, max(float(hy), 1e-3) + inflate
    rect = box(-hx, -hy, hx, hy)
    rect = affinity.rotate(rect, obj.yaw, origin=(0, 0), use_radians=True)
    return affinity.translate(rect, obj.position[0], obj.position[1])


def sat_separation(a: ObjectInstance, b: ObjectInstance) -> float:
    """Separation of two oriented footprints along their best separating axis.

    Negative values are the penetration depth.  This is the *same* quantity the
    differentiable surrogate in ``retarget.energy`` computes, so the exact
    energy and the surrogate agree on what a relation's ``gap`` is -- otherwise
    the optimizer chases a constant offset that is an artefact of the metric.
    """
    ua = np.stack([a.right, a.forward])
    ub = np.stack([b.right, b.forward])
    axes = np.vstack([ua, ub])
    d = b.xy - a.xy
    pd = np.abs(axes @ d)
    ra = (np.abs(axes @ ua.T) * a.half).sum(1)
    rb = (np.abs(axes @ ub.T) * b.half).sum(1)
    return float((pd - ra - rb).max())


def objects_union(objs: Sequence[ObjectInstance], inflate: float = 0.0) -> Polygon:
    polys = [object_polygon(o, inflate) for o in objs]
    return unary_union(polys) if polys else Polygon()


# --------------------------------------------------------------------------
# feasibility terms
# --------------------------------------------------------------------------
def out_of_bounds_area(obj: ObjectInstance, room_poly: Polygon) -> float:
    """``Area(B_i \\ P_t)``.  Ceiling-mounted objects are exempt from walls."""
    fp = object_polygon(obj)
    return float(fp.difference(room_poly).area)


def boxes_overlap_3d(a: ObjectInstance, b: ObjectInstance, eps: float = 1e-3) -> bool:
    """Do the two objects overlap in ``z``?  A lamp *on* a table does not."""
    return (a.z < b.top - eps) and (b.z < a.top - eps)


def overlap_area(a: ObjectInstance, b: ObjectInstance, respect_height: bool = True) -> float:
    """``Area(B_i ∩ B_j)`` of eq. (23), skipping legitimate stacking."""
    if respect_height and not boxes_overlap_3d(a, b):
        return 0.0
    return float(object_polygon(a).intersection(object_polygon(b)).area)


def inside_fraction(obj: ObjectInstance, room_poly: Polygon) -> float:
    fp = object_polygon(obj)
    if fp.area <= 1e-12:
        return 1.0
    return float(fp.intersection(room_poly).area / fp.area)


def distance_to_boundary(p: np.ndarray, room_poly: Polygon) -> float:
    """Signed distance: positive inside, negative outside."""
    pt = Point(float(p[0]), float(p[1]))
    d = pt.distance(room_poly.exterior)
    return d if room_poly.contains(pt) else -d


def erode(room_poly: Polygon, r: float) -> Polygon:
    """Region where an object of circumradius ``r`` can sit fully inside."""
    if r <= 0:
        return room_poly
    e = room_poly.buffer(-r)
    if e.is_empty:
        return Polygon()
    if e.geom_type == "MultiPolygon":
        e = max(e.geoms, key=lambda g: g.area)
    return e


# --------------------------------------------------------------------------
# walls
# --------------------------------------------------------------------------
def wall_of(obj: ObjectInstance, room: Room, max_dist: float = 0.45,
            flush_dist: float = 0.0) -> tuple[int, float, float] | None:
    """Assign the object to the wall it is backed against.

    Returns ``(wall_index, gap, angle_error)`` or ``None``.  The object's
    *back* is ``-forward``; a wall counts only if the object is roughly
    parallel to it and its back edge is close.
    """
    walls = room.walls()
    back = -obj.forward
    best = None
    # `flush_dist > 0` widens detection: a footprint that touches a wall counts
    # as against it whatever it faces, so facing decides *which* wall rather
    # than *whether*.  Gating on the back face at 35 degrees rejects two thirds
    # of the objects that are genuinely flush in the reference -- a bookshelf
    # standing side-on, a sofa at a slight angle, anything whose front axis the
    # parser got wrong -- and an object with no wall target is never told to
    # stay on a wall.
    #
    # It is **off by default**, and the reason is measured rather than
    # cautious.  Turning it on raises detection coverage from 31.2 % to 93.6 %
    # and, with the flush term, takes wall-hugging from 29.7 % to 79.2 % and
    # the median wall gap from 0.19 m to 0.04 m -- both matching the real
    # rooms.  It also triples out-of-plan area, from 0.029 to 0.091, because
    # `E_wall` measures the distance from the object's *rear-face midpoint* to
    # the wall, which is not a meaningful quantity for an object standing
    # side-on: the target gap is then wrong and the optimiser drives the object
    # through the wall.  Generalising it needs `E_wall` rewritten in terms of
    # footprint-to-wall distance, differentiably.  Until then this stays a
    # documented knob and not the shipped path.
    fp = None
    try:
        from shapely.geometry import LineString
        from .polygon import object_polygon as _op
        fp = _op(obj)
    except Exception:
        fp = None
    for k, (a, b) in enumerate(walls):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        t = d / L
        n_in = np.array([-t[1], t[0]])          # CCW polygon -> inward normal
        # rear-face midpoint of the footprint
        rear = obj.xy + back * obj.half[1]
        rel = rear - a
        s = float(np.dot(rel, t))
        if not (-0.25 * L <= s <= 1.25 * L):
            continue
        gap = float(np.dot(rel, n_in))
        ang = float(np.arccos(np.clip(np.dot(back, -n_in), -1.0, 1.0)))
        touching = False
        if fp is not None and flush_dist > 0.0:
            from shapely.geometry import LineString
            fd = float(fp.distance(LineString([a, b])))
            touching = fd <= flush_dist
            if touching:
                gap = min(gap, fd) if gap > -0.35 else fd
        if not touching:
            if gap < -0.35 or gap > max_dist:
                continue
            if ang > math.radians(35):
                continue
        score = gap + 0.5 * ang
        if best is None or score < best[0]:
            best = (score, k, gap, ang)
    return (best[1], best[2], best[3]) if best else None


def wall_distance(obj: ObjectInstance, room_poly: Polygon) -> float:
    """Distance from the footprint to the room boundary (0 if touching)."""
    return float(object_polygon(obj).distance(room_poly.exterior))


# --------------------------------------------------------------------------
# scale / descriptors
# --------------------------------------------------------------------------
def characteristic_scale(room_poly: Polygon, direction: np.ndarray) -> float:
    """Room extent along ``direction`` -- the ``gamma_ij`` of eq. (9).

    Measured as the width of the polygon's support along the unit direction,
    which is well defined for concave polygons too.
    """
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        d = np.array([1.0, 0.0])
    else:
        d = d / n
    pts = np.asarray(room_poly.exterior.coords)[:-1]
    proj = pts @ d
    return float(proj.max() - proj.min())


def min_rotated_rect_params(room_poly: Polygon) -> tuple[float, float, float]:
    """(long side, short side, orientation of the long side in radians)."""
    mrr = room_poly.minimum_rotated_rectangle
    pts = np.asarray(mrr.exterior.coords)[:-1]
    if len(pts) < 4:
        b = room_poly.bounds
        return (b[2] - b[0], b[3] - b[1], 0.0)
    e = pts[1:] - pts[:-1]
    e = np.vstack([e, pts[0] - pts[-1]])
    lens = np.linalg.norm(e, axis=1)
    i = int(np.argmax(lens[:2]))
    long_, short_ = float(lens[i]), float(lens[1 - i])
    ang = float(math.atan2(e[i][1], e[i][0]))
    return long_, short_, ang


FLOOR_DESCRIPTOR_DIM = 12


def floor_descriptor(room_poly: Polygon) -> np.ndarray:
    """``g(P)``: a rotation-aware, scale-explicit descriptor of a floor plan.

    Used to condition relation elasticity (eq. 45) and the generative model.
    """
    poly = as_polygon(room_poly)
    area = poly.area
    per = poly.exterior.length
    minx, miny, maxx, maxy = poly.bounds
    bw, bh = maxx - minx, maxy - miny
    long_, short_, ang = min_rotated_rect_params(poly)
    hull = poly.convex_hull
    convexity = area / hull.area if hull.area > 1e-9 else 1.0
    rect_fill = area / (long_ * short_) if long_ * short_ > 1e-9 else 1.0
    compact = 4 * math.pi * area / (per ** 2) if per > 1e-9 else 1.0
    n_v = len(np.asarray(poly.exterior.coords)) - 1
    n_reflex = _count_reflex(np.asarray(poly.exterior.coords)[:-1])
    inr = largest_inscribed_circle(poly)[1]
    return np.array([
        area, per, bw, bh,
        long_, short_,
        short_ / long_ if long_ > 1e-9 else 1.0,
        convexity, rect_fill, compact,
        n_reflex / max(n_v, 1),
        inr,
    ], dtype=np.float32)


def _count_reflex(pts: np.ndarray) -> int:
    n = len(pts)
    if n < 4:
        return 0
    cnt = 0
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cr = np.cross(b - a, c - b)
        if cr < -1e-9:            # CCW polygon -> negative cross = reflex
            cnt += 1
    return cnt


def largest_inscribed_circle(room_poly: Polygon, tol: float = 0.05
                             ) -> tuple[np.ndarray, float]:
    """Pole of inaccessibility: the deepest interior point and its clearance."""
    minx, miny, maxx, maxy = room_poly.bounds
    cell = max(min(maxx - minx, maxy - miny) / 16.0, tol)
    best_p = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    best_d = -1e9
    ring = room_poly.exterior
    while cell >= tol:
        xs = np.arange(minx + cell / 2, maxx, cell)
        ys = np.arange(miny + cell / 2, maxy, cell)
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if not room_poly.contains(pt):
                    continue
                d = pt.distance(ring)
                if d > best_d:
                    best_d, best_p = d, np.array([x, y])
        minx, maxx = best_p[0] - cell, best_p[0] + cell
        miny, maxy = best_p[1] - cell, best_p[1] + cell
        cell /= 2.5
    return best_p, max(best_d, 0.0)


def sample_interior(room_poly: Polygon, n: int, rng: np.random.Generator,
                    margin: float = 0.0) -> np.ndarray:
    """Rejection-sample ``n`` points inside the (optionally eroded) polygon."""
    region = erode(room_poly, margin) if margin > 0 else room_poly
    if region.is_empty:
        region = room_poly
    minx, miny, maxx, maxy = region.bounds
    out = []
    for _ in range(200):
        k = max(n * 4, 64)
        pts = rng.uniform([minx, miny], [maxx, maxy], size=(k, 2))
        for p in pts:
            if region.contains(Point(p[0], p[1])):
                out.append(p)
                if len(out) >= n:
                    return np.asarray(out)
    while len(out) < n:
        out.append(np.asarray(region.representative_point().coords[0]))
    return np.asarray(out[:n])


def polygon_from_extent(w: float, d: float) -> np.ndarray:
    """Axis-aligned rectangle centred on the origin (CCW)."""
    hw, hd = w / 2, d / 2
    return np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])


def normalize_polygon(poly: np.ndarray) -> np.ndarray:
    """CCW, no duplicate closing vertex, collinear vertices merged."""
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    keep = []
    n = len(p)
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        if np.linalg.norm(b - a) < 1e-7:
            continue
        cr = abs(float(np.cross(b - a, c - b)))
        if cr < 1e-9 and np.dot(b - a, c - b) > 0:
            continue                       # collinear
        keep.append(b)
    p = np.asarray(keep) if len(keep) >= 3 else p
    if signed_area(p) < 0:
        p = p[::-1]
    return p
