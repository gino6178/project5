"""3D-FRONT parser -> ReRoom ``Scene`` objects (plan section 3.1).

3D-FRONT is the main layout supervision: professionally designed rooms with
furniture semantics, poses and 3D-FUTURE assets.  Two details matter and are
handled explicitly here:

* **Coordinates.**  3D-FRONT is y-up.  ReRoom is z-up with the floor in xy, so
  ``(x, y, z)_front -> (x, -z, y)_reroom``; that mapping is right-handed, and a
  rotation about 3D-FRONT's +y becomes a rotation about ReRoom's +z with the
  *same* angle.  Child rotations are quaternions in ``[x, y, z, w]`` order and
  are, in 99.8 % of cases, pure yaw.
* **Floor polygons.**  The room boundary is taken from the union of its
  ``Floor`` meshes, not from a bounding box, so L-shaped, slanted and concave
  rooms survive parsing.  This is what makes 3D-FRONT usable for the plan's
  irregular-room experiments at all.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..core.categories import canonical_category, canonical_room_type
from ..core.scene import ObjectInstance, Opening, Room, Scene
from ..geom.polygon import normalize_polygon

__all__ = ["parse_scene_file", "load_bboxes", "ROOM_WHITELIST",
           "FRONT_AXIS_OFFSET", "front_to_reroom"]

ROOM_WHITELIST = ("Bedroom", "MasterBedroom", "SecondBedroom", "KidsRoom",
                  "ElderlyRoom", "NannyRoom", "LivingRoom", "LivingDiningRoom",
                  "DiningRoom", "Library")

# 3D-FUTURE models face their local +z, which becomes ReRoom's local -y once
# the y-up -> z-up mapping is applied, so the parsed yaw needs a half turn.
# Measured, not assumed: `scripts/check_front_axis.py` reports that with this
# offset 93.2 % of wall-backed objects (wardrobes, TV stands, beds) face into
# the room, versus 6.0 % without it.
FRONT_AXIS_OFFSET = math.pi


def front_to_reroom(p: np.ndarray) -> np.ndarray:
    """(x, y, z) y-up  ->  (x, -z, y) z-up."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return np.array([p[0], -p[2], p[1]])
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def _yaw_from_quat(q) -> float:
    """Yaw about the up axis from a ``[x, y, z, w]`` quaternion."""
    x, y, z, w = [float(v) for v in q]
    # general formula, exact for pure-yaw and a sane projection otherwise
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))


def load_bboxes(path: str) -> dict:
    """``{model_id: [minx, miny, minz, maxx, maxy, maxz]}`` in model space."""
    with open(path) as fh:
        d = json.load(fh)
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}


def _floor_polygon(meshes: list[dict]) -> Polygon | None:
    """Union the room's floor triangles and take the largest resulting face."""
    tris = []
    for m in meshes:
        v = np.asarray(m["xyz"], dtype=float).reshape(-1, 3)
        f = np.asarray(m["faces"], dtype=int).reshape(-1, 3)
        pts = np.stack([v[:, 0], -v[:, 2]], axis=1)
        for tri in f:
            p = Polygon(pts[tri])
            if p.is_valid and p.area > 1e-6:
                tris.append(p)
            elif p.area > 1e-6:
                p = p.buffer(0)
                if p.area > 1e-6:
                    tris.append(p)
    if not tris:
        return None
    u = unary_union(tris).buffer(1e-4).buffer(-1e-4)
    if u.is_empty:
        return None
    if isinstance(u, MultiPolygon):
        u = max(u.geoms, key=lambda g: g.area)
    if u.geom_type != "Polygon":
        return None
    return u


def _simplify_polygon(poly: Polygon, tol: float = 0.06) -> np.ndarray | None:
    p = poly.simplify(tol, preserve_topology=True)
    if p.is_empty or p.geom_type != "Polygon":
        return None
    pts = np.asarray(p.exterior.coords)[:-1]
    if len(pts) < 3:
        return None
    return normalize_polygon(pts)


def _opening_from_mesh(m: dict, kind: str) -> Opening | None:
    v = np.asarray(m["xyz"], dtype=float).reshape(-1, 3)
    if len(v) < 3:
        return None
    pts = np.stack([v[:, 0], -v[:, 2]], axis=1)
    c = pts.mean(0)
    d = pts - c
    if float(np.abs(d).max()) < 1e-6:
        return None
    # principal direction of the opening's footprint
    u, s, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
    axis = vt[0]
    proj = d @ axis
    p0 = c + axis * proj.min()
    p1 = c + axis * proj.max()
    if float(np.linalg.norm(p1 - p0)) < 0.25:
        return None
    return Opening(kind=kind, p0=p0, p1=p1,
                   z0=float(v[:, 1].min()), z1=float(v[:, 1].max()))


# 3D-FRONT's furniture list also holds architecture that is not furniture.
_SKIP_TITLE = re.compile(
    r"curtain|door/|window/|kitchen cabinet|/cbnt|ceiling|baseboard|"
    r"flue|hole|slab|beam|column|pipe|appliance/range hood", re.I)


def parse_scene_file(path: str, bboxes: dict,
                     categories: dict | None = None,
                     room_whitelist=ROOM_WHITELIST,
                     min_objects: int = 4, max_objects: int = 40,
                     min_area: float = 5.0, max_area: float = 60.0,
                     simplify_tol: float = 0.06) -> list[Scene]:
    """Parse one 3D-FRONT house into per-room ReRoom scenes."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:
        return []

    furn = {f["uid"]: f for f in d.get("furniture", [])}
    meshes = {m["uid"]: m for m in d.get("mesh", [])}
    house = os.path.splitext(os.path.basename(path))[0]
    out: list[Scene] = []

    for room in d.get("scene", {}).get("room", []):
        rtype = room.get("type", "")
        if room_whitelist and not any(w.lower() in rtype.lower()
                                      for w in room_whitelist):
            continue
        floors, objs, openings = [], [], []
        wall_h = []
        for child in room.get("children", []):
            ref = child.get("ref")
            if ref in meshes:
                m = meshes[ref]
                t = m.get("type", "")
                if t == "Floor":
                    floors.append(m)
                elif t in ("Door", "Window"):
                    op = _opening_from_mesh(m, t.lower())
                    if op is not None:
                        openings.append(op)
                elif t in ("WallInner", "WallOuter"):
                    v = np.asarray(m["xyz"], dtype=float).reshape(-1, 3)
                    if len(v):
                        wall_h.append(float(v[:, 1].max()))
                continue
            f = furn.get(ref)
            if f is None or not f.get("jid"):
                continue
            bb = bboxes.get(f["jid"])
            if bb is None:
                continue
            scale = np.asarray(child.get("scale", [1, 1, 1]), dtype=float)
            lo, hi = bb[:3] * scale, bb[3:] * scale
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            size_f = hi - lo                       # y-up extents
            centre_local = (lo + hi) / 2.0
            yaw = _yaw_from_quat(child.get("rot", [0, 0, 0, 1]))
            c, s = math.cos(yaw), math.sin(yaw)
            # rotate the local centre about the up axis, then translate
            cx = c * centre_local[0] + s * centre_local[2]
            cz = -s * centre_local[0] + c * centre_local[2]
            pos = np.asarray(child.get("pos", [0, 0, 0]), dtype=float)
            world = pos + np.array([cx, centre_local[1], cz])
            p = front_to_reroom(world)
            size = np.array([size_f[0], size_f[2], size_f[1]])   # (sx, sz, sy)
            if float(size.min()) <= 1e-2 or float(size.max()) > 8.0:
                continue
            if not (np.all(np.isfinite(size)) and np.all(np.isfinite(p))
                    and math.isfinite(yaw)):
                continue
            title = f.get("title") or ""
            if _SKIP_TITLE.search(title):
                continue
            # the authoritative label is 3D-FUTURE's own category for the jid;
            # the 3D-FRONT title ("bed/king-size bed") is the fallback
            raw = (categories or {}).get(f["jid"]) or title
            cat = canonical_category(raw, title.split("/")[0] if title else None)
            if cat == "misc" and not raw:
                continue
            objs.append(ObjectInstance(
                oid=f"{child.get('instanceid', ref)}",
                category=cat,
                position=np.array([p[0], p[1], p[2] - size[2] / 2.0]),
                yaw=yaw + FRONT_AXIS_OFFSET,
                size=size, jid=f["jid"], raw_category=raw,
                meta={"instanceid": child.get("instanceid")}))

        if len(objs) < min_objects:
            continue
        fp = _floor_polygon(floors)
        if fp is None or not (min_area <= fp.area <= max_area):
            continue
        poly = _simplify_polygon(fp, simplify_tol)
        if poly is None or len(poly) > 24:
            continue
        if len(objs) > max_objects:
            objs.sort(key=lambda o: -o.footprint_area)
            objs = objs[:max_objects]

        height = float(np.median(wall_h)) if wall_h else 2.8
        r = Room(polygon=poly, height=float(np.clip(height, 2.2, 3.6)),
                 openings=openings, room_type=canonical_room_type(rtype))
        # shape descriptors are stored so experiments can stratify by how
        # irregular the *source* room already is
        sp = Polygon(poly)
        hull_a = float(sp.convex_hull.area)
        mrr_a = float(sp.minimum_rotated_rectangle.area)
        sc = Scene(scene_id=f"{house}__{room.get('instanceid', rtype)}",
                   room=r, objects=objs, source="3D-FRONT",
                   meta={"house": house, "raw_room_type": rtype,
                         "convexity": float(sp.area / hull_a) if hull_a > 1e-9 else 1.0,
                         "rect_fill": float(sp.area / mrr_a) if mrr_a > 1e-9 else 1.0,
                         "n_vertices": int(len(poly))})
        out.append(sc)
    return out
